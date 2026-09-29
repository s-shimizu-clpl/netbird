package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ProviderNameGemini is the stable label for the Google Gemini parser, used
// as the llm.provider metadata value and the cost-meter formula selector.
const ProviderNameGemini = "gemini"

// GeminiParser implements the Parser interface for the Google Gemini API
// surface: generativelanguage.googleapis.com and the Vertex "google"
// publisher. Gemini carries the model in the URL path
// (/v1beta/models/{model}:generateContent); the request middleware extracts
// it there, so this parser focuses on the response shapes. The one exception
// is the interactions endpoint, whose body carries the model.
type GeminiParser struct{}

// geminiPathHints are substring patterns that mark a request as
// Gemini-shaped. The action suffixes are Gemini's API contract; the
// interactions endpoint is the only one without a path-embedded model.
// ":streamgeneratecontent" is listed even though ":generatecontent" is a
// substring of it: batch and stream actions do not contain the bare action,
// so each action is listed explicitly.
var geminiPathHints = []string{
	":generatecontent",
	":streamgeneratecontent",
	":batchgeneratecontent",
	"/interactions",
}

// Provider returns ProviderGemini.
func (GeminiParser) Provider() Provider { return ProviderGemini }

// ProviderName returns the stable label used for metrics and metadata.
func (GeminiParser) ProviderName() string { return ProviderNameGemini }

// DetectFromURL reports whether the path is a Gemini API endpoint. The match
// is case-insensitive and substring-based so a proxy prefix strip does not
// defeat detection.
func (GeminiParser) DetectFromURL(path string) bool {
	lower := strings.ToLower(path)
	for _, hint := range geminiPathHints {
		if strings.Contains(lower, hint) {
			return true
		}
	}
	return false
}

// ParseRequest reads the model from the interactions body, the only Gemini
// endpoint that carries it outside the URL path. For the
// :generateContent family the model and streaming flag live in the path and
// the request middleware extracts them via parseGeminiPath, so this returns
// empty facts.
func (GeminiParser) ParseRequest(body []byte) (RequestFacts, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return RequestFacts{}, fmt.Errorf("decode gemini request: %w: %v", ErrMalformedRequest, err)
	}
	return RequestFacts{Model: req.Model}, nil
}

// geminiUsageMetadata is Gemini's usage block. cachedContentTokenCount is a
// SUBSET of promptTokenCount (the portion served from an explicit
// cachedContent reference), so the cost meter bills it with the OpenAI-style
// carve-out rather than the Anthropic-style additive buckets.
// thoughtsTokenCount carries reasoning tokens, which Google bills as output.
type geminiUsageMetadata struct {
	PromptTokenCount        int64 `json:"promptTokenCount"`
	CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int64 `json:"thoughtsTokenCount"`
	TotalTokenCount         int64 `json:"totalTokenCount"`
	CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
}

func (u geminiUsageMetadata) toUsage() Usage {
	total := u.TotalTokenCount
	usage := Usage{
		InputTokens:       u.PromptTokenCount,
		OutputTokens:      u.CandidatesTokenCount + u.ThoughtsTokenCount,
		TotalTokens:       total,
		CachedInputTokens: u.CachedContentTokenCount,
	}
	if usage.TotalTokens == 0 && (usage.InputTokens > 0 || usage.OutputTokens > 0) {
		usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	}
	return usage
}

// addUsage accumulates one response's usage into acc.
func addUsage(acc *Usage, u geminiUsageMetadata) {
	other := u.toUsage()
	acc.InputTokens += other.InputTokens
	acc.OutputTokens += other.OutputTokens
	acc.TotalTokens += other.TotalTokens
	acc.CachedInputTokens += other.CachedInputTokens
}

// geminiResponse accepts the three sync response shapes in one struct:
// generateContent (usageMetadata at the top level), batchGenerateContent
// (per-result usageMetadata under response.results[].response, with the
// aggregate mirrored at response.usageMetadata), and interactions (a
// snake_case usage block in the newer API's naming).
type geminiResponse struct {
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata"`
	Response      *struct {
		Results []struct {
			Response struct {
				UsageMetadata *geminiUsageMetadata `json:"usageMetadata"`
			} `json:"response"`
		} `json:"results"`
		UsageMetadata *geminiUsageMetadata `json:"usageMetadata"`
	} `json:"response"`
	Usage *struct {
		InputTokens  *int64 `json:"input_tokens"`
		OutputTokens *int64 `json:"output_tokens"`
		TotalTokens  *int64 `json:"total_tokens"`
		Details      *struct {
			CachedTokens *int64 `json:"cached_tokens"`
		} `json:"input_tokens_details"`
	} `json:"usage"`
}

// ParseResponse decodes the non-streaming Gemini response envelope. Status
// codes other than 200 are treated as non-LLM responses so the caller can
// skip cost accounting without aborting the request.
func (GeminiParser) ParseResponse(status int, contentType string, body []byte) (Usage, error) {
	if status != 200 {
		return Usage{}, fmt.Errorf("gemini status %d: %w", status, ErrNotLLMResponse)
	}
	if isEventStream(contentType) {
		return Usage{}, ErrStreamingUnsupported
	}
	if !isJSON(contentType) {
		return Usage{}, fmt.Errorf("gemini content-type %q: %w", contentType, ErrNotLLMResponse)
	}

	var resp geminiResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return Usage{}, fmt.Errorf("decode gemini response: %w: %v", ErrMalformedResponse, err)
	}

	// Batch replies carry a mirror aggregate under response.usageMetadata.
	// Prefer summing the per-result blocks: the mirror is the provider's
	// arithmetic and a partial failure would leave it overstating what the
	// individual results report.
	var usage Usage
	if resp.Response != nil {
		for _, r := range resp.Response.Results {
			if r.Response.UsageMetadata != nil {
				addUsage(&usage, *r.Response.UsageMetadata)
			}
		}
		if usage.TotalTokens == 0 && usage.InputTokens == 0 && usage.OutputTokens == 0 {
			if resp.Response.UsageMetadata != nil {
				return resp.Response.UsageMetadata.toUsage(), nil
			}
		}
		return usage, nil
	}
	if resp.UsageMetadata != nil {
		return resp.UsageMetadata.toUsage(), nil
	}
	return interactionsUsage(resp.Usage), nil
}

// interactionsUsage maps the interactions endpoint's snake_case usage block.
// An explicit cached_tokens of 0 is honored over a missing details object,
// mirroring openAICachedTokens.
func interactionsUsage(usage *struct {
	InputTokens  *int64 `json:"input_tokens"`
	OutputTokens *int64 `json:"output_tokens"`
	TotalTokens  *int64 `json:"total_tokens"`
	Details      *struct {
		CachedTokens *int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}) Usage {
	if usage == nil {
		return Usage{}
	}
	u := Usage{
		InputTokens:  derefInt64(usage.InputTokens),
		OutputTokens: derefInt64(usage.OutputTokens),
		TotalTokens:  derefInt64(usage.TotalTokens),
	}
	if usage.Details != nil {
		u.CachedInputTokens = derefInt64(usage.Details.CachedTokens)
	}
	if u.TotalTokens == 0 && (u.InputTokens > 0 || u.OutputTokens > 0) {
		u.TotalTokens = u.InputTokens + u.OutputTokens
	}
	return u
}

// geminiPart is one content part of a Gemini message. Only text parts carry
// extractable content; image and inline-data parts are skipped.
type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

func contentsText(contents []geminiContent) string {
	var b strings.Builder
	for _, c := range contents {
		var text string
		for _, p := range c.Parts {
			if p.Text != "" {
				text += p.Text
			}
		}
		if text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		if c.Role != "" {
			b.WriteString(c.Role)
			b.WriteString(": ")
		}
		b.WriteString(text)
	}
	return b.String()
}

// ExtractPrompt returns the user-visible prompt from a Gemini request body.
// Handles generateContent (systemInstruction + contents[]) and the batch
// shape (batch.input_config.requests.requests[].request). Returns "" when
// nothing extractable is found.
func (GeminiParser) ExtractPrompt(body []byte) string {
	var req struct {
		Contents          []geminiContent `json:"contents"`
		SystemInstruction *struct {
			Parts []geminiPart `json:"parts"`
		} `json:"systemInstruction"`
		Batch *struct {
			InputConfig struct {
				Requests struct {
					Requests []struct {
						Request struct {
							Contents          []geminiContent `json:"contents"`
							SystemInstruction *struct {
								Parts []geminiPart `json:"parts"`
							} `json:"systemInstruction"`
						} `json:"request"`
					} `json:"requests"`
				} `json:"requests"`
			} `json:"input_config"`
		} `json:"batch"`
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}

	var b strings.Builder
	if req.Batch != nil {
		for _, item := range req.Batch.InputConfig.Requests.Requests {
			if s := geminiPromptText(item.Request.Contents, item.Request.SystemInstruction); s != "" {
				if b.Len() > 0 {
					b.WriteByte('\n')
				}
				b.WriteString(s)
			}
		}
		return b.String()
	}
	if len(req.Contents) > 0 || req.SystemInstruction != nil {
		return geminiPromptText(req.Contents, req.SystemInstruction)
	}
	if len(req.Input) > 0 {
		return extractContentParts(req.Input)
	}
	return ""
}

func geminiPromptText(contents []geminiContent, sysIns *struct {
	Parts []geminiPart `json:"parts"`
}) string {
	var b strings.Builder
	if sysIns != nil {
		var sys string
		for _, p := range sysIns.Parts {
			sys += p.Text
		}
		if sys != "" {
			b.WriteString("system: ")
			b.WriteString(sys)
		}
	}
	if s := contentsText(contents); s != "" {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(s)
	}
	return b.String()
}

// ExtractCompletion returns the assistant text from a non-streaming Gemini
// response: candidates[].content.parts[].text for generateContent, the same
// per result for batchGenerateContent.
func (GeminiParser) ExtractCompletion(status int, contentType string, body []byte) string {
	if status != 200 || isEventStream(contentType) || !isJSON(contentType) {
		return ""
	}
	var resp struct {
		Candidates []geminiCandidate `json:"candidates"`
		Response   *struct {
			Results []struct {
				Response struct {
					Candidates []geminiCandidate `json:"candidates"`
				} `json:"response"`
			} `json:"results"`
		} `json:"response"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return ""
	}
	var b strings.Builder
	appendText := func(text string) {
		if text == "" {
			return
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(text)
	}
	for _, cand := range resp.Candidates {
		appendText(candText(cand))
	}
	if resp.Response != nil {
		for _, r := range resp.Response.Results {
			for _, cand := range r.Response.Candidates {
				appendText(candText(cand))
			}
		}
	}
	return b.String()
}

type geminiCandidate struct {
	Content geminiContent `json:"content"`
}

func candText(c geminiCandidate) string {
	var text string
	for _, p := range c.Content.Parts {
		text += p.Text
	}
	return text
}

// ExtractSessionID has no Gemini-native marker; session grouping relies on
// the request headers handled by the middleware. Returns "".
func (GeminiParser) ExtractSessionID([]byte) string { return "" }
