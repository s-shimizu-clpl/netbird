package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ProviderNameGemini is the stable label for the Google Gemini parser, used as
// the llm.provider metadata value and the cost-meter formula selector.
const ProviderNameGemini = "gemini"

// GeminiParser implements the Parser interface for the Google Gemini API
// (generativelanguage.googleapis.com). Two request shapes reach it:
//
//   - generateContent / streamGenerateContent, where the model is a path
//     segment ("/v1beta/models/{model}:generateContent") and the body carries
//     contents[] + systemInstruction. The request middleware reads the model
//     off the path, the way it does for Bedrock and Vertex.
//   - interactions, where the model is an ordinary body field and the reply is
//     a list of steps rather than candidates.
//
// Both are handled here because they share a credential, an upstream and a
// pricing surface; splitting them would make a provider record ambiguous.
type GeminiParser struct{}

// geminiPathHints are the endpoint markers that identify a Gemini request. The
// generate/embed families are method suffixes after a colon, so matching them
// cannot collide with the OpenAI-shaped "/v1/models" listing.
var geminiPathHints = []string{
	":generatecontent",
	":streamgeneratecontent",
	":counttokens",
	":embedcontent",
	":batchembedcontents",
	"/v1beta/interactions",
}

// Provider returns ProviderGemini.
func (GeminiParser) Provider() Provider { return ProviderGemini }

// ProviderName returns the stable label used for metrics and metadata.
func (GeminiParser) ProviderName() string { return ProviderNameGemini }

// DetectFromURL reports whether the given request path looks like a Gemini API
// endpoint. The match is case-insensitive and substring-based.
func (GeminiParser) DetectFromURL(path string) bool {
	lower := strings.ToLower(path)
	for _, hint := range geminiPathHints {
		if strings.Contains(lower, hint) {
			return true
		}
	}
	return false
}

// geminiRequest is the union of the two request bodies. generateContent leaves
// Model empty (it is a path segment) and fills Contents; interactions fills
// Model, Stream and Input.
type geminiRequest struct {
	Model             string          `json:"model"`
	Stream            *bool           `json:"stream"`
	SystemInstruction *geminiContent  `json:"systemInstruction"`
	SystemSnake       *geminiContent  `json:"system_instruction"`
	Contents          []geminiContent `json:"contents"`
	// Input is the interactions-API prompt: a bare string, a Content object,
	// or an array of steps.
	Input json.RawMessage `json:"input"`
}

// geminiContent is one turn of the conversation. Role is absent on
// systemInstruction and on a single-turn prompt.
type geminiContent struct {
	Role  string `json:"role"`
	Parts []struct {
		Text string `json:"text"`
	} `json:"parts"`
}

// text joins the content's text parts, dropping the non-text ones (inline data,
// function calls) that carry no prompt an operator would read.
func (c geminiContent) text() string {
	var b strings.Builder
	for _, p := range c.Parts {
		if p.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(p.Text)
	}
	return b.String()
}

// ParseRequest extracts the model name and streaming flag from a Gemini request
// body. A generateContent body carries neither — the model is a path segment
// and streaming is the method name — so the request middleware supplies both
// and this returns empty facts for it.
func (GeminiParser) ParseRequest(body []byte) (RequestFacts, error) {
	var req geminiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return RequestFacts{}, fmt.Errorf("decode gemini request: %w: %v", ErrMalformedRequest, err)
	}
	return RequestFacts{
		Model:  NormalizeGeminiModel(req.Model),
		Stream: ptrDeref(req.Stream),
	}, nil
}

// geminiResponse is the union of the two response envelopes' usage blocks.
//
// UsageMetadata is the generateContent shape: cachedContentTokenCount is a
// SUBSET of promptTokenCount (Google bills it at the cheaper cached rate),
// while thoughtsTokenCount is additive and billed as output. Usage is the
// interactions shape, which names the same quantities differently.
type geminiResponse struct {
	UsageMetadata *struct {
		PromptTokenCount        int64 `json:"promptTokenCount"`
		CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
		ThoughtsTokenCount      int64 `json:"thoughtsTokenCount"`
		ToolUsePromptTokenCount int64 `json:"toolUsePromptTokenCount"`
		CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
		TotalTokenCount         int64 `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	Usage *struct {
		TotalInputTokens   int64 `json:"total_input_tokens"`
		TotalOutputTokens  int64 `json:"total_output_tokens"`
		TotalCachedTokens  int64 `json:"total_cached_tokens"`
		TotalThoughtTokens int64 `json:"total_thought_tokens"`
		TotalToolUseTokens int64 `json:"total_tool_use_tokens"`
	} `json:"usage"`
}

// usage folds whichever envelope was present into the provider-agnostic shape.
// Thought and tool-use tokens join the output bucket: Google bills them at the
// output rate, and the cost meter has no separate bucket for them.
func (r geminiResponse) usage() Usage {
	switch {
	case r.UsageMetadata != nil:
		u := r.UsageMetadata
		out := Usage{
			InputTokens:       u.PromptTokenCount + u.ToolUsePromptTokenCount,
			OutputTokens:      u.CandidatesTokenCount + u.ThoughtsTokenCount,
			TotalTokens:       u.TotalTokenCount,
			CachedInputTokens: u.CachedContentTokenCount,
		}
		if out.TotalTokens == 0 {
			out.TotalTokens = out.InputTokens + out.OutputTokens
		}
		return out
	case r.Usage != nil:
		u := r.Usage
		out := Usage{
			InputTokens:       u.TotalInputTokens + u.TotalToolUseTokens,
			OutputTokens:      u.TotalOutputTokens + u.TotalThoughtTokens,
			CachedInputTokens: u.TotalCachedTokens,
		}
		out.TotalTokens = out.InputTokens + out.OutputTokens
		return out
	default:
		return Usage{}
	}
}

// ParseResponse decodes the non-streaming Gemini response envelope. Status
// codes other than 200 are treated as non-LLM responses so the caller can skip
// cost accounting without aborting the request.
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
	return resp.usage(), nil
}

// ExtractPrompt returns the user-visible prompt text from a Gemini request
// body, handling the generateContent (systemInstruction + contents[]) and the
// interactions (input) shapes. Returns "" on any decode failure.
func (GeminiParser) ExtractPrompt(body []byte) string {
	var req geminiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	var b strings.Builder
	system := req.SystemInstruction
	if system == nil {
		system = req.SystemSnake
	}
	if system != nil {
		if s := system.text(); s != "" {
			b.WriteString("system: ")
			b.WriteString(s)
		}
	}
	for _, c := range req.Contents {
		text := c.text()
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
	if b.Len() == 0 {
		return geminiInputText(req.Input)
	}
	return b.String()
}

// geminiInputText reads the interactions-API "input", which is a bare string, a
// Content object, or an array of either. Returns "" for a shape it cannot read.
func geminiInputText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var one geminiContent
	if err := json.Unmarshal(raw, &one); err == nil {
		return one.text()
	}
	var many []geminiContent
	if err := json.Unmarshal(raw, &many); err != nil {
		return ""
	}
	var b strings.Builder
	for _, c := range many {
		text := c.text()
		if text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(text)
	}
	return b.String()
}

// geminiCompletion is the union of the two response bodies' answer shapes:
// generateContent returns candidates[].content.parts[].text, interactions
// returns steps[].content[].text on the model_output steps.
type geminiCompletion struct {
	Candidates []struct {
		Content geminiContent `json:"content"`
	} `json:"candidates"`
	Steps []struct {
		Type    string `json:"type"`
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	} `json:"steps"`
}

// geminiModelOutputStep is the interactions step type carrying the answer; the
// others (function calls, thoughts) are not completion text.
const geminiModelOutputStep = "model_output"

// ExtractCompletion returns the assistant text from a non-streaming Gemini
// response. Returns "" when status/content-type indicate the body is not
// parseable or no text part is present.
func (GeminiParser) ExtractCompletion(status int, contentType string, body []byte) string {
	if status != 200 || isEventStream(contentType) || !isJSON(contentType) {
		return ""
	}
	var resp geminiCompletion
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
	for _, c := range resp.Candidates {
		appendText(c.Content.text())
	}
	for _, step := range resp.Steps {
		if step.Type != geminiModelOutputStep {
			continue
		}
		for _, part := range step.Content {
			appendText(part.Text)
		}
	}
	return b.String()
}

// ExtractSessionID has no Gemini-native marker: the interactions API groups a
// conversation server-side by previous_interaction_id, which identifies the
// preceding turn rather than the session. Session grouping relies on the
// request headers handled by the middleware. Returns "".
func (GeminiParser) ExtractSessionID([]byte) string { return "" }
