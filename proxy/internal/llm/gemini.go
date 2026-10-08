package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// GeminiParser implements the Parser interface for Google Gemini APIs,
// covering both Google AI Studio (generativelanguage.googleapis.com) and
// Google Cloud Vertex AI (publishers/google).
type GeminiParser struct{}

var geminiPathHints = []string{
	":generatecontent",
	":streamgeneratecontent",
	":counttokens",
}

// Provider returns ProviderGemini.
func (GeminiParser) Provider() Provider { return ProviderGemini }

// ProviderName returns the stable label used for metrics and metadata.
func (GeminiParser) ProviderName() string { return ProviderNameGemini }

// DetectFromURL reports whether the request path targets a Gemini model endpoint.
// Gemini endpoints use the colon action syntax (e.g. :generateContent).
func (GeminiParser) DetectFromURL(path string) bool {
	lower := strings.ToLower(path)
	for _, hint := range geminiPathHints {
		if strings.Contains(lower, hint) {
			return true
		}
	}
	return false
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiRequest struct {
	Model             string          `json:"model"`
	Contents          []geminiContent `json:"contents"`
	SystemInstruction *geminiContent  `json:"systemInstruction"`
}

// ParseRequest extracts facts from a Gemini request body. Model is typically
// carried in the URL path, but may be present in body in some clients.
func (GeminiParser) ParseRequest(body []byte) (RequestFacts, error) {
	if len(body) == 0 {
		return RequestFacts{}, nil
	}
	var req geminiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return RequestFacts{}, fmt.Errorf("decode gemini request: %w: %v", ErrMalformedRequest, err)
	}
	return RequestFacts{
		Model: req.Model,
	}, nil
}

type geminiUsageMetadata struct {
	PromptTokenCount        *int64 `json:"promptTokenCount"`
	CandidatesTokenCount    *int64 `json:"candidatesTokenCount"`
	TotalTokenCount         *int64 `json:"totalTokenCount"`
	CachedContentTokenCount *int64 `json:"cachedContentTokenCount"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
	Index        int           `json:"index"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate    `json:"candidates"`
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata"`
	ModelVersion  string               `json:"modelVersion"`
}

// ParseResponse decodes a Gemini response body (single object or JSON array for
// chunked responses) and extracts token accounting.
func (GeminiParser) ParseResponse(status int, contentType string, body []byte) (Usage, error) {
	if status != 200 {
		return Usage{}, fmt.Errorf("gemini status %d: %w", status, ErrNotLLMResponse)
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return Usage{}, fmt.Errorf("empty gemini response: %w", ErrMalformedResponse)
	}

	if trimmed[0] == '[' {
		var list []geminiResponse
		if err := json.Unmarshal(trimmed, &list); err != nil {
			return Usage{}, fmt.Errorf("decode gemini array response: %w: %v", ErrMalformedResponse, err)
		}
		for i := len(list) - 1; i >= 0; i-- {
			if list[i].UsageMetadata != nil {
				return buildGeminiUsage(list[i].UsageMetadata), nil
			}
		}
		return Usage{}, nil
	}

	var resp geminiResponse
	if err := json.Unmarshal(trimmed, &resp); err != nil {
		return Usage{}, fmt.Errorf("decode gemini response: %w: %v", ErrMalformedResponse, err)
	}
	if resp.UsageMetadata == nil {
		return Usage{}, nil
	}
	return buildGeminiUsage(resp.UsageMetadata), nil
}

func buildGeminiUsage(u *geminiUsageMetadata) Usage {
	usage := Usage{
		InputTokens:       derefInt64(u.PromptTokenCount),
		OutputTokens:      derefInt64(u.CandidatesTokenCount),
		TotalTokens:       derefInt64(u.TotalTokenCount),
		CachedInputTokens: derefInt64(u.CachedContentTokenCount),
	}
	if usage.TotalTokens == 0 && (usage.InputTokens > 0 || usage.OutputTokens > 0) {
		usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	}
	return usage
}

// ExtractPrompt collects user and system text from the Gemini request body.
func (GeminiParser) ExtractPrompt(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var req geminiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}

	var parts []string
	if req.SystemInstruction != nil {
		for _, p := range req.SystemInstruction.Parts {
			if text := strings.TrimSpace(p.Text); text != "" {
				parts = append(parts, text)
			}
		}
	}
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if text := strings.TrimSpace(p.Text); text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

// ExtractCompletion collects assistant response text from a non-streaming
// Gemini response body.
func (GeminiParser) ExtractCompletion(status int, contentType string, body []byte) string {
	if status != 200 || len(body) == 0 {
		return ""
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return ""
	}

	if trimmed[0] == '[' {
		var list []geminiResponse
		if err := json.Unmarshal(trimmed, &list); err != nil {
			return ""
		}
		var parts []string
		for _, item := range list {
			for _, cand := range item.Candidates {
				for _, p := range cand.Content.Parts {
					if p.Text != "" {
						parts = append(parts, p.Text)
					}
				}
			}
		}
		return strings.Join(parts, "")
	}

	var resp geminiResponse
	if err := json.Unmarshal(trimmed, &resp); err != nil {
		return ""
	}
	var parts []string
	for _, cand := range resp.Candidates {
		for _, p := range cand.Content.Parts {
			if p.Text != "" {
				parts = append(parts, p.Text)
			}
		}
	}
	return strings.Join(parts, "")
}

// ExtractSessionID returns a session identifier from the Gemini body if present.
func (GeminiParser) ExtractSessionID([]byte) string {
	return ""
}
