package llm_response_parser

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/netbirdio/netbird/proxy/internal/llm"
)

// geminiStreamUsage is the usage block of a streamed Gemini chunk. Pointer
// fields tell "absent" from zero so a chunk that omits usage — every chunk but
// the last, on the generateContent surface — leaves the running totals alone.
// The generateContent (camelCase) and interactions (snake_case) names are both
// accepted so one decode covers either surface.
type geminiStreamUsage struct {
	PromptTokenCount        *int64 `json:"promptTokenCount"`
	CandidatesTokenCount    *int64 `json:"candidatesTokenCount"`
	ThoughtsTokenCount      *int64 `json:"thoughtsTokenCount"`
	ToolUsePromptTokenCount *int64 `json:"toolUsePromptTokenCount"`
	CachedContentTokenCount *int64 `json:"cachedContentTokenCount"`
	TotalTokenCount         *int64 `json:"totalTokenCount"`

	TotalInputTokens   *int64 `json:"total_input_tokens"`
	TotalOutputTokens  *int64 `json:"total_output_tokens"`
	TotalCachedTokens  *int64 `json:"total_cached_tokens"`
	TotalThoughtTokens *int64 `json:"total_thought_tokens"`
	TotalToolUseTokens *int64 `json:"total_tool_use_tokens"`
}

// geminiStreamChunk matches both Gemini streaming envelopes. A
// streamGenerateContent chunk is a partial GenerateContentResponse: text rides
// candidates[].content.parts[].text and the final chunk carries usageMetadata.
// An interactions event wraps the same accounting under interaction.usage and
// delivers text as step deltas.
type geminiStreamChunk struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata *geminiStreamUsage `json:"usageMetadata"`

	Delta *struct {
		Text string `json:"text"`
	} `json:"delta"`
	Interaction *struct {
		Usage *geminiStreamUsage `json:"usage"`
	} `json:"interaction"`
}

// accumulateGeminiStream concatenates the per-chunk text deltas and lifts the
// usage block off whichever chunk carries it. Gemini repeats the cumulative
// counts on every chunk that has them, so the last one observed wins.
func accumulateGeminiStream(body []byte) (llm.Usage, string) {
	var (
		usage      llm.Usage
		completion strings.Builder
	)
	scanner := llm.NewScanner(bytes.NewReader(body))
	for {
		ev, err := scanner.Next()
		if err != nil {
			break
		}
		if ev.Data == "" {
			continue
		}

		var chunk geminiStreamChunk
		if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
			continue
		}
		for _, c := range chunk.Candidates {
			for _, part := range c.Content.Parts {
				completion.WriteString(part.Text)
			}
		}
		if chunk.Delta != nil {
			completion.WriteString(chunk.Delta.Text)
		}

		u := chunk.UsageMetadata
		if u == nil && chunk.Interaction != nil {
			u = chunk.Interaction.Usage
		}
		applyGeminiStreamUsage(u, &usage)
	}
	return usage, completion.String()
}

// applyGeminiStreamUsage folds a non-nil usage block into the running totals.
// Thought and tool-use tokens join the buckets Google bills them under, and the
// total is backfilled when the chunk omits it.
func applyGeminiStreamUsage(u *geminiStreamUsage, usage *llm.Usage) {
	if u == nil {
		return
	}
	input := derefInt64(u.PromptTokenCount) + derefInt64(u.ToolUsePromptTokenCount) +
		derefInt64(u.TotalInputTokens) + derefInt64(u.TotalToolUseTokens)
	output := derefInt64(u.CandidatesTokenCount) + derefInt64(u.ThoughtsTokenCount) +
		derefInt64(u.TotalOutputTokens) + derefInt64(u.TotalThoughtTokens)
	cached := derefInt64(u.CachedContentTokenCount) + derefInt64(u.TotalCachedTokens)
	if input == 0 && output == 0 && cached == 0 {
		return
	}
	usage.InputTokens = input
	usage.OutputTokens = output
	usage.CachedInputTokens = cached
	usage.TotalTokens = derefInt64(u.TotalTokenCount)
	if usage.TotalTokens == 0 {
		usage.TotalTokens = input + output
	}
}
