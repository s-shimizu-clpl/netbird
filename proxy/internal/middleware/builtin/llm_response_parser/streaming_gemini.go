package llm_response_parser

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/netbirdio/netbird/proxy/internal/llm"
)

// geminiStreamChunk is one data frame of streamGenerateContent's SSE
// stream. Gemini emits no [DONE] sentinel and no terminal event type; the
// stream simply ends. Only frames carrying usageMetadata change the running
// usage — every frame's block is cumulative, so the last one wins.
type geminiStreamChunk struct {
	Candidates []struct {
		Content geminiStreamContent `json:"content"`
	} `json:"candidates"`
	UsageMetadata *geminiStreamUsageMetadata `json:"usageMetadata"`
}

type geminiStreamContent struct {
	Parts []struct {
		Text string `json:"text"`
	} `json:"parts"`
}

type geminiStreamUsageMetadata struct {
	PromptTokenCount        int64 `json:"promptTokenCount"`
	CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int64 `json:"thoughtsTokenCount"`
	TotalTokenCount         int64 `json:"totalTokenCount"`
	CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
}

// accumulateGeminiStream walks the streamGenerateContent SSE frames,
// concatenating candidates[].content.parts[].text and adopting the last
// usageMetadata frame seen. A truncated stream returns the partial
// completion and the most recent usage the provider reported before the
// cut.
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
		for _, cand := range chunk.Candidates {
			for _, p := range cand.Content.Parts {
				completion.WriteString(p.Text)
			}
		}
		if chunk.UsageMetadata != nil {
			usage = geminiStreamUsage(chunk.UsageMetadata)
		}
	}
	return usage, completion.String()
}

func geminiStreamUsage(u *geminiStreamUsageMetadata) llm.Usage {
	total := u.TotalTokenCount
	if total == 0 {
		total = u.PromptTokenCount + u.CandidatesTokenCount + u.ThoughtsTokenCount
	}
	return llm.Usage{
		InputTokens:       u.PromptTokenCount,
		OutputTokens:      u.CandidatesTokenCount + u.ThoughtsTokenCount,
		TotalTokens:       total,
		CachedInputTokens: u.CachedContentTokenCount,
	}
}
