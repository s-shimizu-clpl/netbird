package llm_response_parser

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// geminiSSE is a streamGenerateContent?alt=sse response: every chunk is a
// partial GenerateContentResponse, and the counts are cumulative rather than
// per-chunk deltas.
const geminiSSE = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"AI "}]}}],"usageMetadata":{"promptTokenCount":12,"totalTokenCount":13}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"learns "}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"patterns."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"cachedContentTokenCount":8,"candidatesTokenCount":5,"thoughtsTokenCount":3,"totalTokenCount":20}}

`

func TestAccumulateGeminiStream(t *testing.T) {
	usage, completion := accumulateStream("gemini", []byte(geminiSSE))

	assert.Equal(t, "AI learns patterns.", completion, "text deltas concatenate in order")
	assert.Equal(t, int64(12), usage.InputTokens)
	assert.Equal(t, int64(8), usage.OutputTokens, "candidates plus thoughts")
	assert.Equal(t, int64(8), usage.CachedInputTokens)
	assert.Equal(t, int64(20), usage.TotalTokens, "the last chunk's cumulative total wins")
}

// TestAccumulateGeminiStream_Truncated: a body cut mid-stream must still yield
// the text and counts seen so far rather than nothing at all.
func TestAccumulateGeminiStream_Truncated(t *testing.T) {
	truncated := `data: {"candidates":[{"content":{"parts":[{"text":"AI "}]}}],"usageMetadata":{"promptTokenCount":12,"totalTokenCount":13}}

data: {"candidates":[{"content":{"parts":[{"text":"lea`

	usage, completion := accumulateStream("gemini", []byte(truncated))

	assert.Equal(t, "AI ", completion, "the incomplete frame is dropped, the complete one is kept")
	assert.Equal(t, int64(12), usage.InputTokens)
}

// TestAccumulateGeminiStream_Interactions covers the interactions surface,
// whose events carry step deltas and a terminal usage block under its own
// field names.
func TestAccumulateGeminiStream_Interactions(t *testing.T) {
	body := `event: step.delta
data: {"delta":{"text":"AI learns"}}

event: interaction.completed
data: {"interaction":{"status":"completed","usage":{"total_input_tokens":12,"total_output_tokens":5,"total_thought_tokens":3}}}

`

	usage, completion := accumulateStream("gemini", []byte(body))

	assert.Equal(t, "AI learns", completion)
	assert.Equal(t, int64(12), usage.InputTokens)
	assert.Equal(t, int64(8), usage.OutputTokens, "thoughts are billed as output")
	assert.Equal(t, int64(20), usage.TotalTokens, "no total on the wire, so it is derived")
}
