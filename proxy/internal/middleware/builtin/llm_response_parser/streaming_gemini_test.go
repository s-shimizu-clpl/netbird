package llm_response_parser

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/netbirdio/netbird/proxy/internal/middleware"
)

// Gemini's streamGenerateContent emits data frames only — no [DONE]
// sentinel, no terminal event. The accumulator must still lift the final
// usage and concatenate the candidate text.
func TestAccumulateGeminiStream_NoDoneSentinel(t *testing.T) {
	body := loadFixture(t, "gemini_stream.txt")
	require.Contains(t, string(body), "The ", "fixture sanity: first frame carries its text")

	usage, completion := accumulateGeminiStream(body)
	assert.Equal(t, int64(6), usage.InputTokens, "promptTokenCount from the final usageMetadata")
	assert.Equal(t, int64(12), usage.OutputTokens, "candidates plus thoughts tokens bill as output")
	assert.Equal(t, int64(18), usage.TotalTokens, "totalTokenCount from the final frame")
	assert.Equal(t, int64(4), usage.CachedInputTokens, "cached subset from the final frame")
	assert.Equal(t, "The capital of France is Paris.", completion, "candidate parts concatenate in order")
}

// usageMetadata is cumulative across frames; a mid-stream frame must not
// overwrite the running usage with a stale total once a later frame arrived.
func TestAccumulateGeminiStream_LastWins(t *testing.T) {
	body := []byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"b\"}]}}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":7,\"totalTokenCount\":12}}\n\n")
	usage, completion := accumulateGeminiStream(body)
	assert.Equal(t, int64(5), usage.InputTokens, "last frame's cumulative usage wins")
	assert.Equal(t, int64(7), usage.OutputTokens)
	assert.Equal(t, int64(12), usage.TotalTokens)
	assert.Equal(t, "ab", completion)
}

// A stream cut before the final usage frame still meters the partial
// counts observed so far.
func TestAccumulateGeminiStream_Truncated(t *testing.T) {
	body := []byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"par\"}]}}]}\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\"")
	usage, completion := accumulateGeminiStream(body)
	assert.Zero(t, usage.TotalTokens, "no usage frame means no tokens metered")
	assert.Equal(t, "par", completion, "text before the cut survives")
}

func TestInvoke_GeminiStreaming(t *testing.T) {
	m := newTestMiddleware(t)
	body := loadFixture(t, "gemini_stream.txt")

	in := &middleware.Input{
		Slot:        middleware.SlotOnResponse,
		Status:      200,
		RespHeaders: []middleware.KV{{Key: "Content-Type", Value: "text/event-stream"}},
		RespBody:    body,
		Metadata: []middleware.KV{
			{Key: middleware.KeyLLMProvider, Value: "gemini"},
			{Key: middleware.KeyLLMModel, Value: "gemini-2.5-pro"},
		},
	}

	out, err := m.Invoke(context.Background(), in)
	require.NoError(t, err, "Invoke must not error on streaming Gemini body")

	inTokens, ok := metaValue(out.Metadata, middleware.KeyLLMInputTokens)
	require.True(t, ok, "input tokens must be emitted")
	assert.Equal(t, "6", inTokens, "input tokens from the final usageMetadata frame")

	outTokens, _ := metaValue(out.Metadata, middleware.KeyLLMOutputTokens)
	assert.Equal(t, "12", outTokens, "candidates+thoughts tokens")

	completion, ok := metaValue(out.Metadata, middleware.KeyLLMResponseCompletion)
	require.True(t, ok, "completion must be emitted for streaming responses")
	assert.Equal(t, "The capital of France is Paris.", completion)
}
