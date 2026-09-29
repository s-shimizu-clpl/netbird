package llm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGeminiParser_DetectFromURL(t *testing.T) {
	require.True(t, GeminiParser{}.DetectFromURL("/v1beta/models/gemini-2.5-pro:generateContent"), "generateContent path")
	require.True(t, GeminiParser{}.DetectFromURL("/v1beta/models/gemini-2.5-pro:streamGenerateContent"), "streamGenerateContent path")
	require.True(t, GeminiParser{}.DetectFromURL("/v1beta/models/gemini-3.8-flash:batchGenerateContent"), "batchGenerateContent path")
	require.True(t, GeminiParser{}.DetectFromURL("/v1beta/interactions"), "interactions path")
	require.False(t, GeminiParser{}.DetectFromURL("/v1/chat/completions"), "openai path is not gemini")
	require.False(t, GeminiParser{}.DetectFromURL("/v1/messages"), "anthropic path is not gemini")
	require.False(t, GeminiParser{}.DetectFromURL("/model/x/invoke"), "bedrock path is not gemini")
}

func TestGeminiParser_ParseRequest_InteractionsModel(t *testing.T) {
	facts, err := GeminiParser{}.ParseRequest([]byte(`{"model":"gemini-3.8-flash","input":"hi"}`))
	require.NoError(t, err)
	require.Equal(t, "gemini-3.8-flash", facts.Model, "interactions body carries the model")
	require.False(t, facts.Stream, "generateContent family streams from the path, not the body")
}

func TestGeminiParser_ParseRequest_GenerateContentNoModel(t *testing.T) {
	// generateContent bodies carry no model field — it lives in the URL
	// path and the request middleware extracts it there.
	facts, err := GeminiParser{}.ParseRequest([]byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`))
	require.NoError(t, err)
	require.Empty(t, facts.Model, "generateContent body has no model to read")
}

func TestGeminiParser_ParseResponse_GenerateContent(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("fixtures", "gemini_generate_content.json"))
	require.NoError(t, err, "fixture must be readable")
	u, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(6), u.InputTokens, "promptTokenCount is the input bucket")
	require.Equal(t, int64(9), u.OutputTokens, "candidatesTokenCount is the output bucket")
	require.Equal(t, int64(4), u.CachedInputTokens, "cachedContentTokenCount is the cached subset")
	require.Equal(t, int64(15), u.TotalTokens, "totalTokenCount is the provider total")
}

// thoughtsTokenCount (reasoning) is billed as output, so it joins
// candidatesTokenCount rather than sitting outside the metered total.
func TestGeminiParser_ParseResponse_ThoughtsBilledAsOutput(t *testing.T) {
	body := []byte(`{"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"thoughtsTokenCount":30,"totalTokenCount":150}}`)
	u, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(100), u.InputTokens)
	require.Equal(t, int64(50), u.OutputTokens, "thinking tokens bill as output")
	require.Equal(t, int64(150), u.TotalTokens)
}

func TestGeminiParser_ParseResponse_TotalBackfill(t *testing.T) {
	body := []byte(`{"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3}}`)
	u, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(10), u.TotalTokens, "total backfills when the provider omits it")
}

func TestGeminiParser_ParseResponse_Batch(t *testing.T) {
	body := []byte(`{"response":{"results":[
		{"response":{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15,"cachedContentTokenCount":2}}},
		{"response":{"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":7,"totalTokenCount":27}}}
	],"usageMetadata":{"promptTokenCount":999,"candidatesTokenCount":999,"totalTokenCount":1998}}}`)
	u, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(30), u.InputTokens, "batch sums per-result prompt tokens")
	require.Equal(t, int64(12), u.OutputTokens, "batch sums per-result candidate tokens")
	require.Equal(t, int64(2), u.CachedInputTokens, "batch sums per-result cached tokens")
	require.Equal(t, int64(42), u.TotalTokens, "batch sums per-result totals, not the mirror aggregate")
}

func TestGeminiParser_ParseResponse_BatchMirrorFallback(t *testing.T) {
	// results carrying no usage at all: fall back to the response-level
	// aggregate so a batch reply still meters.
	body := []byte(`{"response":{"results":[{"response":{}}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6}}}`)
	u, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(4), u.InputTokens, "falls back to the response-level aggregate")
	require.Equal(t, int64(6), u.TotalTokens)
}

func TestGeminiParser_ParseResponse_Interactions(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("fixtures", "gemini_interactions.json"))
	require.NoError(t, err, "fixture must be readable")
	u, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(10), u.InputTokens, "interactions input_tokens")
	require.Equal(t, int64(20), u.OutputTokens, "interactions output_tokens")
	require.Equal(t, int64(4), u.CachedInputTokens, "interactions cached subset")
	require.Equal(t, int64(30), u.TotalTokens, "interactions total_tokens")
}

// An explicit cached_tokens of 0 must be honored over a missing details
// object — same pointer-priority rule as openAICachedTokens.
func TestGeminiParser_ParseResponse_InteractionsZeroCache(t *testing.T) {
	body := []byte(`{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":0}}}`)
	u, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Zero(t, u.CachedInputTokens, "explicit zero stays zero")
}

func TestGeminiParser_ParseResponse_NonSuccess(t *testing.T) {
	_, err := GeminiParser{}.ParseResponse(429, "application/json", []byte(`{"error":{"message":"quota"}}`))
	require.ErrorIs(t, err, ErrNotLLMResponse, "non-200 is not an LLM response")
}

func TestGeminiParser_ParseResponse_StreamingUnsupported(t *testing.T) {
	_, err := GeminiParser{}.ParseResponse(200, "text/event-stream", []byte("data: {}"))
	require.ErrorIs(t, err, ErrStreamingUnsupported, "event-stream must route to the streaming accumulator")
}

func TestGeminiParser_ExtractPrompt_GenerateContent(t *testing.T) {
	body := []byte(`{
		"systemInstruction":{"parts":[{"text":"be brief"}]},
		"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":[{"text":"hello"}]}]
	}`)
	require.Equal(t, "system: be brief\nuser: hi\nmodel: hello", GeminiParser{}.ExtractPrompt(body))
}

func TestGeminiParser_ExtractPrompt_Batch(t *testing.T) {
	body := []byte(`{"batch":{"display_name":"d","input_config":{"requests":{"requests":[
		{"request":{"contents":[{"role":"user","parts":[{"text":"q1"}]}]},"metadata":{"key":"request-1"}},
		{"request":{"contents":[{"role":"user","parts":[{"text":"q2"}]}]},"metadata":{"key":"request-2"}}
	]}}}}`)
	require.Equal(t, "user: q1\nuser: q2", GeminiParser{}.ExtractPrompt(body), "batch expands nested requests")
}

func TestGeminiParser_ExtractPrompt_Interactions(t *testing.T) {
	require.Equal(t, "hello", GeminiParser{}.ExtractPrompt([]byte(`{"model":"m","input":"hello"}`)), "string input")
	require.Equal(t, "a\nb", GeminiParser{}.ExtractPrompt([]byte(`{"input":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`)), "parts input")
}

func TestGeminiParser_ExtractCompletion_GenerateContent(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("fixtures", "gemini_generate_content.json"))
	require.NoError(t, err, "fixture must be readable")
	require.Equal(t, "The capital of France is Paris.", GeminiParser{}.ExtractCompletion(200, "application/json", body))
}

func TestGeminiParser_ExtractCompletion_Batch(t *testing.T) {
	body := []byte(`{"response":{"results":[
		{"response":{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}},
		{"response":{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}}
	]}}`)
	require.Equal(t, "a\nb", GeminiParser{}.ExtractCompletion(200, "application/json", body))
}

func TestGeminiParser_ExtractSessionID(t *testing.T) {
	require.Empty(t, GeminiParser{}.ExtractSessionID([]byte(`{"contents":[]}`)), "no gemini-native session marker")
}

func TestGeminiParser_RegisteredByName(t *testing.T) {
	p, ok := ParserByName(ProviderNameGemini)
	require.True(t, ok, "gemini parser is registered")
	require.Equal(t, ProviderGemini, p.Provider())
}
