package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGeminiParser_DetectFromURL(t *testing.T) {
	cases := []struct {
		name string
		path string
		want bool
	}{
		{"generateContent", "/v1beta/models/gemini-3.8-flash:generateContent", true},
		{"streaming", "/v1beta/models/gemini-3.8-flash:streamGenerateContent", true},
		{"v1 alias", "/v1/models/gemini-2.5-pro:generateContent", true},
		{"count tokens", "/v1beta/models/gemini-2.5-pro:countTokens", true},
		{"embeddings", "/v1beta/models/gemini-embedding-001:embedContent", true},
		{"interactions", "/v1beta/interactions", true},
		// The OpenAI-shaped listing and per-model lookup share the "/v1/models"
		// prefix and must stay with the OpenAI surface.
		{"model listing", "/v1/models", false},
		{"model lookup", "/v1/models/gpt-5.5", false},
		{"chat completions", "/v1/chat/completions", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, GeminiParser{}.DetectFromURL(tc.path))
		})
	}
}

// TestGeminiParser_ParseRequestReadsTheInteractionsBody: generateContent puts
// the model in the path, so only the interactions body carries one.
func TestGeminiParser_ParseRequestReadsTheInteractionsBody(t *testing.T) {
	facts, err := GeminiParser{}.ParseRequest([]byte(`{
		"model":"models/gemini-3.8-flash",
		"input":"Explain how AI works in a few words",
		"stream":true
	}`))
	require.NoError(t, err)
	assert.Equal(t, "gemini-3.8-flash", facts.Model, "the resource prefix is stripped")
	assert.True(t, facts.Stream)

	generate, err := GeminiParser{}.ParseRequest([]byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`))
	require.NoError(t, err)
	assert.Empty(t, generate.Model, "generateContent carries no model in the body")
	assert.False(t, generate.Stream)
}

func TestGeminiParser_ParseResponseUsageMetadata(t *testing.T) {
	body := []byte(`{
		"candidates":[{"content":{"role":"model","parts":[{"text":"AI learns patterns."}]}}],
		"usageMetadata":{
			"promptTokenCount":1200,
			"cachedContentTokenCount":900,
			"candidatesTokenCount":40,
			"thoughtsTokenCount":60,
			"totalTokenCount":1300
		}
	}`)

	usage, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)

	assert.Equal(t, int64(1200), usage.InputTokens, "cached tokens are counted inside the prompt total")
	assert.Equal(t, int64(100), usage.OutputTokens, "thoughts are billed as output")
	assert.Equal(t, int64(900), usage.CachedInputTokens)
	assert.Equal(t, int64(1300), usage.TotalTokens, "Google's own total wins over a recomputed one")
	assert.Zero(t, usage.CacheCreationTokens, "Gemini has no additive cache-write bucket")
}

func TestGeminiParser_ParseResponseInteractionsUsage(t *testing.T) {
	body := []byte(`{
		"id":"int_1","status":"completed",
		"steps":[{"type":"model_output","content":[{"type":"text","text":"AI learns patterns."}]}],
		"usage":{
			"total_input_tokens":800,
			"total_output_tokens":30,
			"total_cached_tokens":500,
			"total_thought_tokens":20,
			"total_tool_use_tokens":10
		}
	}`)

	usage, err := GeminiParser{}.ParseResponse(200, "application/json", body)
	require.NoError(t, err)

	assert.Equal(t, int64(810), usage.InputTokens, "tool-use tokens join the input bucket")
	assert.Equal(t, int64(50), usage.OutputTokens)
	assert.Equal(t, int64(500), usage.CachedInputTokens)
	assert.Equal(t, int64(860), usage.TotalTokens, "the interactions envelope reports no total, so it is derived")
}

func TestGeminiParser_ParseResponseSkipsNonLLMBodies(t *testing.T) {
	_, err := GeminiParser{}.ParseResponse(429, "application/json", []byte(`{"error":{"code":429}}`))
	assert.ErrorIs(t, err, ErrNotLLMResponse)

	_, err = GeminiParser{}.ParseResponse(200, "text/event-stream", []byte("data: {}\n\n"))
	assert.ErrorIs(t, err, ErrStreamingUnsupported)
}

func TestGeminiParser_ExtractPrompt(t *testing.T) {
	generate := GeminiParser{}.ExtractPrompt([]byte(`{
		"systemInstruction":{"parts":[{"text":"Be brief."}]},
		"contents":[
			{"role":"user","parts":[{"text":"Explain AI"}]},
			{"role":"model","parts":[{"text":"It learns."}]},
			{"role":"user","parts":[{"text":"Shorter"}]}
		]
	}`))
	assert.Equal(t, "system: Be brief.\nuser: Explain AI\nmodel: It learns.\nuser: Shorter", generate)

	interactions := GeminiParser{}.ExtractPrompt([]byte(`{"model":"gemini-3.8-flash","input":"Explain AI"}`))
	assert.Equal(t, "Explain AI", interactions, "the interactions input is a bare string")

	assert.Empty(t, GeminiParser{}.ExtractPrompt([]byte(`not json`)))
}

func TestGeminiParser_ExtractCompletion(t *testing.T) {
	generate := GeminiParser{}.ExtractCompletion(200, "application/json", []byte(`{
		"candidates":[{"content":{"role":"model","parts":[{"text":"first"},{"text":"second"}]}}]
	}`))
	assert.Equal(t, "first\nsecond", generate)

	// Only model_output steps are the answer; a tool call is not completion text.
	interactions := GeminiParser{}.ExtractCompletion(200, "application/json", []byte(`{
		"steps":[
			{"type":"function_call","content":[{"type":"text","text":"lookup(x)"}]},
			{"type":"model_output","content":[{"type":"text","text":"AI learns patterns."}]}
		]
	}`))
	assert.Equal(t, "AI learns patterns.", interactions)

	assert.Empty(t, GeminiParser{}.ExtractCompletion(500, "application/json", []byte(`{}`)))
}
