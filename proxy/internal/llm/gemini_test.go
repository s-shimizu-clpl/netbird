package llm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGeminiParser_DetectFromURL(t *testing.T) {
	parser := GeminiParser{}

	cases := map[string]bool{
		"/v1beta/models/gemini-2.5-flash:generateContent":                                                     true,
		"/v1beta/models/gemini-2.5-flash:streamGenerateContent":                                               true,
		"/v1/models/gemini-1.5-pro:countTokens":                                                               true,
		"/v1/projects/my-prj/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent": true,
		"/v1/chat/completions": false,
		"/v1/messages":         false,
		"/v1/models":           false,
		"":                     false,
	}

	for path, want := range cases {
		require.Equal(t, want, parser.DetectFromURL(path), "detect %q", path)
	}
}

func TestGeminiParser_ParseRequest(t *testing.T) {
	parser := GeminiParser{}

	body := []byte(`{
		"model": "gemini-2.5-flash",
		"contents": [
			{"role": "user", "parts": [{"text": "Hello world"}]}
		]
	}`)
	facts, err := parser.ParseRequest(body)
	require.NoError(t, err)
	require.Equal(t, "gemini-2.5-flash", facts.Model)

	emptyFacts, err := parser.ParseRequest(nil)
	require.NoError(t, err)
	require.Empty(t, emptyFacts.Model)
}

func TestGeminiParser_ParseResponse_SingleObject(t *testing.T) {
	parser := GeminiParser{}

	body := []byte(`{
		"candidates": [
			{
				"content": {
					"parts": [{"text": "Hello!"}],
					"role": "model"
				},
				"finishReason": "STOP",
				"index": 0
			}
		],
		"usageMetadata": {
			"promptTokenCount": 10,
			"candidatesTokenCount": 5,
			"totalTokenCount": 15,
			"cachedContentTokenCount": 3
		},
		"modelVersion": "gemini-2.5-flash"
	}`)

	usage, err := parser.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(10), usage.InputTokens)
	require.Equal(t, int64(5), usage.OutputTokens)
	require.Equal(t, int64(15), usage.TotalTokens)
	require.Equal(t, int64(3), usage.CachedInputTokens)
}

func TestGeminiParser_ParseResponse_Array(t *testing.T) {
	parser := GeminiParser{}

	body := []byte(`[
		{
			"candidates": [{"content": {"parts": [{"text": "chunk1"}]}}]
		},
		{
			"candidates": [{"content": {"parts": [{"text": "chunk2"}]}}],
			"usageMetadata": {
				"promptTokenCount": 20,
				"candidatesTokenCount": 10,
				"totalTokenCount": 30,
				"cachedContentTokenCount": 8
			}
		}
	]`)

	usage, err := parser.ParseResponse(200, "application/json", body)
	require.NoError(t, err)
	require.Equal(t, int64(20), usage.InputTokens)
	require.Equal(t, int64(10), usage.OutputTokens)
	require.Equal(t, int64(30), usage.TotalTokens)
	require.Equal(t, int64(8), usage.CachedInputTokens)
}

func TestGeminiParser_ParseResponse_Non200(t *testing.T) {
	parser := GeminiParser{}
	_, err := parser.ParseResponse(403, "application/json", []byte(`{"error":{"code":403}}`))
	require.ErrorIs(t, err, ErrNotLLMResponse)
}

func TestGeminiParser_ExtractPrompt(t *testing.T) {
	parser := GeminiParser{}

	body := []byte(`{
		"systemInstruction": {
			"parts": [{"text": "You are a bot."}]
		},
		"contents": [
			{"role": "user", "parts": [{"text": "Hi"}]},
			{"role": "model", "parts": [{"text": "Hello"}]},
			{"role": "user", "parts": [{"text": "How are you?"}]}
		]
	}`)

	prompt := parser.ExtractPrompt(body)
	require.Equal(t, "You are a bot.\nHi\nHello\nHow are you?", prompt)
}

func TestGeminiParser_ExtractCompletion(t *testing.T) {
	parser := GeminiParser{}

	singleBody := []byte(`{
		"candidates": [
			{"content": {"parts": [{"text": "Answer 1"}]}}
		]
	}`)
	require.Equal(t, "Answer 1", parser.ExtractCompletion(200, "application/json", singleBody))

	arrayBody := []byte(`[
		{"candidates": [{"content": {"parts": [{"text": "Part 1 "}]}}]},
		{"candidates": [{"content": {"parts": [{"text": "Part 2"}]}}]}
	]`)
	require.Equal(t, "Part 1 Part 2", parser.ExtractCompletion(200, "application/json", arrayBody))

	require.Empty(t, parser.ExtractCompletion(500, "application/json", singleBody))
}
