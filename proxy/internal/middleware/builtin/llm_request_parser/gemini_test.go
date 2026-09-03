package llm_request_parser

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/netbirdio/netbird/proxy/internal/middleware"
)

// TestParseGeminiPath covers the model extraction from a Gemini path. The
// model is a URL segment, so a path the parser declines leaves the router with
// no model to authorise — which is what makes the negative cases matter as
// much as the positive ones.
func TestParseGeminiPath(t *testing.T) {
	cases := map[string]struct {
		ok     bool
		model  string
		stream bool
	}{
		"/v1beta/models/gemini-3.8-flash:generateContent":        {ok: true, model: "gemini-3.8-flash"},
		"/v1beta/models/gemini-3.8-flash:streamGenerateContent":  {ok: true, model: "gemini-3.8-flash", stream: true},
		"/v1/models/gemini-2.5-pro:generateContent":              {ok: true, model: "gemini-2.5-pro"},
		"/v1beta/models/gemini-2.5-pro:countTokens":              {ok: true, model: "gemini-2.5-pro"},
		"/v1beta/models/gemini-embedding-001:embedContent":       {ok: true, model: "gemini-embedding-001"},
		"/v1beta/models/gemini-3.8-flash/operations/op-1:cancel": {ok: false},
		"/v1/models":                       {ok: false},
		"/v1/models/gpt-5.5":               {ok: false},
		"/v1beta/models/gemini-3.8-flash:": {ok: false},
		"/v1beta/tunedModels/my-tuned-model:generateContent": {ok: false},
	}
	for path, want := range cases {
		gm, ok := parseGeminiPath(path)
		require.Equal(t, want.ok, ok, "parse outcome for %q", path)
		if !want.ok {
			continue
		}
		assert.Equal(t, want.model, gm.model, "model for %q", path)
		assert.Equal(t, want.stream, gm.stream, "stream flag for %q", path)
	}
}

// TestInvoke_GeminiPathEmitsSurfaceAndModel: the body of a generateContent
// request names no model, so without the path extraction the cost meter has
// nothing to price the request against.
func TestInvoke_GeminiPathEmitsSurfaceAndModel(t *testing.T) {
	mw := newMiddleware(t)

	out, err := mw.Invoke(context.Background(), &middleware.Input{
		URL: "/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
		Body: []byte(`{
			"systemInstruction":{"parts":[{"text":"Be brief."}]},
			"contents":[{"role":"user","parts":[{"text":"Explain AI"}]}]
		}`),
	})
	require.NoError(t, err)
	require.Equal(t, middleware.DecisionAllow, out.Decision)

	provider, ok := metaValue(t, out.Metadata, middleware.KeyLLMProvider)
	require.True(t, ok, "the gemini surface must be stamped")
	assert.Equal(t, "gemini", provider)

	model, ok := metaValue(t, out.Metadata, middleware.KeyLLMModel)
	require.True(t, ok, "the model must be read off the path")
	assert.Equal(t, "gemini-3.8-flash", model)

	stream, _ := metaValue(t, out.Metadata, middleware.KeyLLMStream)
	assert.Equal(t, "true", stream, "streamGenerateContent is a streaming method")

	prompt, ok := metaValue(t, out.Metadata, middleware.KeyLLMRequestPromptRaw)
	require.True(t, ok, "the Gemini parser must read the contents[] body")
	assert.Equal(t, "system: Be brief.\nuser: Explain AI", prompt)
}

// TestInvoke_GeminiInteractionsReadsTheBodyModel: the interactions endpoint is
// the one Gemini surface that carries the model in the body, so it must route
// through the ordinary parser path rather than the path extraction.
func TestInvoke_GeminiInteractionsReadsTheBodyModel(t *testing.T) {
	mw := newMiddleware(t)

	out, err := mw.Invoke(context.Background(), &middleware.Input{
		URL:  "/v1beta/interactions",
		Body: []byte(`{"model":"gemini-3.8-flash","input":"Explain how AI works in a few words"}`),
	})
	require.NoError(t, err)

	provider, _ := metaValue(t, out.Metadata, middleware.KeyLLMProvider)
	assert.Equal(t, "gemini", provider)

	model, _ := metaValue(t, out.Metadata, middleware.KeyLLMModel)
	assert.Equal(t, "gemini-3.8-flash", model)

	prompt, _ := metaValue(t, out.Metadata, middleware.KeyLLMRequestPromptRaw)
	assert.Equal(t, "Explain how AI works in a few words", prompt)
}
