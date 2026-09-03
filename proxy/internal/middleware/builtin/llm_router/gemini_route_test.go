package llm_router

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/netbirdio/netbird/proxy/internal/middleware"
)

func geminiRoute(models ...string) ProviderRoute {
	return ProviderRoute{
		ID: "gemini-prod", Gemini: true, Vendor: "gemini",
		Models:          models,
		AllowedGroupIDs: []string{defaultTestGroup},
		UpstreamScheme:  "https",
		UpstreamHost:    "generativelanguage.googleapis.com",
		AuthHeaderName:  "x-goog-api-key",
		AuthHeaderValue: "AIza-test",
	}
}

func TestRouter_GeminiPathRoutesToTheGeminiProvider(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{geminiRoute("gemini-3.8-flash")}})

	out, err := mw.Invoke(context.Background(), pathRoutedInput(
		"/v1beta/models/gemini-3.8-flash:generateContent",
		"gemini",
		"gemini-3.8-flash",
	))
	require.NoError(t, err)
	require.Equal(t, middleware.DecisionAllow, out.Decision)
	require.NotNil(t, out.Mutations)
	require.NotNil(t, out.Mutations.RewriteUpstream)

	assert.Equal(t, "generativelanguage.googleapis.com", out.Mutations.RewriteUpstream.Host)
	require.NotNil(t, out.Mutations.RewriteUpstream.AuthHeader, "the credential must be injected")
	assert.Equal(t, "x-goog-api-key", out.Mutations.RewriteUpstream.AuthHeader.Name,
		"Gemini takes its key in its own header, not Authorization")
}

// A model the operator never registered must not be reachable through a
// registered Gemini credential.
func TestRouter_GeminiModelAllowlistEnforced(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{geminiRoute("gemini-3.8-flash")}})

	out, err := mw.Invoke(context.Background(), pathRoutedInput(
		"/v1beta/models/gemini-3.1-pro-preview:generateContent",
		"gemini",
		"gemini-3.1-pro-preview",
	))
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionDeny, out.Decision)
	require.NotNil(t, out.DenyReason)
	assert.Equal(t, denyCodeNotRoutable, out.DenyReason.Code)
}

// An operator who pasted the resource name Google's listing reports must get
// the same routing as one who registered the bare id.
func TestRouter_GeminiResourceNameClaimsTheBareModel(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{geminiRoute("models/gemini-3.8-flash")}})

	out, err := mw.Invoke(context.Background(), pathRoutedInput(
		"/v1beta/models/gemini-3.8-flash:generateContent",
		"gemini",
		"gemini-3.8-flash",
	))
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionAllow, out.Decision)
}

// The Gemini listing carries no model, so it routes by path; without it a
// client's startup enumeration reaches the synth placeholder upstream.
func TestRouter_GeminiListingRoutesAndIsBounded(t *testing.T) {
	route := geminiRoute("gemini-3.8-flash", "gemini-2.5-pro")
	mw := New(Config{Providers: []ProviderRoute{route}})

	out, err := mw.Invoke(context.Background(), &middleware.Input{
		Slot:       middleware.SlotOnRequest,
		Method:     "GET",
		URL:        "/v1beta/models",
		UserGroups: []string{defaultTestGroup},
	})
	require.NoError(t, err)
	require.Equal(t, middleware.DecisionAllow, out.Decision)
	require.NotNil(t, out.Mutations)
	require.NotNil(t, out.Mutations.RewriteUpstream)

	assert.Equal(t, "generativelanguage.googleapis.com", out.Mutations.RewriteUpstream.Host)
	assert.Equal(t, []string{"gemini-3.8-flash", "gemini-2.5-pro"},
		out.Mutations.RewriteUpstream.DiscoveryModels,
		"the picker is bounded to what the record registers")

	nonInference, ok := lookupMetadata(out.Metadata, middleware.KeyLLMNonInference)
	require.True(t, ok, "a listing spends no tokens")
	assert.Equal(t, "true", nonInference)
}

// isGeminiPath must not claim the OpenAI-shaped listing or per-model lookup:
// they share the "/v1/models" prefix and are authorised through the model table.
func TestIsGeminiPath(t *testing.T) {
	claimed := []string{
		"/v1beta/models/gemini-3.8-flash:generateContent",
		"/v1beta/models/gemini-3.8-flash:streamGenerateContent",
		"/v1/models/gemini-2.5-pro:generateContent",
	}
	for _, path := range claimed {
		assert.True(t, isGeminiPath(path), "must be routed by path: %q", path)
	}

	notClaimed := []string{
		"/v1/models",
		"/v1/models/gpt-5.5",
		"/v1beta/models",
		"/v1/chat/completions",
		"/v1beta/models/gemini-3.8-flash/operations/op-1:cancel",
	}
	for _, path := range notClaimed {
		assert.False(t, isGeminiPath(path), "must not be routed by path: %q", path)
	}
}
