package llm_router

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/netbirdio/netbird/proxy/internal/middleware"
)

// pathRoutedInput builds an Input mimicking the post-llm_request_parser state
// for a path-routed (Vertex/Bedrock) request: a request URL plus the model and
// (optionally) provider/vendor metadata the parser emits.
func pathRoutedInput(url, provider, model string) *middleware.Input {
	md := []middleware.KV{{Key: middleware.KeyLLMModel, Value: model}}
	if provider != "" {
		md = append(md, middleware.KV{Key: middleware.KeyLLMProvider, Value: provider})
	}
	return &middleware.Input{
		Slot:       middleware.SlotOnRequest,
		URL:        url,
		Metadata:   md,
		UserGroups: []string{defaultTestGroup},
	}
}

func vertexRoute() ProviderRoute {
	return ProviderRoute{
		ID: "vertex-prod", Vertex: true,
		AllowedGroupIDs: []string{defaultTestGroup},
		UpstreamScheme:  "https",
		UpstreamHost:    "europe-west1-aiplatform.googleapis.com",
		AuthHeaderName:  "Authorization",
		AuthHeaderValue: "Bearer x",
	}
}

// The google publisher speaks the Gemini surface now that a Gemini parser
// exists, so a request the parser tagged "gemini" routes and meters like any
// other Vertex publisher.
func TestRouter_VertexGoogleGeminiAllowed(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{vertexRoute()}})
	in := pathRoutedInput(
		"/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-pro:generateContent",
		"gemini", // google -> request parser emits llm.provider=gemini
		"gemini-2.5-pro",
	)
	out, err := mw.Invoke(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionAllow, out.Decision, "google publisher must meter on the gemini surface")
}

// A Vertex publisher with no parser surface (no llm.provider) must still be
// denied, not forwarded unmetered. The invariant that once covered google
// still guards every genuinely unknown publisher.
func TestRouter_VertexUnmeterablePublisherDenied(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{vertexRoute()}})
	in := pathRoutedInput(
		"/v1/projects/p/locations/global/publishers/mistralai/models/mistral-large:predict",
		"", // mistralai -> request parser emits NO llm.provider
		"mistral-large",
	)
	out, err := mw.Invoke(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionDeny, out.Decision, "unmeterable Vertex publisher must deny")
	assert.Equal(t, 403, out.DenyStatus, "unmeterable deny is a 403")
	require.NotNil(t, out.DenyReason)
	assert.Equal(t, denyCodeUnmeterable, out.DenyReason.Code, "deny code must flag the unmeterable publisher")
}

// A Vertex publisher with a parser surface (anthropic) is allowed.
func TestRouter_VertexMeterablePublisherAllowed(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{vertexRoute()}})
	in := pathRoutedInput(
		"/v1/projects/p/locations/global/publishers/anthropic/models/claude-sonnet-4-5:rawPredict",
		"anthropic",
		"claude-sonnet-4-5",
	)
	out, err := mw.Invoke(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionAllow, out.Decision, "meterable Vertex publisher must allow")
}

// A path-routed provider with an explicit Models list must reject models not in
// the list (the provider credential can't be used for unauthorised models).
func TestRouter_PathRoutedModelAllowlistEnforced(t *testing.T) {
	route := ProviderRoute{
		ID: "bedrock-prod", Bedrock: true,
		Models:          []string{"anthropic.claude-sonnet-4-5"},
		AllowedGroupIDs: []string{defaultTestGroup},
		UpstreamScheme:  "https",
		UpstreamHost:    "bedrock-runtime.eu-central-1.amazonaws.com",
		AuthHeaderName:  "Authorization",
		AuthHeaderValue: "Bearer x",
	}
	mw := New(Config{Providers: []ProviderRoute{route}})

	allowed := pathRoutedInput(
		"/model/eu.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
		"bedrock", "anthropic.claude-sonnet-4-5",
	)
	out, err := mw.Invoke(context.Background(), allowed)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionAllow, out.Decision, "model in the allowlist must be served")

	denied := pathRoutedInput(
		"/model/amazon.nova-pro-v1:0/invoke",
		"bedrock", "amazon.nova-pro",
	)
	out, err = mw.Invoke(context.Background(), denied)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionDeny, out.Decision, "model outside the allowlist must deny")
	require.NotNil(t, out.DenyReason)
	assert.Equal(t, denyCodeNotRoutable, out.DenyReason.Code, "unlisted model denies as not-routable")
}

// A "/bedrock" gateway-namespace prefix routes the same as the native path and
// records the prefix on the rewrite so the proxy strips it before forwarding.
func TestRouter_BedrockNamespacePrefixStripped(t *testing.T) {
	route := ProviderRoute{
		ID: "bedrock-prod", Bedrock: true,
		AllowedGroupIDs: []string{defaultTestGroup},
		UpstreamScheme:  "https",
		UpstreamHost:    "bedrock-runtime.eu-central-1.amazonaws.com",
		AuthHeaderName:  "Authorization",
		AuthHeaderValue: "Bearer x",
	}
	mw := New(Config{Providers: []ProviderRoute{route}})

	prefixed := pathRoutedInput(
		"/bedrock/model/eu.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke-with-response-stream",
		"bedrock", "anthropic.claude-sonnet-4-5",
	)
	out, err := mw.Invoke(context.Background(), prefixed)
	require.NoError(t, err)
	require.Equal(t, middleware.DecisionAllow, out.Decision, "prefixed Bedrock path must route")
	require.NotNil(t, out.Mutations)
	require.NotNil(t, out.Mutations.RewriteUpstream)
	assert.Equal(t, "/bedrock", out.Mutations.RewriteUpstream.StripPathPrefix,
		"namespace prefix must be recorded so the proxy strips it before forwarding")

	native := pathRoutedInput(
		"/model/eu.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
		"bedrock", "anthropic.claude-sonnet-4-5",
	)
	out, err = mw.Invoke(context.Background(), native)
	require.NoError(t, err)
	require.Equal(t, middleware.DecisionAllow, out.Decision, "native Bedrock path must route")
	require.NotNil(t, out.Mutations.RewriteUpstream)
	assert.Empty(t, out.Mutations.RewriteUpstream.StripPathPrefix,
		"native path carries no namespace prefix to strip")
}

func geminiRoute(models ...string) ProviderRoute {
	return ProviderRoute{
		ID: "gemini-prod", Gemini: true,
		Models:          models,
		AllowedGroupIDs: []string{defaultTestGroup},
		UpstreamScheme:  "https",
		UpstreamHost:    "generativelanguage.googleapis.com",
		AuthHeaderName:  "x-goog-api-key",
		AuthHeaderValue: "secret",
	}
}

func TestIsGeminiPath(t *testing.T) {
	yes := []string{
		"/v1beta/models/gemini-2.5-pro:generateContent",
		"/v1beta/models/gemini-2.5-pro:streamGenerateContent",
		"/v1/models/gemini-2.5-flash:batchGenerateContent",
		"/models/gemini-2.5-flash:generateContent",
		"/v1beta/models/gemini-2.5-pro:streamBatchGenerateContent",
		"/v1beta/interactions",
	}
	for _, p := range yes {
		assert.True(t, isGeminiPath(p), "%q must be recognised as a Gemini path", p)
	}
	no := []string{
		"/v1/chat/completions",
		"/v1beta/models/gemini-2.5-pro:countTokens",
		"/v1beta/models/gemini-2.5-pro",
		"/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-pro:generateContent",
		"/v1beta/models/gemini-2.5-pro:GenerateSomethingElse",
		"/v1beta/models/gemini-2.5-pro:",
	}
	for _, p := range no {
		assert.False(t, isGeminiPath(p), "%q must not be a Gemini path", p)
	}
}

func TestRouter_GeminiPathRouted(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{geminiRoute("gemini-2.5-pro")}})

	in := pathRoutedInput("/v1beta/models/gemini-2.5-pro:generateContent", "gemini", "gemini-2.5-pro")
	out, err := mw.Invoke(context.Background(), in)
	require.NoError(t, err)
	require.Equal(t, middleware.DecisionAllow, out.Decision, "listed Gemini model must route")
	require.NotNil(t, out.Mutations.RewriteUpstream)
	assert.Equal(t, "generativelanguage.googleapis.com", out.Mutations.RewriteUpstream.Host)
	assert.Equal(t, "x-goog-api-key", out.Mutations.RewriteUpstream.AuthHeader.Name,
		"Gemini credential rides the x-goog-api-key header")

	denied := pathRoutedInput("/v1beta/models/gemini-3.1-pro-preview:generateContent", "gemini", "gemini-3.1-pro-preview")
	out, err = mw.Invoke(context.Background(), denied)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionDeny, out.Decision, "model outside the Gemini allowlist must deny")
}

// routeClaimsModel: an operator who registered the versioned form still
// serves a request whose path carried the bare id (the parser strips
// "@version").
func TestRouter_GeminiRouteClaimsNormalizedModel(t *testing.T) {
	route := geminiRoute("gemini-2.5-pro@latest")
	assert.True(t, routeClaimsModel(route, "gemini-2.5-pro"),
		"a registered versioned id must claim the normalized request model")
	assert.False(t, routeClaimsModel(geminiRoute("gemini-2.5-pro"), "gemini-3.1-pro-preview"),
		"an unlisted model stays unclaimed")
}

// Gemini has no model listing endpoint, so a Gemini route must never be
// picked for GET /v1/models — it would 404 at the upstream.
func TestRouter_GeminiExcludedFromModellessListing(t *testing.T) {
	mw := New(Config{Providers: []ProviderRoute{geminiRoute()}})
	in := &middleware.Input{
		Slot:       middleware.SlotOnRequest,
		URL:        "/v1/models",
		Method:     "GET",
		UserGroups: []string{defaultTestGroup},
	}
	out, err := mw.Invoke(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionDeny, out.Decision,
		"a Gemini-only config must not serve the OpenAI-style model listing")
}

// A path-routed provider with no configured Models is catch-all: any model the
// credential can reach is served (preserves the zero-config behaviour).
func TestRouter_PathRoutedCatchAllServesAnyModel(t *testing.T) {
	route := ProviderRoute{
		ID: "bedrock-catchall", Bedrock: true,
		AllowedGroupIDs: []string{defaultTestGroup},
		UpstreamScheme:  "https",
		UpstreamHost:    "bedrock-runtime.eu-central-1.amazonaws.com",
		AuthHeaderName:  "Authorization",
		AuthHeaderValue: "Bearer x",
	}
	mw := New(Config{Providers: []ProviderRoute{route}})
	in := pathRoutedInput(
		"/model/amazon.nova-pro-v1:0/invoke",
		"bedrock", "amazon.nova-pro",
	)
	out, err := mw.Invoke(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, middleware.DecisionAllow, out.Decision, "catch-all path-routed provider serves any model")
}
