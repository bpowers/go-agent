package claude

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	llmtesting "github.com/bpowers/go-agent/llm/testing"
)

func TestClaude_HeaderConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		headers         map[string]string
		expectedHeaders map[string]string
	}{
		{
			name:            "No headers",
			headers:         nil,
			expectedHeaders: nil,
		},
		{
			name: "Single header",
			headers: map[string]string{
				"X-Custom-Header": "custom-value",
			},
			expectedHeaders: map[string]string{
				"X-Custom-Header": "custom-value",
			},
		},
		{
			name: "Multiple headers",
			headers: map[string]string{
				"X-Custom-Header":  "custom-value",
				"X-Request-ID":     "req-123",
				"X-Correlation-ID": "corr-456",
			},
			expectedHeaders: map[string]string{
				"X-Custom-Header":  "custom-value",
				"X-Request-ID":     "req-123",
				"X-Correlation-ID": "corr-456",
			},
		},
		{
			name: "Headers with special characters",
			headers: map[string]string{
				"Authorization":   "Bearer special-token",
				"X-API-Version":   "2024-01-01",
				"Accept-Language": "en-US,en;q=0.9",
			},
			expectedHeaders: map[string]string{
				"Authorization":   "Bearer special-token",
				"X-API-Version":   "2024-01-01",
				"Accept-Language": "en-US,en;q=0.9",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewClient(AnthropicURL, "test-key",
				WithModel("claude-3-haiku"),
				WithHeaders(tt.headers))
			require.NoError(t, err, "Failed to create client with headers")
			require.NotNil(t, client)

			// Use the test helper to validate headers
			if tt.expectedHeaders == nil {
				// When no headers are provided, validate that Headers() returns nil
				validator, ok := client.(llmtesting.HeadersValidator)
				require.True(t, ok, "Client should implement HeadersValidator")
				assert.Nil(t, validator.Headers(), "Headers should be nil when not provided")
			} else {
				llmtesting.TestHeaderConfiguration(t, client, tt.expectedHeaders)
			}
		})
	}
}

func TestClaude_HeadersPersistence(t *testing.T) {
	t.Parallel()

	// Test that headers are preserved across the client lifetime
	customHeaders := map[string]string{
		"X-Custom-Header": "persistent-value",
		"X-Tracking-ID":   "track-789",
	}

	client, err := NewClient(AnthropicURL, "test-key",
		WithModel("claude-3-haiku"),
		WithHeaders(customHeaders))
	require.NoError(t, err)

	// Validate using the HeadersValidator interface
	validator, ok := client.(llmtesting.HeadersValidator)
	require.True(t, ok, "Client should implement HeadersValidator")

	// Verify the headers are stored correctly
	assert.Equal(t, customHeaders, validator.Headers(), "Headers should match what was passed to NewClient")

	// Create a chat session and verify client still has correct headers
	chat := client.NewChat("Test system prompt")
	assert.NotNil(t, chat)

	// Headers should still be the same
	assert.Equal(t, customHeaders, validator.Headers(), "Headers should remain unchanged after creating chat")
}

func TestClaude_HeadersWithOtherOptions(t *testing.T) {
	t.Parallel()

	// Test that headers work correctly alongside other options
	customHeaders := map[string]string{
		"X-Environment": "testing",
	}

	client, err := NewClient("https://custom.anthropic.com/v1", "test-key",
		WithModel("claude-3-haiku"),
		WithHeaders(customHeaders))
	require.NoError(t, err)

	// Verify all configurations are applied
	validator, ok := client.(llmtesting.HeadersValidator)
	require.True(t, ok)
	assert.Equal(t, customHeaders, validator.Headers())

	// Also verify BaseURL is still working
	baseValidator, ok := client.(llmtesting.BaseURLValidator)
	require.True(t, ok)
	assert.Equal(t, "https://custom.anthropic.com/v1", baseValidator.BaseURL())
}
