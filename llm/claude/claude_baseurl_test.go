package claude

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	llmtesting "github.com/bpowers/go-agent/llm/testing"
)

func TestClaude_BaseURLConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		baseURL     string
		expectedURL string
	}{
		{
			name:        "Default Anthropic URL",
			baseURL:     AnthropicURL,
			expectedURL: AnthropicURL,
		},
		{
			name:        "Custom proxy URL",
			baseURL:     "https://proxy.example.com/v1",
			expectedURL: "https://proxy.example.com/v1",
		},
		{
			name:        "Empty URL should use default",
			baseURL:     "",
			expectedURL: AnthropicURL,
		},
		{
			name:        "Enterprise gateway URL",
			baseURL:     "https://enterprise.anthropic.com/v1",
			expectedURL: "https://enterprise.anthropic.com/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewClient(tt.baseURL, "test-key", WithModel("claude-opus-4"))
			require.NoError(t, err, "Failed to create client with base URL %s", tt.baseURL)
			require.NotNil(t, client)

			// Use the test helper to validate BaseURL
			llmtesting.TestBaseURLConfiguration(t, client, tt.expectedURL)
		})
	}
}

func TestClaude_BaseURLPersistence(t *testing.T) {
	t.Parallel()

	// Test that BaseURL is preserved across the client lifetime
	customURL := "https://custom.anthropic.com/v1"
	client, err := NewClient(customURL, "test-key", WithModel("claude-opus-4"))
	require.NoError(t, err)

	// Validate using the BaseURLValidator interface
	validator, ok := client.(llmtesting.BaseURLValidator)
	require.True(t, ok, "Client should implement BaseURLValidator")

	// Verify the BaseURL is stored correctly
	assert.Equal(t, customURL, validator.BaseURL(), "BaseURL should match what was passed to NewClient")

	// Create a chat session and verify client still has correct BaseURL
	chat := client.NewChat("Test system prompt")
	assert.NotNil(t, chat)

	// BaseURL should still be the same
	assert.Equal(t, customURL, validator.BaseURL(), "BaseURL should remain unchanged after creating chat")
}

func TestClaude_EmptyBaseURLUsesDefault(t *testing.T) {
	t.Parallel()

	// Test that empty BaseURL defaults to AnthropicURL
	client, err := NewClient("", "test-key", WithModel("claude-opus-4"))
	require.NoError(t, err)

	// Validate using the BaseURLValidator interface
	validator, ok := client.(llmtesting.BaseURLValidator)
	require.True(t, ok, "Client should implement BaseURLValidator")

	// Verify the BaseURL defaults to AnthropicURL
	assert.Equal(t, AnthropicURL, validator.BaseURL(), "Empty BaseURL should default to AnthropicURL")
}
