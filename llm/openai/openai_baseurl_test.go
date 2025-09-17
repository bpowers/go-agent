package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	llmtesting "github.com/bpowers/go-agent/llm/testing"
)

func TestOpenAI_BaseURLConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		baseURL     string
		expectedURL string
	}{
		{
			name:        "Default OpenAI URL",
			baseURL:     OpenAIURL,
			expectedURL: OpenAIURL,
		},
		{
			name:        "Custom proxy URL",
			baseURL:     "https://proxy.example.com/v1",
			expectedURL: "https://proxy.example.com/v1",
		},
		{
			name:        "Ollama URL",
			baseURL:     OllamaURL,
			expectedURL: OllamaURL,
		},
		{
			name:        "Gemini compatibility URL",
			baseURL:     GeminiURL,
			expectedURL: GeminiURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewClient(tt.baseURL, "test-key", WithModel("gpt-4"))
			require.NoError(t, err, "Failed to create client with base URL %s", tt.baseURL)
			require.NotNil(t, client)

			// Use the test helper to validate BaseURL
			llmtesting.TestBaseURLConfiguration(t, client, tt.expectedURL)
		})
	}
}

func TestOpenAI_BaseURLPersistence(t *testing.T) {
	t.Parallel()

	// Test that BaseURL is preserved across the client lifetime
	customURL := "https://custom.openai.com/v1"
	client, err := NewClient(customURL, "test-key", WithModel("gpt-4"))
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
