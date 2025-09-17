package gemini

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	llmtesting "github.com/bpowers/go-agent/llm/testing"
)

func TestGemini_BaseURLConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		baseURL     string
		expectedURL string
	}{
		{
			name:        "Default (empty) URL",
			baseURL:     "",
			expectedURL: "",
		},
		{
			name:        "Custom proxy URL",
			baseURL:     "https://proxy.example.com/v1beta",
			expectedURL: "https://proxy.example.com/v1beta",
		},
		{
			name:        "Enterprise gateway URL",
			baseURL:     "https://enterprise.google.com/genai",
			expectedURL: "https://enterprise.google.com/genai",
		},
		{
			name:        "Regional endpoint URL",
			baseURL:     "https://europe-west4-aiplatform.googleapis.com",
			expectedURL: "https://europe-west4-aiplatform.googleapis.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			opts := []Option{WithModel("gemini-1.5-pro")}
			if tt.baseURL != "" {
				opts = append(opts, WithBaseURL(tt.baseURL))
			}

			client, err := NewClient("test-key", opts...)
			require.NoError(t, err, "Failed to create client with base URL %s", tt.baseURL)
			require.NotNil(t, client)

			// Use the test helper to validate BaseURL
			llmtesting.TestBaseURLConfiguration(t, client, tt.expectedURL)
		})
	}
}

func TestGemini_BaseURLPersistence(t *testing.T) {
	t.Parallel()

	// Test that BaseURL is preserved across the client lifetime
	customURL := "https://custom.google.com/genai"
	client, err := NewClient("test-key", WithModel("gemini-1.5-pro"), WithBaseURL(customURL))
	require.NoError(t, err)

	// Validate using the BaseURLValidator interface
	validator, ok := client.(llmtesting.BaseURLValidator)
	require.True(t, ok, "Client should implement BaseURLValidator")

	// Verify the BaseURL is stored correctly
	assert.Equal(t, customURL, validator.BaseURL(), "BaseURL should match what was passed via WithBaseURL")

	// Create a chat session and verify client still has correct BaseURL
	chat := client.NewChat("Test system prompt")
	assert.NotNil(t, chat)

	// BaseURL should still be the same
	assert.Equal(t, customURL, validator.BaseURL(), "BaseURL should remain unchanged after creating chat")
}

func TestGemini_WithoutBaseURLOption(t *testing.T) {
	t.Parallel()

	// Test that without WithBaseURL option, baseURL is empty
	client, err := NewClient("test-key", WithModel("gemini-1.5-pro"))
	require.NoError(t, err)

	// Validate using the BaseURLValidator interface
	validator, ok := client.(llmtesting.BaseURLValidator)
	require.True(t, ok, "Client should implement BaseURLValidator")

	// Verify the BaseURL is empty when not specified
	assert.Equal(t, "", validator.BaseURL(), "BaseURL should be empty when WithBaseURL is not used")
}
