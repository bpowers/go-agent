package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/llm/claude"
	"github.com/bpowers/go-agent/llm/openai"
	llmtesting "github.com/bpowers/go-agent/llm/testing"
)

func TestNewClient_HeadersPropagation(t *testing.T) {
	t.Parallel()

	headers := map[string]string{
		"X-Custom-Header":  "custom-value",
		"X-Request-ID":     "req-123",
		"X-Correlation-ID": "corr-456",
	}

	tests := []struct {
		name     string
		model    string
		provider string
		headers  map[string]string
		apiKey   string
	}{
		{
			name:     "OpenAI with headers",
			model:    "gpt-4",
			provider: "OpenAI",
			headers:  headers,
			apiKey:   "test-openai-key",
		},
		{
			name:     "Claude with headers",
			model:    "claude-opus-4",
			provider: "Claude",
			headers:  headers,
			apiKey:   "test-claude-key",
		},
		{
			name:     "Gemini with headers",
			model:    "gemini-1.5-pro",
			provider: "Gemini",
			headers:  headers,
			apiKey:   "test-gemini-key",
		},
		{
			name:     "Ollama with headers",
			model:    "llama2",
			provider: "Ollama",
			headers:  headers,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			config := &Config{
				Model:   tt.model,
				APIKey:  tt.apiKey,
				Headers: tt.headers,
			}

			client, err := NewClient(config)
			require.NoError(t, err, "Failed to create client for %s", tt.provider)

			llmtesting.TestHeaderConfiguration(t, client, tt.headers)
		})
	}
}

func TestNewClient_NilHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		model    string
		provider string
		apiKey   string
	}{
		{
			name:     "OpenAI with nil headers",
			model:    "gpt-4",
			provider: "OpenAI",
			apiKey:   "test-openai-key",
		},
		{
			name:     "Claude with nil headers",
			model:    "claude-opus-4",
			provider: "Claude",
			apiKey:   "test-claude-key",
		},
		{
			name:     "Gemini with nil headers",
			model:    "gemini-1.5-pro",
			provider: "Gemini",
			apiKey:   "test-gemini-key",
		},
		{
			name:     "Ollama with nil headers",
			model:    "llama2",
			provider: "Ollama",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			config := &Config{
				Model:   tt.model,
				APIKey:  tt.apiKey,
				Headers: nil,
			}

			client, err := NewClient(config)
			require.NoError(t, err, "Failed to create client for %s", tt.provider)

			validator, ok := client.(llmtesting.HeadersValidator)
			if !ok {
				t.Skip("Client doesn't implement HeadersValidator interface")
			}

			actualHeaders := validator.Headers()
			assert.Nil(t, actualHeaders, "Expected nil headers for %s", tt.provider)
		})
	}
}

func TestNewClient_EmptyHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		model    string
		provider string
		apiKey   string
	}{
		{
			name:     "OpenAI with empty headers",
			model:    "gpt-4",
			provider: "OpenAI",
			apiKey:   "test-openai-key",
		},
		{
			name:     "Claude with empty headers",
			model:    "claude-opus-4",
			provider: "Claude",
			apiKey:   "test-claude-key",
		},
		{
			name:     "Gemini with empty headers",
			model:    "gemini-1.5-pro",
			provider: "Gemini",
			apiKey:   "test-gemini-key",
		},
		{
			name:     "Ollama with empty headers",
			model:    "llama2",
			provider: "Ollama",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			config := &Config{
				Model:   tt.model,
				APIKey:  tt.apiKey,
				Headers: map[string]string{},
			}

			client, err := NewClient(config)
			require.NoError(t, err, "Failed to create client for %s", tt.provider)

			validator, ok := client.(llmtesting.HeadersValidator)
			if !ok {
				t.Skip("Client doesn't implement HeadersValidator interface")
			}

			actualHeaders := validator.Headers()
			assert.Empty(t, actualHeaders, "Expected empty headers for %s", tt.provider)
		})
	}
}

func TestNewClient_HeadersWithBaseURL(t *testing.T) {
	t.Parallel()

	customURL := "https://custom.example.com/v1"
	headers := map[string]string{
		"X-Custom-Header": "custom-value",
		"Authorization":   "Bearer special-token",
	}

	tests := []struct {
		name        string
		model       string
		provider    string
		apiKey      string
		expectedURL string
	}{
		{
			name:        "OpenAI with headers and custom URL",
			model:       "gpt-4",
			provider:    "OpenAI",
			apiKey:      "test-key",
			expectedURL: customURL,
		},
		{
			name:        "Claude with headers and custom URL",
			model:       "claude-opus-4",
			provider:    "Claude",
			apiKey:      "test-key",
			expectedURL: customURL,
		},
		{
			name:        "Ollama with headers and custom URL",
			model:       "llama2",
			provider:    "Ollama",
			expectedURL: customURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			config := &Config{
				Model:   tt.model,
				APIKey:  tt.apiKey,
				BaseURL: customURL,
				Headers: headers,
			}

			client, err := NewClient(config)
			require.NoError(t, err, "Failed to create client for %s", tt.provider)

			llmtesting.TestHeaderConfiguration(t, client, headers)

			if tt.provider != "Gemini" {
				llmtesting.TestBaseURLConfiguration(t, client, tt.expectedURL)
			}
		})
	}
}

func TestNewClient_HeadersConsistency(t *testing.T) {
	t.Parallel()

	headers := map[string]string{
		"X-Test-Header": "test-value",
	}

	t.Run("OpenAI headers match provider implementation", func(t *testing.T) {
		t.Parallel()

		config := &Config{
			Model:   "gpt-4",
			APIKey:  "test-key",
			Headers: headers,
		}

		llmClient, err := NewClient(config)
		require.NoError(t, err)

		directClient, err := openai.NewClient(
			openai.OpenAIURL,
			"test-key",
			openai.WithModel("gpt-4"),
			openai.WithHeaders(headers),
		)
		require.NoError(t, err)

		llmValidator, ok := llmClient.(llmtesting.HeadersValidator)
		require.True(t, ok, "llm client must implement HeadersValidator")

		directValidator, ok := directClient.(llmtesting.HeadersValidator)
		require.True(t, ok, "direct client must implement HeadersValidator")

		assert.Equal(t, directValidator.Headers(), llmValidator.Headers(),
			"Headers should match between llm.NewClient and direct provider client")
	})

	t.Run("Claude headers match provider implementation", func(t *testing.T) {
		t.Parallel()

		config := &Config{
			Model:   "claude-opus-4",
			APIKey:  "test-key",
			Headers: headers,
		}

		llmClient, err := NewClient(config)
		require.NoError(t, err)

		directClient, err := claude.NewClient(
			claude.AnthropicURL,
			"test-key",
			claude.WithModel("claude-opus-4"),
			claude.WithHeaders(headers),
		)
		require.NoError(t, err)

		llmValidator, ok := llmClient.(llmtesting.HeadersValidator)
		require.True(t, ok, "llm client must implement HeadersValidator")

		directValidator, ok := directClient.(llmtesting.HeadersValidator)
		require.True(t, ok, "direct client must implement HeadersValidator")

		assert.Equal(t, directValidator.Headers(), llmValidator.Headers(),
			"Headers should match between llm.NewClient and direct provider client")
	})
}
