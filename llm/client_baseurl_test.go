package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/llm/claude"
	"github.com/bpowers/go-agent/llm/openai"
	llmtesting "github.com/bpowers/go-agent/llm/testing"
)

func TestNewClient_BaseURLConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		config      *Config
		expectedURL string
	}{
		{
			name: "OpenAI with default URL",
			config: &Config{
				Model:  "gpt-4",
				APIKey: "test-key",
			},
			expectedURL: openai.OpenAIURL,
		},
		{
			name: "OpenAI with custom URL",
			config: &Config{
				Model:   "gpt-4",
				APIKey:  "test-key",
				BaseURL: "https://custom.openai.com/v1",
			},
			expectedURL: "https://custom.openai.com/v1",
		},
		{
			name: "Claude with default URL",
			config: &Config{
				Model:  "claude-opus-4",
				APIKey: "test-key",
			},
			expectedURL: claude.AnthropicURL,
		},
		{
			name: "Claude with custom URL",
			config: &Config{
				Model:   "claude-opus-4",
				APIKey:  "test-key",
				BaseURL: "https://custom.anthropic.com/v1",
			},
			expectedURL: "https://custom.anthropic.com/v1",
		},
		{
			name: "Gemini with default URL",
			config: &Config{
				Model:  "gemini-1.5-pro",
				APIKey: "test-key",
			},
			expectedURL: "", // Gemini default is empty (uses SDK default)
		},
		{
			name: "Gemini with custom URL",
			config: &Config{
				Model:   "gemini-1.5-pro",
				APIKey:  "test-key",
				BaseURL: "https://custom.gemini.com",
			},
			expectedURL: "https://custom.gemini.com",
		},
		{
			name: "Ollama with default URL",
			config: &Config{
				Model: "llama2",
			},
			expectedURL: openai.OllamaURL,
		},
		{
			name: "Ollama with custom URL",
			config: &Config{
				Model:   "llama2",
				BaseURL: "http://remote-ollama:11434/v1",
			},
			expectedURL: "http://remote-ollama:11434/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewClient(tt.config)
			require.NoError(t, err, "Failed to create client")
			require.NotNil(t, client)

			// Use the test helper to validate BaseURL
			llmtesting.TestBaseURLConfiguration(t, client, tt.expectedURL)
		})
	}
}

func TestNewClient_BaseURLOverridesDefaults(t *testing.T) {
	t.Parallel()

	// Test that BaseURL in Config overrides provider defaults
	customURL := "https://enterprise-proxy.example.com/v1"

	providers := []struct {
		model    string
		provider string
	}{
		{"gpt-4", "OpenAI"},
		{"claude-opus-4", "Claude"},
		{"gemini-1.5-pro", "Gemini"},
		{"llama2", "Ollama"},
	}

	for _, p := range providers {
		t.Run(p.provider, func(t *testing.T) {
			t.Parallel()

			config := &Config{
				Model:   p.model,
				APIKey:  "test-key",
				BaseURL: customURL,
			}

			client, err := NewClient(config)
			require.NoError(t, err, "Failed to create %s client with custom BaseURL", p.provider)
			require.NotNil(t, client)

			// For Gemini, the custom URL should be used
			expectedURL := customURL
			if p.provider == "Gemini" {
				// Gemini stores the exact URL passed
				llmtesting.TestBaseURLConfiguration(t, client, customURL)
			} else {
				// Other providers should use the custom URL
				llmtesting.TestBaseURLConfiguration(t, client, expectedURL)
			}
		})
	}
}

func TestNewClient_EmptyBaseURLUsesProviderDefaults(t *testing.T) {
	t.Parallel()

	tests := []struct {
		model       string
		provider    string
		expectedURL string
	}{
		{"gpt-4", "OpenAI", openai.OpenAIURL},
		{"claude-opus-4", "Claude", claude.AnthropicURL},
		{"gemini-1.5-pro", "Gemini", ""}, // Gemini default is empty
		{"llama2", "Ollama", openai.OllamaURL},
	}

	for _, tt := range tests {
		t.Run(tt.provider, func(t *testing.T) {
			t.Parallel()

			config := &Config{
				Model:   tt.model,
				APIKey:  "test-key",
				BaseURL: "", // Explicitly empty
			}

			client, err := NewClient(config)
			require.NoError(t, err, "Failed to create %s client with empty BaseURL", tt.provider)
			require.NotNil(t, client)

			// Validate the default URL is used
			llmtesting.TestBaseURLConfiguration(t, client, tt.expectedURL)
		})
	}
}

func TestNewClient_BaseURLValidation(t *testing.T) {
	t.Parallel()

	// Test that clients are created successfully with various BaseURL formats
	urls := []string{
		"https://api.example.com/v1",
		"http://localhost:8080/v1",
		"https://proxy.corp.net:443/openai/v1",
		"http://192.168.1.100:11434/v1",
		"https://region-specific.provider.com/api",
	}

	for _, url := range urls {
		t.Run(url, func(t *testing.T) {
			t.Parallel()

			config := &Config{
				Model:   "gpt-4",
				APIKey:  "test-key",
				BaseURL: url,
			}

			client, err := NewClient(config)
			assert.NoError(t, err, "Should create client with URL: %s", url)
			assert.NotNil(t, client)

			// Verify the URL was accepted
			llmtesting.TestBaseURLConfiguration(t, client, url)
		})
	}
}
