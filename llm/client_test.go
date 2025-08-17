package llm

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsResponsesModel(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		{"GPT-5", "gpt-5", true},
		{"GPT-5 Turbo", "gpt-5-turbo", true},
		{"O1 Preview", "o1-preview", true},
		{"O1 Mini", "o1-mini", true},
		{"O3", "o3", true},
		{"O3 Mini", "o3-mini", true},
		{"GPT-4", "gpt-4", false},
		{"GPT-4o", "gpt-4o", false},
		{"Claude", "claude-3", false},
		{"Gemini", "gemini-pro", false},
		// Case insensitive
		{"GPT-5 Upper", "GPT-5", true},
		{"O1 Upper", "O1-PREVIEW", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := isResponsesModel(tt.model)
			assert.Equal(t, tt.expected, result, "isResponsesModel failed for model: %s", tt.model)
		})
	}
}

func TestDetectProvider(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		model    string
		expected ModelProvider
	}{
		// OpenAI models
		{"GPT-5", "gpt-5", ProviderOpenAI},
		{"GPT-5 Turbo", "gpt-5-turbo", ProviderOpenAI},
		{"GPT-4", "gpt-4", ProviderOpenAI},
		{"GPT-4o", "gpt-4o", ProviderOpenAI},
		{"GPT-4o-mini", "gpt-4o-mini", ProviderOpenAI},
		{"GPT-3.5", "gpt-3.5-turbo", ProviderOpenAI},
		{"O1 Preview", "o1-preview", ProviderOpenAI},
		{"O1 Mini", "o1-mini", ProviderOpenAI},
		{"O3", "o3", ProviderOpenAI},

		// Claude models
		{"Claude 3 Opus", "claude-3-opus-20240229", ProviderClaude},
		{"Claude 3.5 Sonnet", "claude-3-5-sonnet-20241022", ProviderClaude},
		{"Claude 3 Haiku", "claude-3-haiku-20240307", ProviderClaude},
		{"Claude Instant", "claude-instant-1.2", ProviderClaude},

		// Gemini models
		{"Gemini Pro", "gemini-pro", ProviderGemini},
		{"Gemini 1.5 Flash", "gemini-1.5-flash", ProviderGemini},
		{"Gemini 1.5 Pro", "gemini-1.5-pro", ProviderGemini},
		{"Gemini Ultra", "gemini-ultra", ProviderGemini},

		// Ollama models
		{"Llama 2", "llama2", ProviderOllama},
		{"Llama 3", "llama3", ProviderOllama},
		{"Mistral", "mistral", ProviderOllama},
		{"Mixtral", "mixtral", ProviderOllama},
		{"CodeLlama", "codellama", ProviderOllama},
		{"Qwen", "qwen", ProviderOllama},
		{"Phi", "phi", ProviderOllama},
		{"DeepSeek", "deepseek-coder", ProviderOllama},

		// Unknown models
		{"Unknown Model", "unknown-model-xyz", ProviderUnknown},
		{"Random", "random", ProviderUnknown},

		// Case insensitive tests
		{"GPT-4 Upper", "GPT-4", ProviderOpenAI},
		{"Claude Upper", "CLAUDE-3-OPUS", ProviderClaude},
		{"Gemini Mixed", "GeMiNi-PrO", ProviderGemini},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := detectProvider(tt.model)
			assert.Equal(t, tt.expected, result, "Provider detection failed for model: %s", tt.model)
		})
	}
}

func TestNewClient(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		config    *Config
		envVars   map[string]string
		shouldErr bool
		errMsg    string
	}{
		{
			name: "OpenAI with API key",
			config: &Config{
				Model:  "gpt-4",
				APIKey: "test-openai-key",
			},
			shouldErr: false,
		},
		{
			name: "OpenAI with env var",
			config: &Config{
				Model: "gpt-4",
			},
			envVars: map[string]string{
				"OPENAI_API_KEY": "env-openai-key",
			},
			shouldErr: false,
		},
		{
			name: "OpenAI without API key",
			config: &Config{
				Model: "gpt-4",
			},
			shouldErr: true,
			errMsg:    "openAI API key required",
		},
		{
			name: "Claude with API key",
			config: &Config{
				Model:  "claude-3-opus",
				APIKey: "test-claude-key",
			},
			shouldErr: false,
		},
		{
			name: "Claude with env var",
			config: &Config{
				Model: "claude-3-opus",
			},
			envVars: map[string]string{
				"ANTHROPIC_API_KEY": "env-claude-key",
			},
			shouldErr: false,
		},
		{
			name: "Claude without API key",
			config: &Config{
				Model: "claude-3-opus",
			},
			shouldErr: true,
			errMsg:    "anthropic API key required",
		},
		{
			name: "Gemini with API key",
			config: &Config{
				Model:  "gemini-pro",
				APIKey: "test-gemini-key",
			},
			shouldErr: false,
		},
		{
			name: "Gemini with GEMINI_API_KEY env var",
			config: &Config{
				Model: "gemini-pro",
			},
			envVars: map[string]string{
				"GEMINI_API_KEY": "env-gemini-key",
			},
			shouldErr: false,
		},
		{
			name: "Gemini with GOOGLE_API_KEY env var",
			config: &Config{
				Model: "gemini-pro",
			},
			envVars: map[string]string{
				"GOOGLE_API_KEY": "env-google-key",
			},
			shouldErr: false,
		},
		{
			name: "Gemini without API key",
			config: &Config{
				Model: "gemini-pro",
			},
			shouldErr: true,
			errMsg:    "gemini API key required",
		},
		{
			name: "Ollama model (no API key needed)",
			config: &Config{
				Model: "llama2",
			},
			shouldErr: false,
		},
		{
			name: "Unknown model",
			config: &Config{
				Model: "unknown-model",
			},
			shouldErr: true,
			errMsg:    "unknown model provider",
		},
		{
			name: "Debug mode enabled",
			config: &Config{
				Model:  "gpt-4",
				APIKey: "test-key",
				Debug:  true,
			},
			shouldErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Note: Cannot use t.Parallel() here due to environment variable manipulation
			// Clear any existing API key env vars first
			oldOpenAI := os.Getenv("OPENAI_API_KEY")
			oldAnthropic := os.Getenv("ANTHROPIC_API_KEY")
			oldGemini := os.Getenv("GEMINI_API_KEY")
			oldGoogle := os.Getenv("GOOGLE_API_KEY")

			os.Unsetenv("OPENAI_API_KEY")
			os.Unsetenv("ANTHROPIC_API_KEY")
			os.Unsetenv("GEMINI_API_KEY")
			os.Unsetenv("GOOGLE_API_KEY")

			defer func() {
				os.Setenv("OPENAI_API_KEY", oldOpenAI)
				os.Setenv("ANTHROPIC_API_KEY", oldAnthropic)
				os.Setenv("GEMINI_API_KEY", oldGemini)
				os.Setenv("GOOGLE_API_KEY", oldGoogle)
			}()

			// Set up environment variables
			for key, value := range tt.envVars {
				os.Setenv(key, value)
			}

			client, err := NewClient(tt.config)

			if tt.shouldErr {
				assert.Error(t, err)
				if tt.errMsg != "" && err != nil {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
			}
		})
	}
}
