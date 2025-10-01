package llm

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/internal/logging"
	"github.com/bpowers/go-agent/llm/claude"
	"github.com/bpowers/go-agent/llm/gemini"
	"github.com/bpowers/go-agent/llm/openai"
)

var logger = logging.Logger().With("component", "llm")

// Config holds the LLM client configuration
type Config struct {
	Model        string
	Provider     string
	APIKey       string
	BaseURL      string            // Optional base URL override for the API endpoint
	Headers      map[string]string // Optional custom HTTP headers
	Temperature  float64
	MaxTokens    int
	SystemPrompt string
	// LogLevel sets the library-wide log level (affects all providers).
	// Values: -1=don't change (default), 0=Error, 1=Warn, 2=Info, 3=Debug
	// Note: This is a global setting that affects all LLM providers in the process.
	LogLevel int
}

// ModelProvider represents the different LLM providers
type ModelProvider int

const (
	ProviderOpenAI ModelProvider = iota
	ProviderClaude
	ProviderGemini
	ProviderOllama
	ProviderUnknown
)

// NewClient creates a chat client based on the configuration
func NewClient(config *Config) (chat.Client, error) {
	// Set log level if specified (affects library-wide logging)
	if config.LogLevel >= 0 && config.LogLevel <= 3 {
		levels := []slog.Level{
			slog.LevelError, // 0
			slog.LevelWarn,  // 1
			slog.LevelInfo,  // 2
			slog.LevelDebug, // 3
		}
		SetLogLevel(levels[config.LogLevel])
	}

	provider := detectProvider(config.Model, config.Provider)
	apiKey := config.APIKey

	switch provider {
	case ProviderOpenAI:
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf("openAI API key required (set -api-key or OPENAI_API_KEY)")
		}

		opts := []openai.Option{
			openai.WithModel(config.Model),
		}

		// Use Responses API for gpt-5, o1, and o3 models
		if isResponsesModel(config.Model) {
			opts = append(opts, openai.WithAPI(openai.Responses))
		}

		if config.Headers != nil {
			opts = append(opts, openai.WithHeaders(config.Headers))
		}

		baseURL := config.BaseURL
		if baseURL == "" {
			baseURL = openai.OpenAIURL
		}
		logger.Info("using OpenAI client", "model", config.Model)
		return openai.NewClient(baseURL, apiKey, opts...)

	case ProviderClaude:
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf("anthropic API key required (set -api-key or ANTHROPIC_API_KEY)")
		}

		opts := []claude.Option{
			claude.WithModel(config.Model),
		}

		if config.Headers != nil {
			opts = append(opts, claude.WithHeaders(config.Headers))
		}

		baseURL := config.BaseURL
		if baseURL == "" {
			baseURL = claude.AnthropicURL
		}
		logger.Info("using Claude client", "model", config.Model)
		return claude.NewClient(baseURL, apiKey, opts...)

	case ProviderGemini:
		if apiKey == "" {
			apiKey = os.Getenv("GEMINI_API_KEY")
			if apiKey == "" {
				apiKey = os.Getenv("GOOGLE_API_KEY")
			}
		}
		if apiKey == "" {
			return nil, fmt.Errorf("gemini API key required (set -api-key, GEMINI_API_KEY, or GOOGLE_API_KEY)")
		}

		opts := []gemini.Option{
			gemini.WithModel(config.Model),
		}
		if config.BaseURL != "" {
			opts = append(opts, gemini.WithBaseURL(config.BaseURL))
		}
		if config.Headers != nil {
			opts = append(opts, gemini.WithHeaders(config.Headers))
		}

		logger.Info("using Gemini client", "model", config.Model)
		return gemini.NewClient(apiKey, opts...)

	case ProviderOllama:
		// Ollama doesn't require an API key
		opts := []openai.Option{
			openai.WithModel(config.Model),
		}
		if config.Headers != nil {
			opts = append(opts, openai.WithHeaders(config.Headers))
		}

		baseURL := config.BaseURL
		if baseURL == "" {
			baseURL = openai.OllamaURL
		}
		logger.Info("using OpenAI client locally w/ ollama", "model", config.Model)
		return openai.NewClient(baseURL, "", opts...)

	default:
		return nil, fmt.Errorf("unknown model provider for model: %s", config.Model)
	}
}

// isResponsesModel checks if the model should use the Responses API
func isResponsesModel(model string) bool {
	modelLower := strings.ToLower(model)
	// gpt-5, o1, and o3 models use the Responses API
	return strings.HasPrefix(modelLower, "gpt-5") ||
		strings.HasPrefix(modelLower, "o1-") ||
		strings.HasPrefix(modelLower, "o3")
}

// detectProvider detects the provider from the model name
func detectProvider(model, provider string) ModelProvider {
	if provider != "" {
		switch provider {
		case "openai":
			return ProviderOpenAI
		case "anthropic":
			return ProviderClaude
		case "google":
			return ProviderGemini
		case "ollama":
			return ProviderOllama
		}
	}

	modelLower := strings.ToLower(model)

	// OpenAI models
	if strings.HasPrefix(modelLower, "gpt-") ||
		strings.HasPrefix(modelLower, "o1-") ||
		strings.HasPrefix(modelLower, "o3") { // o3 doesn't have a dash
		return ProviderOpenAI
	}

	// Claude models
	if strings.HasPrefix(modelLower, "claude-") {
		return ProviderClaude
	}

	// Gemini models
	if strings.HasPrefix(modelLower, "gemini-") {
		return ProviderGemini
	}

	// Ollama models (common ones)
	if strings.HasPrefix(modelLower, "llama") ||
		strings.HasPrefix(modelLower, "mistral") ||
		strings.HasPrefix(modelLower, "mixtral") ||
		strings.HasPrefix(modelLower, "qwen") ||
		strings.HasPrefix(modelLower, "phi") ||
		strings.HasPrefix(modelLower, "deepseek") ||
		strings.HasPrefix(modelLower, "codellama") {
		return ProviderOllama
	}

	return ProviderUnknown
}
