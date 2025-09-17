package llm

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/claude"
	"github.com/bpowers/go-agent/llm/gemini"
	"github.com/bpowers/go-agent/llm/openai"
)

// Config holds the LLM client configuration
type Config struct {
	Model        string
	APIKey       string
	BaseURL      string // Optional base URL override for the API endpoint
	Temperature  float64
	MaxTokens    int
	SystemPrompt string
	Debug        bool
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
	provider := detectProvider(config.Model)
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

		if config.Debug {
			opts = append(opts, openai.WithDebug(true))
		}

		baseURL := config.BaseURL
		if baseURL == "" {
			baseURL = openai.OpenAIURL
		}
		log.Printf("Using OpenAI client with model %q\n", config.Model)
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
		if config.Debug {
			opts = append(opts, claude.WithDebug(true))
		}

		baseURL := config.BaseURL
		if baseURL == "" {
			baseURL = claude.AnthropicURL
		}
		log.Printf("Using Claude client with model %q\n", config.Model)
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
		if config.Debug {
			opts = append(opts, gemini.WithDebug(true))
		}

		log.Printf("Using Gemini client with model %q\n", config.Model)
		return gemini.NewClient(apiKey, opts...)

	case ProviderOllama:
		// Ollama doesn't require an API key
		opts := []openai.Option{
			openai.WithModel(config.Model),
		}
		if config.Debug {
			opts = append(opts, openai.WithDebug(true))
		}

		baseURL := config.BaseURL
		if baseURL == "" {
			baseURL = openai.OllamaURL
		}
		log.Printf("Using OpenAI client locally w/ ollama with model %q\n", config.Model)
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
func detectProvider(model string) ModelProvider {
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
