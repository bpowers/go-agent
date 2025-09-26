package openai

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
	llmtesting "github.com/bpowers/go-agent/llm/testing"
)

// testToolDef implements chat.ToolDef for testing
type testToolDef struct {
	name        string
	description string
	jsonSchema  string
}

func (t *testToolDef) MCPJsonSchema() string {
	return t.jsonSchema
}

func (t *testToolDef) Name() string {
	return t.name
}

func (t *testToolDef) Description() string {
	return t.description
}

const provider = "openai"

func getTestModel() string {
	return "gpt-5-nano"
}

func getAPIKey() string {
	return os.Getenv("OPENAI_API_KEY")
}

func TestOpenAIIntegration_Streaming(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	tests := []struct {
		name string
		api  API
	}{
		{"ChatCompletions", ChatCompletions},
		{"Responses", Responses},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			client, err := NewClient(OpenAIURL, getAPIKey(), WithModel(getTestModel()), WithAPI(tt.api))
			require.NoError(t, err, "Failed to create OpenAI client")
			require.NotNil(t, client)

			// Use the test helper for streaming
			llmtesting.TestStreaming(t, client)
		})
	}
}

func TestOpenAIIntegration_ToolCalling(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	tests := []struct {
		name string
		api  API
	}{
		{"ChatCompletions", ChatCompletions},
		{"Responses", Responses},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			client, err := NewClient(OpenAIURL, getAPIKey(), WithModel(getTestModel()), WithAPI(tt.api))
			require.NoError(t, err, "Failed to create OpenAI client")
			require.NotNil(t, client)

			// Use the test helper for tool calling
			llmtesting.TestToolsSummarizesFile(t, client)
		})
	}
}

func TestOpenAIIntegration_ToolCallingWithContext(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	tests := []struct {
		name string
		api  API
	}{
		{"ChatCompletions", ChatCompletions},
		{"Responses", Responses},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			client, err := NewClient(OpenAIURL, getAPIKey(), WithModel(getTestModel()), WithAPI(tt.api))
			require.NoError(t, err, "Failed to create OpenAI client")
			require.NotNil(t, client)

			// Use the test helper for context-aware tool calling
			llmtesting.TestWritesFile(t, client)
		})
	}
}

func TestOpenAIIntegration_TokenUsage(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	tests := []struct {
		name string
		api  API
	}{
		{"ChatCompletions", ChatCompletions},
		{"Responses", Responses},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			client, err := NewClient(OpenAIURL, getAPIKey(), WithModel(getTestModel()), WithAPI(tt.api))
			require.NoError(t, err, "Failed to create OpenAI client")
			require.NotNil(t, client)

			// Create a chat session
			chatSession := client.NewChat("You are a helpful assistant.")

			// Send a simple message
			ctx := context.Background()
			response, err := chatSession.Message(ctx, chat.Message{
				Role:    chat.UserRole,
				Content: "Say 'Hello World' and nothing else.",
			})
			require.NoError(t, err, "Failed to send message")

			// Verify we got a response
			assert.NotEmpty(t, response.Content, "Expected non-empty response")

			// Check token usage
			usage, err := chatSession.TokenUsage()
			require.NoError(t, err, "Failed to get token usage")

			// Verify we have non-zero token usage
			assert.Greater(t, usage.Cumulative.InputTokens, 0, "Expected input tokens to be greater than 0")
			assert.Greater(t, usage.Cumulative.OutputTokens, 0, "Expected output tokens to be greater than 0")
			assert.Greater(t, usage.Cumulative.TotalTokens, 0, "Expected total tokens to be greater than 0")

			t.Logf("[%s] Token usage - Input: %d, Output: %d, Total: %d, Cached: %d",
				tt.name, usage.Cumulative.InputTokens, usage.Cumulative.OutputTokens, usage.Cumulative.TotalTokens, usage.Cumulative.CachedTokens)

			// Check max tokens
			maxTokens := chatSession.MaxTokens()
			assert.NotZero(t, maxTokens, "Expected non-zero max tokens")
			t.Logf("[%s] Max tokens for model %s: %d", tt.name, getTestModel(), maxTokens)
		})
	}
}

func TestOpenAIIntegration_TokenUsageCumulative(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	tests := []struct {
		name string
		api  API
	}{
		{"ChatCompletions", ChatCompletions},
		{"Responses", Responses},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			client, err := NewClient(OpenAIURL, getAPIKey(), WithModel(getTestModel()), WithAPI(tt.api))
			require.NoError(t, err, "Failed to create OpenAI client")
			require.NotNil(t, client)

			// Use the test helper for cumulative token usage
			llmtesting.TestTokenUsageCumulative(t, client)
		})
	}
}

func TestOpenAIIntegration_ToolCallStreamEvents(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	tests := []struct {
		name string
		api  API
	}{
		{"ChatCompletions", ChatCompletions},
		// Note: Responses API doesn't support tools yet
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			client, err := NewClient(OpenAIURL, getAPIKey(), WithModel(getTestModel()), WithAPI(tt.api))
			require.NoError(t, err, "Failed to create OpenAI client")
			require.NotNil(t, client)

			// Use the test helper for tool call stream events
			llmtesting.TestToolCallStreamEvents(t, client)

			// Test both tool call and result events
			llmtesting.TestToolCallAndResultStreamEvents(t, client)
		})
	}
}

func TestOpenAIIntegration_ToolRegistration(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	tests := []struct {
		name string
		api  API
	}{
		{"ChatCompletions", ChatCompletions},
		{"Responses", Responses},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			client, err := NewClient(OpenAIURL, getAPIKey(), WithModel(getTestModel()), WithAPI(tt.api))
			require.NoError(t, err, "Failed to create OpenAI client")
			require.NotNil(t, client)

			chatSession := client.NewChat("You are a helpful assistant.")

			// Register a simple tool
			toolDef := &testToolDef{
				name:        "test_tool",
				description: "A test tool",
				jsonSchema: `{
					"name": "test_tool",
					"description": "A test tool",
					"inputSchema": {
						"type": "object",
						"properties": {
							"message": {"type": "string"}
						}
					}
				}`,
			}

			err = chatSession.RegisterTool(toolDef, func(ctx context.Context, input string) string {
				return `{"result": "Tool called successfully"}`
			})
			require.NoError(t, err, "Failed to register tool")

			// List tools
			tools := chatSession.ListTools()
			assert.Len(t, tools, 1, "Expected 1 tool")
			if len(tools) > 0 {
				assert.Equal(t, "test_tool", tools[0], "Expected tool name 'test_tool'")
			}

			// Deregister tool
			chatSession.DeregisterTool("test_tool")
			tools = chatSession.ListTools()
			assert.Empty(t, tools, "Expected no tools after deregistration")
		})
	}
}

func TestOpenAIIntegration_MaxTokensByModel(t *testing.T) {
	t.Parallel()
	tests := []struct {
		model       string
		expectedMax int
	}{
		{"gpt-5", 128000},
		{"gpt-5-mini", 128000},
		{"gpt-5-nano", 128000},
		{"gpt-4.5-preview", 16384},
		{"gpt-4.1", 32768},
		{"gpt-4.1-mini", 32768},
		{"gpt-4o", 16384},
		{"gpt-4o-mini", 16384},
		{"gpt-4-turbo", 4096},
		{"gpt-4", 8192},
		{"o3", 100000},
		{"o3-mini", 100000},
		{"o4-mini", 100000},
		{"gpt-3.5-turbo", 4096},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()
			maxTokens := getModelMaxTokens(tt.model)
			assert.Equal(t, tt.expectedMax, maxTokens)
		})
	}
}
