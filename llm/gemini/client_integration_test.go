package gemini

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

const provider = "gemini"

func getTestModel() string {
	return "gemini-2.5-flash"
}

func getAPIKey() string {
	return os.Getenv("GEMINI_API_KEY")
}

func TestGeminiIntegration_Streaming(t *testing.T) {
	// Not parallel - helps with rate limiting
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for streaming
	llmtesting.TestStreaming(t, client)
}

func TestGeminiIntegration_ToolCalling(t *testing.T) {
	// Not parallel - helps with rate limiting
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for tool calling
	llmtesting.TestToolsSummarizesFile(t, client)
}

func TestGeminiIntegration_ToolCallingWithContext(t *testing.T) {
	// Not parallel - helps with rate limiting
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for context-aware tool calling
	llmtesting.TestWritesFile(t, client)
}

func TestGeminiIntegration_TokenUsage(t *testing.T) {
	// Not parallel - helps with rate limiting
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Create a chat session
	chatSession := client.NewChat("You are a helpful assistant.")

	// Send a simple message
	ctx := context.Background()
	response, err := chatSession.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Say 'Hello World' and nothing else.",
	})
	require.NoError(t, err)

	// Verify we got a response
	assert.NotEmpty(t, response.Content)

	// Check token usage
	usage, err := chatSession.TokenUsage()
	require.NoError(t, err)

	// Verify we have non-zero token usage
	assert.Greater(t, usage.Cumulative.InputTokens, 0)
	assert.Greater(t, usage.Cumulative.OutputTokens, 0)
	assert.Greater(t, usage.Cumulative.TotalTokens, 0)

	t.Logf("Token usage - Input: %d, Output: %d, Total: %d, Cached: %d",
		usage.Cumulative.InputTokens, usage.Cumulative.OutputTokens, usage.Cumulative.TotalTokens, usage.Cumulative.CachedTokens)

	// Check max tokens
	maxTokens := chatSession.MaxTokens()
	assert.NotZero(t, maxTokens)
	t.Logf("Max tokens for model %s: %d", getTestModel(), maxTokens)
}

func TestGeminiIntegration_TokenUsageCumulative(t *testing.T) {
	// Not parallel - helps with rate limiting
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for cumulative token usage
	llmtesting.TestTokenUsageCumulative(t, client)
}

func TestGeminiIntegration_ToolCallStreamEvents(t *testing.T) {
	// Not parallel - helps with rate limiting
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for tool call stream events
	llmtesting.TestToolCallStreamEvents(t, client)
}

func TestGeminiIntegration_ToolRegistration(t *testing.T) {
	// Not parallel - helps with rate limiting
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
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
	require.NoError(t, err)

	// List tools
	tools := chatSession.ListTools()
	assert.Len(t, tools, 1)
	if len(tools) > 0 {
		assert.Equal(t, "test_tool", tools[0])
	}

	// Deregister tool
	chatSession.DeregisterTool("test_tool")
	tools = chatSession.ListTools()
	assert.Empty(t, tools)
}

func TestGeminiIntegration_MaxTokensByModel(t *testing.T) {
	// Not parallel - helps with rate limiting
	tests := []struct {
		model       string
		expectedMax int
	}{
		{"gemini-2.5-pro", 65536},
		{"gemini-2.5-flash", 65536},
		{"gemini-2.5-flash-lite", 65536},
		{"gemini-2.0-flash", 8192},
		{"gemini-2.0-flash-lite", 8192},
		{"gemini-1.5-pro", 8192},
		{"gemini-1.5-flash", 8192},
		{"gemini-1.5-flash-8b", 8192},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			// Not parallel - helps with rate limiting
			maxTokens := getModelMaxTokens(tt.model)
			assert.Equal(t, tt.expectedMax, maxTokens)
		})
	}
}
