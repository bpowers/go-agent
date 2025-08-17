package claude

import (
	"context"
	"encoding/json"
	"fmt"
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

const provider = "claude"

func getTestModel() string {
	return "claude-3-5-haiku-20241022"
}

func getAPIKey() string {
	return os.Getenv("ANTHROPIC_API_KEY")
}

func TestClaudeIntegration_Streaming(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for streaming
	llmtesting.TestStreaming(t, client)
}

func TestClaudeIntegration_ToolCalling(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for tool calling
	llmtesting.TestToolsSummarizesFile(t, client)
}

func TestClaudeIntegration_ToolCallingWithContext(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for context-aware tool calling
	llmtesting.TestWritesFile(t, client)
}

func TestClaudeIntegration_TokenUsage(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
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

func TestClaudeIntegration_TokenUsageCumulative(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for cumulative token usage
	llmtesting.TestTokenUsageCumulative(t, client)
}

func TestClaudeIntegration_ToolCallStreamEvents(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for tool call stream events
	llmtesting.TestToolCallStreamEvents(t, client)
}

func TestClaudeIntegration_ToolRegistration(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
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

func TestClaudeIntegration_SimpleToolCall(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(ClaudeURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	chatSession := client.NewChat("You are a helpful assistant.")

	// Register a simple echo tool
	toolDef := &testToolDef{
		name:        "echo",
		description: "Echo back the provided message",
		jsonSchema: `{
			"name": "echo",
			"description": "Echo back the provided message",
			"inputSchema": {
				"type": "object",
				"properties": {
					"message": {
						"type": "string",
						"description": "The message to echo back"
					}
				},
				"required": ["message"]
			}
		}`,
	}

	err = chatSession.RegisterTool(toolDef, func(ctx context.Context, input string) string {
		var req struct {
			Message string `json:"message"`
		}
		if err := json.Unmarshal([]byte(input), &req); err != nil {
			return fmt.Sprintf(`{"error": "failed to parse input: %s"}`, err.Error())
		}
		return fmt.Sprintf(`{"result": "Echo: %s"}`, req.Message)
	})
	require.NoError(t, err)

	// Ask Claude to use the tool
	ctx := context.Background()
	response, err := chatSession.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Please use the echo tool to echo back the message 'Hello World'",
	})

	// Just make sure we get some response and no error for now
	require.NoError(t, err)
	assert.NotEmpty(t, response.Content)

	t.Logf("Response: %s", response.Content)
}

func TestClaudeIntegration_MaxTokensByModel(t *testing.T) {
	t.Parallel()
	tests := []struct {
		model       string
		expectedMax int
	}{
		{"claude-opus-4-1", 32000},
		{"claude-opus-4", 32000},
		{"claude-sonnet-4", 64000},
		{"claude-3-7-sonnet", 64000},
		{"claude-3-5-haiku", 8192},
		{"claude-3-haiku", 4096},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()
			maxTokens := getModelMaxTokens(tt.model)
			assert.Equal(t, tt.expectedMax, maxTokens)
		})
	}
}
