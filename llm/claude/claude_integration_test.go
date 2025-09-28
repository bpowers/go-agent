package claude

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
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
	return "claude-3-5-haiku-latest"
}

func getAPIKey() string {
	return os.Getenv("ANTHROPIC_API_KEY")
}

func TestClaudeIntegration_Streaming(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for streaming
	llmtesting.TestStreaming(t, client)
}

func TestClaudeIntegration_ToolCalling(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for tool calling
	llmtesting.TestToolsSummarizesFile(t, client)
}

func TestClaudeIntegration_ToolCallingWithContext(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for context-aware tool calling
	llmtesting.TestWritesFile(t, client)
}

func TestClaudeIntegration_TokenUsage(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Create a chat session
	chatSession := client.NewChat("You are a helpful assistant.")

	// Send a simple message
	ctx := context.Background()
	response, err := chatSession.Message(ctx, chat.UserMessage("Say 'Hello World' and nothing else."))
	require.NoError(t, err)

	// Verify we got a response
	assert.NotEmpty(t, response.GetText())

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

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for cumulative token usage
	llmtesting.TestTokenUsageCumulative(t, client)
}

func TestClaudeIntegration_ToolCallStreamEvents(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	// Use the test helper for tool call stream events
	llmtesting.TestToolCallStreamEvents(t, client)

	// Test both tool call and result events
	llmtesting.TestToolCallAndResultStreamEvents(t, client)
}

func TestClaudeIntegration_ToolRegistration(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
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

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
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

	// Track if tool was called and with what parameters
	toolCalled := false
	var toolInput string
	err = chatSession.RegisterTool(toolDef, func(ctx context.Context, input string) string {
		toolCalled = true
		toolInput = input

		var req struct {
			Message string `json:"message"`
		}
		if err := json.Unmarshal([]byte(input), &req); err != nil {
			return fmt.Sprintf(`{"error": "failed to parse input: %s"}`, err.Error())
		}
		// Return a deterministic result that we can verify
		return fmt.Sprintf(`{"result": "Echo: %s", "timestamp": "2024-01-01T00:00:00Z"}`, req.Message)
	})
	require.NoError(t, err)

	// Ask Claude to use the tool
	ctx := context.Background()
	response, err := chatSession.Message(ctx, chat.UserMessage("Please use the echo tool to echo back the message 'Hello World'"))
	require.NoError(t, err)

	// Verify the tool was actually called
	if !toolCalled {
		t.Error("Expected tool to be called, but it wasn't")
	}

	// Verify the tool received the correct input parameters
	var parsedInput struct {
		Message string `json:"message"`
	}
	err = json.Unmarshal([]byte(toolInput), &parsedInput)
	require.NoError(t, err, "Tool input should be valid JSON")

	// Check that the input contains the expected message
	// The LLM might format it slightly differently (e.g., "Hello World" vs "Hello World!")
	if !strings.Contains(strings.ToLower(parsedInput.Message), "hello") ||
		!strings.Contains(strings.ToLower(parsedInput.Message), "world") {
		t.Errorf("Tool input doesn't contain expected message. Got: %s", parsedInput.Message)
	}

	// Verify the response exists and contains meaningful content
	responseText := response.GetText()
	if responseText == "" {
		t.Error("Expected non-empty response content")
	}

	// The response should acknowledge the echo operation in some way
	// Different LLMs might phrase this differently, so we check for key concepts
	responseLower := strings.ToLower(responseText)
	hasEchoMention := strings.Contains(responseLower, "echo") ||
		strings.Contains(responseLower, "hello world") ||
		strings.Contains(responseLower, "message")

	if !hasEchoMention {
		t.Logf("Response doesn't appear to reference the echo operation: %s", responseText)
		// Don't fail the test as LLMs might express this differently
	}

	t.Logf("Tool calling test passed - Tool called: %v, Input: %s, Response: %s",
		toolCalled, toolInput, responseText)
}

func TestClaudeIntegration_SystemReminderWithToolCalls(t *testing.T) {
	t.Parallel()
	llmtesting.SkipIfNoAPIKey(t, provider)

	client, err := NewClient(AnthropicURL, getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err, "Failed to create Claude client")
	require.NotNil(t, client)

	// Use the test helper for system reminder with tool calls
	llmtesting.TestSystemReminderWithToolCalls(t, client)
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
