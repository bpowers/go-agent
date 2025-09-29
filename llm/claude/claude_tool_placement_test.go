package claude

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/examples/fstools"
)

// TestClaudeToolResultPlacement tests whether Claude's API allows text content
// before tool_result blocks, or if tool_result must immediately follow tool_use.
//
// This test deliberately constructs a message with:
// 1. Text content block (system reminder)
// 2. Tool result blocks
//
// If Claude accepts this, we can prepend system reminders before tool results.
// If Claude rejects this, we must append system reminders after tool results.
func TestClaudeToolResultPlacement(t *testing.T) {
	// Skip if no API key
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	// Setup test filesystem using os.Root
	root, err := os.OpenRoot(".")
	require.NoError(t, err)
	defer root.Close()

	// Setup context with test filesystem
	ctx := fstools.WithTestFS(context.Background(), root.FS())

	// Create anthropic client directly to make raw API calls
	anthropicClient := anthropic.NewClient(option.WithAPIKey(apiKey))

	// Build tool definition from fstools
	toolDef := fstools.ReadDirToolDef
	toolJSON := toolDef.MCPJsonSchema()
	var toolSchema map[string]interface{}
	err = json.Unmarshal([]byte(toolJSON), &toolSchema)
	require.NoError(t, err)

	// Get the inputSchema from the MCP definition
	var mcp struct {
		InputSchema json.RawMessage `json:"inputSchema"`
	}
	err = json.Unmarshal([]byte(toolJSON), &mcp)
	require.NoError(t, err)

	// Convert inputSchema to ToolInputSchemaParam
	var inputSchemaParam anthropic.ToolInputSchemaParam
	if len(mcp.InputSchema) > 0 {
		err = json.Unmarshal(mcp.InputSchema, &inputSchemaParam)
		require.NoError(t, err)
	}

	// Create the tool parameter for Claude
	toolParam := anthropic.ToolParam{
		Name:        toolDef.Name(),
		Description: anthropic.String(toolDef.Description()),
		InputSchema: inputSchemaParam,
		Type:        anthropic.ToolTypeCustom,
	}

	tools := []anthropic.ToolUnionParam{
		{
			OfTool: &toolParam,
		},
	}

	// Step 1: Initial request to trigger tool call
	t.Log("Step 1: Sending initial message to trigger tool call...")
	initialResponse, err := anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     "claude-sonnet-4-5",
		MaxTokens: 1024,
		System: []anthropic.TextBlockParam{
			{Text: "You are a helpful assistant. Use the read_dir tool to list files.", Type: "text"},
		},
		Tools: tools,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("List the files in the current directory using the read_dir tool.")),
		},
	})
	require.NoError(t, err)

	// Check for tool use in the response
	var toolUseID string
	for _, block := range initialResponse.Content {
		if block.Type == "tool_use" {
			toolUseID = block.ID
			t.Logf("Got tool call: %s with ID: %s", block.Name, block.ID)
			break
		}
	}

	if toolUseID == "" {
		t.Skip("Model did not call tool - skipping test")
	}

	// Execute the tool
	toolResult := fstools.ReadDirTool(ctx, "{}")
	t.Logf("Tool result: %s", toolResult)
	t.Logf("Tool Use ID: %s", toolUseID)

	// Step 2: Send follow-up with TEXT BEFORE tool result
	// This is the experiment: can we put text before tool_result?
	t.Log("Step 2: Sending follow-up message with TEXT BEFORE tool result...")

	// Convert Content blocks to Param blocks
	assistantContent := make([]anthropic.ContentBlockParamUnion, len(initialResponse.Content))
	for i, block := range initialResponse.Content {
		assistantContent[i] = block.ToParam()
	}

	followUpMessages := []anthropic.MessageParam{
		// Original user message
		anthropic.NewUserMessage(anthropic.NewTextBlock("List the files in the current directory using the read_dir tool.")),
		// Assistant's response with tool call
		anthropic.NewAssistantMessage(assistantContent...),
		// User message with TEXT FIRST, then tool result
		anthropic.NewUserMessage(
			// TEXT CONTENT FIRST (system reminder)
			anthropic.NewTextBlock("<system-reminder>Testing text before tool result</system-reminder>"),
			// TOOL RESULT SECOND
			anthropic.NewToolResultBlock(toolUseID, toolResult, false),
		),
	}

	params := anthropic.MessageNewParams{
		Messages:  followUpMessages,
		Model:     "claude-sonnet-4-5",
		MaxTokens: 1024,
		System: []anthropic.TextBlockParam{
			{Text: "You are a helpful assistant. Use the read_dir tool to list files.", Type: "text"},
		},
		Tools: tools,
	}

	t.Log("Attempting to send message with text BEFORE tool result...")

	// Make the API call
	_, err = anthropicClient.Messages.New(ctx, params)
	if err != nil {
		// Expected: Claude enforces the constraint
		t.Log("❌ Claude rejected text BEFORE tool_result")
		t.Logf("Error: %v", err)

		// Verify this is the specific error we expect
		errStr := err.Error()
		if !strings.Contains(errStr, "tool_use") || !strings.Contains(errStr, "tool_result") {
			t.Errorf("Got unexpected error (not about tool_result placement): %v", err)
			return
		}

		t.Log("")
		t.Log("VERIFIED: Claude API constraint confirmed")
		t.Log("- tool_result blocks MUST immediately follow tool_use blocks")
		t.Log("- Text content CANNOT be placed before tool_result")
		t.Log("- Current implementation (append AFTER tool results) is CORRECT")
		t.Log("")
		t.Log("This differs from OpenAI and Gemini which allow text before tool results")
		return
	}

	// If we get here, Claude accepted it - this would be surprising!
	t.Fatal("❌ UNEXPECTED: Claude accepted text BEFORE tool_result - API constraint no longer exists")
}
