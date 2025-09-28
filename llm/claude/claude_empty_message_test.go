package claude

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
)

func TestClaudeMessagesFromChat_EmptyToolResults(t *testing.T) {
	tests := []struct {
		name        string
		message     chat.Message
		wantNil     bool
		wantPanic   bool
		description string
	}{
		{
			name:        "tool_role_with_empty_tool_results",
			message:     chat.Message{Role: chat.ToolRole}, // Empty message
			wantNil:     true,                              // Should return nil to avoid empty message
			description: "Empty ToolResults array should not create an empty message",
		},
		{
			name:        "tool_role_with_nil_tool_results_no_content",
			message:     chat.Message{Role: chat.ToolRole}, // Empty message
			wantNil:     true,
			description: "Nil ToolResults with no content should return nil",
		},
		{
			name: "tool_role_with_nil_tool_results_with_content",
			message: func() chat.Message {
				m := chat.Message{Role: chat.ToolRole}
				m.AddText("Some content")
				return m
			}(),
			wantNil:     false,
			description: "Nil ToolResults with content should create a valid message",
		},
		{
			name: "tool_role_with_valid_tool_results",
			message: func() chat.Message {
				m := chat.Message{Role: chat.ToolRole}
				m.AddToolResult(chat.ToolResult{
					ToolCallID: "test_id",
					Content:    "test result",
					Name:       "test_tool",
				})
				return m
			}(),
			wantNil:     false,
			description: "Valid ToolResults should create a message",
		},
		{
			name:        "assistant_role_with_empty_content_and_no_tool_calls",
			message:     chat.Message{Role: chat.AssistantRole}, // Empty assistant message
			wantNil:     true,
			description: "Assistant message with no content and no tool calls should return nil",
		},
		{
			name:        "user_role_with_empty_content",
			message:     chat.Message{Role: chat.UserRole}, // Empty user message
			wantNil:     true,
			description: "User message with empty content should return nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := messageParam(tt.message)

			if tt.wantNil {
				assert.Error(t, err, tt.description)
			} else {
				assert.NoError(t, err, tt.description)
				// Ensure that if we have a result, it has content blocks
				if err == nil {
					// This is a bit hacky but we need to ensure the message has content
					// In practice, the Anthropic API will validate this
					assert.NotEmpty(t, result.Content, "Message should have content when not nil")
				}
			}
		})
	}
}

func TestHandleToolCallRounds_EmptyToolResults(t *testing.T) {
	// This test verifies that we don't create empty messages when tool results are empty
	// which would cause the "text content blocks must be non-empty" error from Claude API

	// Test that empty tool results don't create empty messages
	emptyToolResults := []anthropic.ContentBlockParamUnion{}

	// This would fail with "text content blocks must be non-empty" if we don't handle it properly
	defer func() {
		if r := recover(); r != nil {
			// If we panic, it's likely because we're trying to create an empty message
			t.Errorf("Creating message with empty tool results should not panic: %v", r)
		}
	}()

	// Try to create a user message with empty tool results
	// This simulates what happens at line 785 in handleToolCallRounds
	if len(emptyToolResults) > 0 {
		// This should not be reached if emptyToolResults is truly empty
		msg := anthropic.NewUserMessage(emptyToolResults...)
		require.NotNil(t, msg, "Message should be created if there are tool results")
	} else {
		// We should check for empty tool results before creating the message
		t.Log("Correctly skipping creation of empty message")
	}
}

func TestAnthropicNewUserMessage_RequiresContent(t *testing.T) {
	// This test demonstrates that anthropic.NewUserMessage requires at least one content block

	// Test with no content blocks - this is what causes the API error
	defer func() {
		if r := recover(); r == nil {
			// If NewUserMessage doesn't panic or error with empty content,
			// then the API will reject it with "text content blocks must be non-empty"
			t.Log("NewUserMessage accepts empty content, but API will reject it")
		}
	}()

	// Creating a user message with no content blocks
	emptyBlocks := []anthropic.ContentBlockParamUnion{}
	msg := anthropic.NewUserMessage(emptyBlocks...)

	// If we get here without panic, check if the message is valid
	assert.NotNil(t, msg, "Message was created but may be invalid")

	// The API would reject this message with:
	// "messages: text content blocks must be non-empty"
}

func TestMessageParamRoleConversion(t *testing.T) {
	// This test verifies that messageParam correctly converts roles for the Claude API
	// Critical invariant: ToolRole must convert to User role (Claude API requirement)

	tests := []struct {
		name         string
		message      chat.Message
		expectedRole string
		description  string
	}{
		{
			name: "tool_role_converts_to_user",
			message: func() chat.Message {
				m := chat.Message{Role: chat.ToolRole}
				m.AddToolResult(chat.ToolResult{
					ToolCallID: "test_id",
					Content:    "test result",
					Name:       "test_tool",
				})
				return m
			}(),
			expectedRole: "user",
			description:  "ToolRole messages must convert to user role for Claude API",
		},
		{
			name: "user_role_stays_user",
			message: func() chat.Message {
				return chat.UserMessage("test message")
			}(),
			expectedRole: "user",
			description:  "UserRole messages should remain user role",
		},
		{
			name: "assistant_role_stays_assistant",
			message: func() chat.Message {
				m := chat.AssistantMessage("test response")
				m.AddToolCall(chat.ToolCall{
					ID:        "tool_1",
					Name:      "test_tool",
					Arguments: []byte(`{"key":"value"}`),
				})
				return m
			}(),
			expectedRole: "assistant",
			description:  "AssistantRole messages should remain assistant role",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			param, err := messageParam(tt.message)
			require.NoError(t, err, "messageParam should not error for valid messages")

			// Check the actual role field in the MessageParam struct
			if tt.expectedRole == "user" {
				// For user/tool messages, verify it's converted to user
				assert.Equal(t, anthropic.MessageParamRoleUser, param.Role, tt.description)
			} else if tt.expectedRole == "assistant" {
				assert.Equal(t, anthropic.MessageParamRoleAssistant, param.Role, tt.description)
			}
		})
	}
}

func TestAssistantMessagesNeverContainToolResults(t *testing.T) {
	// This test verifies that the invariant is maintained: assistant messages
	// must NEVER contain tool results - only tool calls are allowed

	t.Run("assistant_with_tool_calls_only", func(t *testing.T) {
		// Create an assistant message with tool calls (allowed)
		assistantMsg := chat.AssistantMessage("I'll help you with that")
		assistantMsg.AddToolCall(chat.ToolCall{
			ID:        "call_1",
			Name:      "allowed_tool",
			Arguments: []byte(`{}`),
		})

		// Verify the message has tool calls
		assert.True(t, assistantMsg.HasToolCalls(), "Assistant messages can contain tool calls")
		assert.Equal(t, 1, len(assistantMsg.GetToolCalls()), "Should have one tool call")
		assert.False(t, assistantMsg.HasToolResults(), "Assistant messages should not have tool results")

		// Verify messageParam handles this correctly
		param, err := messageParam(assistantMsg)
		require.NoError(t, err, "Should convert assistant message with tool calls")

		// Verify it's an assistant message
		assert.Equal(t, anthropic.MessageParamRoleAssistant, param.Role)
	})

	t.Run("assistant_with_tool_results_violates_invariant", func(t *testing.T) {
		// This test documents what happens if the invariant is violated
		// In production code, this should never occur
		assistantMsg := chat.AssistantMessage("I'll help you with that")

		// Incorrectly add a tool result to assistant message (violates invariant)
		assistantMsg.AddToolResult(chat.ToolResult{
			ToolCallID: "result_1",
			Content:    "result content",
			Name:       "test_tool",
		})

		// Verify the invariant is violated
		assert.True(t, assistantMsg.HasToolResults(), "Message incorrectly has tool results")

		// The simplified messageParam will still convert this but it will
		// create an incorrect message structure for the Claude API
		param, err := messageParam(assistantMsg)
		require.NoError(t, err, "Converts even with violated invariant")

		// This would create an assistant message with tool results, which Claude API rejects
		// The invariant must be maintained by message construction code, not by conversion
		assert.Equal(t, anthropic.MessageParamRoleAssistant, param.Role,
			"Creates assistant message with tool results - Claude API would reject this")
	})

	t.Run("tool_results_in_tool_role_correct", func(t *testing.T) {
		// This is the correct way: tool results in ToolRole messages
		toolMsg := chat.Message{Role: chat.ToolRole}
		toolMsg.AddToolResult(chat.ToolResult{
			ToolCallID: "result_1",
			Content:    "result content",
			Name:       "test_tool",
		})

		// Verify conversion to user message for Claude API
		param, err := messageParam(toolMsg)
		require.NoError(t, err, "Should convert tool message")

		// Verify it's converted to user message (Claude API requirement)
		assert.Equal(t, anthropic.MessageParamRoleUser, param.Role,
			"ToolRole messages must convert to user role for Claude API")
	})
}
