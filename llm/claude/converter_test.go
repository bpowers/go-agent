package claude

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
)

func TestMessageParam(t *testing.T) {
	tests := []struct {
		name    string
		msg     chat.Message
		want    anthropic.MessageParam
		wantErr bool
		errMsg  string
	}{
		{
			name: "user message with text",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{Text: "Hello, Claude!"},
				},
			},
			want: anthropic.NewUserMessage(
				anthropic.NewTextBlock("Hello, Claude!"),
			),
		},
		{
			name: "user message with multiple text contents",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{Text: "First part"},
					{Text: "Second part"},
				},
			},
			want: anthropic.NewUserMessage(
				anthropic.NewTextBlock("First part"),
				anthropic.NewTextBlock("Second part"),
			),
		},
		{
			name: "assistant message with text",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: "Hello! How can I help you today?"},
				},
			},
			want: anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Hello! How can I help you today?"),
			),
		},
		{
			name: "assistant message with tool call",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:        "tool_123",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"city":"Paris"}`),
						},
					},
				},
			},
			want: anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("tool_123", json.RawMessage(`{"city":"Paris"}`), "get_weather"),
			),
		},
		{
			name: "assistant message with text and tool call",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: "Let me check the weather for you."},
					{
						ToolCall: &chat.ToolCall{
							ID:        "tool_456",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"city":"London"}`),
						},
					},
				},
			},
			want: anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Let me check the weather for you."),
				anthropic.NewToolUseBlock("tool_456", json.RawMessage(`{"city":"London"}`), "get_weather"),
			),
		},
		{
			name: "assistant message with multiple tool calls",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:        "tool_1",
							Name:      "first_tool",
							Arguments: json.RawMessage(`{"arg":"value1"}`),
						},
					},
					{
						ToolCall: &chat.ToolCall{
							ID:        "tool_2",
							Name:      "second_tool",
							Arguments: json.RawMessage(`{"arg":"value2"}`),
						},
					},
				},
			},
			want: anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("tool_1", json.RawMessage(`{"arg":"value1"}`), "first_tool"),
				anthropic.NewToolUseBlock("tool_2", json.RawMessage(`{"arg":"value2"}`), "second_tool"),
			),
		},
		{
			name: "tool role message with tool result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Content:    "The weather in Paris is sunny.",
						},
					},
				},
			},
			want: anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("tool_123", "The weather in Paris is sunny.", false),
			),
		},
		{
			name: "tool role message with error result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Error:      "API rate limit exceeded",
						},
					},
				},
			},
			want: anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("tool_123", `{"error":"API rate limit exceeded"}`, true),
			),
		},
		{
			name: "tool role message with multiple results",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_1",
							Content:    "Result 1",
						},
					},
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_2",
							Content:    "Result 2",
						},
					},
				},
			},
			want: anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("tool_1", "Result 1", false),
				anthropic.NewToolResultBlock("tool_2", "Result 2", false),
			),
		},
		{
			name: "system role converts to user",
			msg: chat.Message{
				Role: "system",
				Contents: []chat.Content{
					{Text: "You are a helpful assistant."},
				},
			},
			want: anthropic.NewUserMessage(
				anthropic.NewTextBlock("You are a helpful assistant."),
			),
		},
		{
			name: "empty message returns error",
			msg: chat.Message{
				Role:     chat.UserRole,
				Contents: []chat.Content{},
			},
			wantErr: true,
			errMsg:  "message has no contents",
		},
		{
			name: "message with only empty text",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{Text: ""},
				},
			},
			wantErr: true,
			errMsg:  "message has no valid content blocks",
		},
		{
			name: "assistant message with tool result",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: "Here's the weather:"},
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Content:    "Sunny",
						},
					},
				},
			},
			want: anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Here's the weather:"),
				anthropic.NewToolResultBlock("tool_123", "Sunny", false),
			),
		},
		{
			name: "mixed content preserves order",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: "First"},
					{
						ToolCall: &chat.ToolCall{
							ID:        "tc1",
							Name:      "tool1",
							Arguments: json.RawMessage(`{}`),
						},
					},
					{Text: "Second"},
					{
						ToolCall: &chat.ToolCall{
							ID:        "tc2",
							Name:      "tool2",
							Arguments: json.RawMessage(`{}`),
						},
					},
				},
			},
			want: anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("First"),
				anthropic.NewToolUseBlock("tc1", json.RawMessage(`{}`), "tool1"),
				anthropic.NewTextBlock("Second"),
				anthropic.NewToolUseBlock("tc2", json.RawMessage(`{}`), "tool2"),
			),
		},
		{
			name: "tool result with empty content uses empty JSON",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Content:    "",
						},
					},
				},
			},
			want: anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("tool_123", "{}", false),
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := messageParam(tt.msg)

			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
