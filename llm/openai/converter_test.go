package openai

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
)

func TestMessageToOpenAI(t *testing.T) {
	tests := []struct {
		name      string
		msg       chat.Message
		wantCount int // number of messages expected
		wantErr   bool
		errMsg    string
		validate  func(t *testing.T, got []openai.ChatCompletionMessageParamUnion)
	}{
		{
			name: "user message with text",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{Text: "Hello, GPT!"},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfUser)
				assert.Equal(t, "Hello, GPT!", got[0].OfUser.Content.OfString.Value)
			},
		},
		{
			name: "user message with multiple text contents concatenated",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{Text: "First part"},
					{Text: "Second part"},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfUser)
				assert.Equal(t, "First part\nSecond part", got[0].OfUser.Content.OfString.Value)
			},
		},
		{
			name: "assistant message with text",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: "Hello! How can I help you today?"},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfAssistant)
				assert.Equal(t, "Hello! How can I help you today?", got[0].OfAssistant.Content.OfString.Value)
			},
		},
		{
			name: "assistant message with tool call",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:        "call_123",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"city":"Paris"}`),
						},
					},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfAssistant)
				assert.Empty(t, got[0].OfAssistant.Content.OfString.Value)
				require.Len(t, got[0].OfAssistant.ToolCalls, 1)
				assert.Equal(t, "call_123", got[0].OfAssistant.ToolCalls[0].ID)
				assert.Equal(t, "get_weather", got[0].OfAssistant.ToolCalls[0].Function.Name)
				assert.Equal(t, `{"city":"Paris"}`, got[0].OfAssistant.ToolCalls[0].Function.Arguments)
			},
		},
		{
			name: "assistant message with text and tool call",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: "Let me check the weather for you."},
					{
						ToolCall: &chat.ToolCall{
							ID:        "call_456",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"city":"London"}`),
						},
					},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfAssistant)
				assert.Equal(t, "Let me check the weather for you.", got[0].OfAssistant.Content.OfString.Value)
				require.Len(t, got[0].OfAssistant.ToolCalls, 1)
				assert.Equal(t, "call_456", got[0].OfAssistant.ToolCalls[0].ID)
				assert.Equal(t, "get_weather", got[0].OfAssistant.ToolCalls[0].Function.Name)
				assert.Equal(t, `{"city":"London"}`, got[0].OfAssistant.ToolCalls[0].Function.Arguments)
			},
		},
		{
			name: "assistant message with multiple tool calls",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:        "call_789",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"city":"Tokyo"}`),
						},
					},
					{
						ToolCall: &chat.ToolCall{
							ID:        "call_abc",
							Name:      "get_time",
							Arguments: json.RawMessage(`{"timezone":"JST"}`),
						},
					},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfAssistant)
				assert.Empty(t, got[0].OfAssistant.Content.OfString.Value)
				require.Len(t, got[0].OfAssistant.ToolCalls, 2)
				assert.Equal(t, "call_789", got[0].OfAssistant.ToolCalls[0].ID)
				assert.Equal(t, "get_weather", got[0].OfAssistant.ToolCalls[0].Function.Name)
				assert.Equal(t, "call_abc", got[0].OfAssistant.ToolCalls[1].ID)
				assert.Equal(t, "get_time", got[0].OfAssistant.ToolCalls[1].Function.Name)
			},
		},
		{
			name: "tool role message with single result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "call_123",
							Name:       "get_weather",
							Content:    `{"temperature": 20, "condition": "sunny"}`,
						},
					},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfTool)
				assert.Equal(t, "call_123", got[0].OfTool.ToolCallID)
				assert.Equal(t, `{"temperature": 20, "condition": "sunny"}`, got[0].OfTool.Content.OfString.Value)
			},
		},
		{
			name: "tool role message with multiple results creates multiple messages",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "call_123",
							Name:       "get_weather",
							Content:    `{"temperature": 20}`,
						},
					},
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "call_456",
							Name:       "get_time",
							Content:    `{"time": "12:00"}`,
						},
					},
				},
			},
			wantCount: 2, // OpenAI requires separate messages for each tool result
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfTool)
				assert.Equal(t, "call_123", got[0].OfTool.ToolCallID)
				assert.Equal(t, `{"temperature": 20}`, got[0].OfTool.Content.OfString.Value)

				require.NotNil(t, got[1].OfTool)
				assert.Equal(t, "call_456", got[1].OfTool.ToolCallID)
				assert.Equal(t, `{"time": "12:00"}`, got[1].OfTool.Content.OfString.Value)
			},
		},
		{
			name: "tool role message with error result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "call_123",
							Name:       "get_weather",
							Error:      "API key invalid",
						},
					},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfTool)
				assert.Equal(t, "call_123", got[0].OfTool.ToolCallID)
				// Error should be formatted as JSON
				assert.Contains(t, got[0].OfTool.Content.OfString.Value, "API key invalid")
				assert.Contains(t, got[0].OfTool.Content.OfString.Value, "error")
			},
		},
		{
			name: "system message with text",
			msg: chat.Message{
				Role: "system",
				Contents: []chat.Content{
					{Text: "You are a helpful assistant."},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfSystem)
				assert.Equal(t, "You are a helpful assistant.", got[0].OfSystem.Content.OfString.Value)
			},
		},
		{
			name: "mixed content preserves order",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: "First text"},
					{
						ToolCall: &chat.ToolCall{
							ID:        "call_1",
							Name:      "tool1",
							Arguments: json.RawMessage(`{}`),
						},
					},
					{Text: "Second text"},
					{
						ToolCall: &chat.ToolCall{
							ID:        "call_2",
							Name:      "tool2",
							Arguments: json.RawMessage(`{}`),
						},
					},
				},
			},
			wantCount: 1,
			validate: func(t *testing.T, got []openai.ChatCompletionMessageParamUnion) {
				require.NotNil(t, got[0].OfAssistant)
				// Text content is concatenated with newlines
				assert.Equal(t, "First text\nSecond text", got[0].OfAssistant.Content.OfString.Value)
				// Tool calls are preserved in order
				require.Len(t, got[0].OfAssistant.ToolCalls, 2)
				assert.Equal(t, "call_1", got[0].OfAssistant.ToolCalls[0].ID)
				assert.Equal(t, "tool1", got[0].OfAssistant.ToolCalls[0].Function.Name)
				assert.Equal(t, "call_2", got[0].OfAssistant.ToolCalls[1].ID)
				assert.Equal(t, "tool2", got[0].OfAssistant.ToolCalls[1].Function.Name)
			},
		},
		{
			name:    "empty message returns error",
			msg:     chat.Message{Role: chat.UserRole, Contents: []chat.Content{}},
			wantErr: true,
			errMsg:  "message has no contents",
		},
		{
			name: "user message without text returns error",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:   "call_123",
							Name: "tool",
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "user message has no text content",
		},
		{
			name: "assistant message with empty contents returns error",
			msg: chat.Message{
				Role:     chat.AssistantRole,
				Contents: []chat.Content{{}, {}}, // empty content structs
			},
			wantErr: true,
			errMsg:  "assistant message has no valid content",
		},
		{
			name: "tool message without results returns error",
			msg: chat.Message{
				Role:     chat.ToolRole,
				Contents: []chat.Content{},
			},
			wantErr: true,
			errMsg:  "message has no contents",
		},
		{
			name:    "unknown role returns error",
			msg:     chat.Message{Role: "unknown", Contents: []chat.Content{{Text: "test"}}},
			wantErr: true,
			errMsg:  "unknown message role: unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := messageToOpenAI(tt.msg)

			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
				return
			}

			require.NoError(t, err)
			assert.Len(t, got, tt.wantCount)
			if tt.validate != nil {
				tt.validate(t, got)
			}
		})
	}
}

func TestMessagesToOpenAI(t *testing.T) {
	msgs := []chat.Message{
		{
			Role: chat.UserRole,
			Contents: []chat.Content{
				{Text: "Hello"},
			},
		},
		{
			Role: chat.AssistantRole,
			Contents: []chat.Content{
				{Text: "Hi! How can I help you?"},
				{
					ToolCall: &chat.ToolCall{
						ID:        "call_123",
						Name:      "get_info",
						Arguments: json.RawMessage(`{}`),
					},
				},
			},
		},
		{
			Role: chat.ToolRole,
			Contents: []chat.Content{
				{
					ToolResult: &chat.ToolResult{
						ToolCallID: "call_123",
						Name:       "get_info",
						Content:    `{"info": "data"}`,
					},
				},
			},
		},
		{
			Role: chat.AssistantRole,
			Contents: []chat.Content{
				{Text: "Based on the information..."},
			},
		},
	}

	got, err := messagesToOpenAI(msgs)
	require.NoError(t, err)

	// Should have 4 OpenAI messages (one for each chat message)
	assert.Len(t, got, 4)

	// Verify first message (User)
	require.NotNil(t, got[0].OfUser)
	assert.Equal(t, "Hello", got[0].OfUser.Content.OfString.Value)

	// Verify second message (Assistant with tool call)
	require.NotNil(t, got[1].OfAssistant)
	assert.Equal(t, "Hi! How can I help you?", got[1].OfAssistant.Content.OfString.Value)
	require.Len(t, got[1].OfAssistant.ToolCalls, 1)
	assert.Equal(t, "call_123", got[1].OfAssistant.ToolCalls[0].ID)

	// Verify third message (Tool result)
	require.NotNil(t, got[2].OfTool)
	assert.Equal(t, "call_123", got[2].OfTool.ToolCallID)
	assert.Equal(t, `{"info": "data"}`, got[2].OfTool.Content.OfString.Value)

	// Verify fourth message (Assistant)
	require.NotNil(t, got[3].OfAssistant)
	assert.Equal(t, "Based on the information...", got[3].OfAssistant.Content.OfString.Value)
}

func TestExtractText(t *testing.T) {
	msg := chat.Message{
		Contents: []chat.Content{
			{Text: "First"},
			{ToolCall: &chat.ToolCall{ID: "123"}},
			{Text: "Second"},
			{ToolResult: &chat.ToolResult{ToolCallID: "123"}},
			{Text: "Third"},
		},
	}

	text := extractText(msg)
	assert.Equal(t, "First\nSecond\nThird", text)
}

func TestExtractToolCalls(t *testing.T) {
	tc1 := chat.ToolCall{ID: "call_1", Name: "tool1"}
	tc2 := chat.ToolCall{ID: "call_2", Name: "tool2"}

	msg := chat.Message{
		Contents: []chat.Content{
			{Text: "Some text"},
			{ToolCall: &tc1},
			{Text: "More text"},
			{ToolCall: &tc2},
		},
	}

	calls := extractToolCalls(msg)
	require.Len(t, calls, 2)
	assert.Equal(t, tc1, calls[0])
	assert.Equal(t, tc2, calls[1])
}

func TestExtractToolResults(t *testing.T) {
	tr1 := chat.ToolResult{ToolCallID: "call_1", Name: "tool1", Content: "result1"}
	tr2 := chat.ToolResult{ToolCallID: "call_2", Name: "tool2", Content: "result2"}

	msg := chat.Message{
		Contents: []chat.Content{
			{Text: "Some text"},
			{ToolResult: &tr1},
			{Text: "More text"},
			{ToolResult: &tr2},
		},
	}

	results := extractToolResults(msg)
	require.Len(t, results, 2)
	assert.Equal(t, tr1, results[0])
	assert.Equal(t, tr2, results[1])
}

func TestBuildOpenAIToolCallParams(t *testing.T) {
	toolCalls := []chat.ToolCall{
		{
			ID:        "call_123",
			Name:      "get_weather",
			Arguments: json.RawMessage(`{"city":"Paris"}`),
		},
		{
			ID:        "call_456",
			Name:      "get_time",
			Arguments: json.RawMessage(`{"timezone":"UTC"}`),
		},
	}

	params := buildOpenAIToolCallParams(toolCalls)

	require.Len(t, params, 2)
	assert.Equal(t, "call_123", params[0].ID)
	assert.Equal(t, "get_weather", params[0].Function.Name)
	assert.Equal(t, `{"city":"Paris"}`, params[0].Function.Arguments)
	assert.Equal(t, "call_456", params[1].ID)
	assert.Equal(t, "get_time", params[1].Function.Name)
	assert.Equal(t, `{"timezone":"UTC"}`, params[1].Function.Arguments)
}

// TestOpenAIRoleMapping verifies that OpenAI uses "tool" role for tool results
// unlike Claude which uses "user" role
func TestOpenAIRoleMapping(t *testing.T) {
	msg := chat.Message{
		Role: chat.ToolRole,
		Contents: []chat.Content{
			{
				ToolResult: &chat.ToolResult{
					ToolCallID: "call_123",
					Name:       "test_tool",
					Content:    "result",
				},
			},
		},
	}

	got, err := messageToOpenAI(msg)
	require.NoError(t, err)
	require.Len(t, got, 1)

	// Verify it's using the Tool role (not User like Claude)
	require.NotNil(t, got[0].OfTool, "OpenAI should use 'tool' role for tool results")
	assert.Equal(t, "call_123", got[0].OfTool.ToolCallID)
	assert.Equal(t, "result", got[0].OfTool.Content.OfString.Value)
}
