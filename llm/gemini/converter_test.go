package gemini

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/bpowers/go-agent/chat"
)

func TestMessageToGemini(t *testing.T) {
	tests := []struct {
		name    string
		msg     chat.Message
		want    []*genai.Content
		wantErr bool
		errMsg  string
	}{
		{
			name: "user message with text",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{Text: "Hello, Gemini!"},
				},
			},
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "Hello, Gemini!"},
					},
				},
			},
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
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "First part\nSecond part"},
					},
				},
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
			want: []*genai.Content{
				{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Hello! How can I help you today?"},
					},
				},
			},
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
			want: []*genai.Content{
				{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tool_123",
								Name: "get_weather",
								Args: map[string]any{
									"city": "Paris",
								},
							},
						},
					},
				},
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
							ID:        "tool_456",
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"city":"London"}`),
						},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Let me check the weather for you."},
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tool_456",
								Name: "get_weather",
								Args: map[string]any{
									"city": "London",
								},
							},
						},
					},
				},
			},
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
			want: []*genai.Content{
				{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tool_1",
								Name: "first_tool",
								Args: map[string]any{
									"arg": "value1",
								},
							},
						},
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tool_2",
								Name: "second_tool",
								Args: map[string]any{
									"arg": "value2",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "tool role message with tool result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Name:       "get_weather",
							Content:    `{"temperature": "22C"}`,
						},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "function",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								ID:   "tool_123",
								Name: "get_weather",
								Response: map[string]any{
									"temperature": "22C",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "tool role message with error result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Name:       "get_weather",
							Error:      "API rate limit exceeded",
						},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "function",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								ID:   "tool_123",
								Name: "get_weather",
								Response: map[string]any{
									"error": "API rate limit exceeded",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "tool role message with plain text result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Name:       "get_weather",
							Content:    "The weather is sunny",
						},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "function",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								ID:   "tool_123",
								Name: "get_weather",
								Response: map[string]any{
									"result": "The weather is sunny",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "tool role message with empty result",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_123",
							Name:       "get_weather",
							Content:    "",
						},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "function",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								ID:   "tool_123",
								Name: "get_weather",
								Response: map[string]any{
									"result": "success",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "tool role message with multiple results",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_1",
							Name:       "first_tool",
							Content:    "Result 1",
						},
					},
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tool_2",
							Name:       "second_tool",
							Content:    "Result 2",
						},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "function",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								ID:   "tool_1",
								Name: "first_tool",
								Response: map[string]any{
									"result": "Result 1",
								},
							},
						},
						{
							FunctionResponse: &genai.FunctionResponse{
								ID:   "tool_2",
								Name: "second_tool",
								Response: map[string]any{
									"result": "Result 2",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "system role converts to user",
			msg: chat.Message{
				Role: "system",
				Contents: []chat.Content{
					{Text: "You are a helpful assistant."},
				},
			},
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "You are a helpful assistant."},
					},
				},
			},
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
			name: "user message with no text returns error",
			msg: chat.Message{
				Role: chat.UserRole,
				Contents: []chat.Content{
					{Text: ""},
				},
			},
			wantErr: true,
			errMsg:  "user/system message has no text content",
		},
		{
			name: "assistant message with empty content returns nil",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{Text: ""},
				},
			},
			want: nil, // Empty assistant messages return nil
		},
		{
			name: "tool role with no results returns error",
			msg: chat.Message{
				Role: chat.ToolRole,
				Contents: []chat.Content{
					{Text: "Some text but no tool results"},
				},
			},
			wantErr: true,
			errMsg:  "tool message has no tool results",
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
			want: []*genai.Content{
				{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "First\nSecond"},
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tc1",
								Name: "tool1",
								Args: map[string]any{},
							},
						},
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tc2",
								Name: "tool2",
								Args: map[string]any{},
							},
						},
					},
				},
			},
		},
		{
			name: "tool call with invalid JSON arguments wraps as raw",
			msg: chat.Message{
				Role: chat.AssistantRole,
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:        "tool_123",
							Name:      "bad_tool",
							Arguments: json.RawMessage(`not valid json`),
						},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tool_123",
								Name: "bad_tool",
								Args: map[string]any{
									"raw": "not valid json",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "unknown role with text treats as user",
			msg: chat.Message{
				Role: "unknown",
				Contents: []chat.Content{
					{Text: "Some text"},
				},
			},
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "Some text"},
					},
				},
			},
		},
		{
			name: "unknown role with no text returns error",
			msg: chat.Message{
				Role: "unknown",
				Contents: []chat.Content{
					{Text: ""},
				},
			},
			wantErr: true,
			errMsg:  "message with unknown role has no text content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := messageToGemini(tt.msg)

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

func TestExtractText(t *testing.T) {
	tests := []struct {
		name string
		msg  chat.Message
		want string
	}{
		{
			name: "single text content",
			msg: chat.Message{
				Contents: []chat.Content{
					{Text: "Hello"},
				},
			},
			want: "Hello",
		},
		{
			name: "multiple text contents",
			msg: chat.Message{
				Contents: []chat.Content{
					{Text: "First"},
					{Text: "Second"},
				},
			},
			want: "First\nSecond",
		},
		{
			name: "text mixed with other content",
			msg: chat.Message{
				Contents: []chat.Content{
					{Text: "Text1"},
					{ToolCall: &chat.ToolCall{ID: "tc1", Name: "tool"}},
					{Text: "Text2"},
					{ToolResult: &chat.ToolResult{ToolCallID: "tc1", Content: "result"}},
				},
			},
			want: "Text1\nText2",
		},
		{
			name: "no text content",
			msg: chat.Message{
				Contents: []chat.Content{
					{ToolCall: &chat.ToolCall{ID: "tc1", Name: "tool"}},
				},
			},
			want: "",
		},
		{
			name: "empty text content",
			msg: chat.Message{
				Contents: []chat.Content{
					{Text: ""},
				},
			},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractText(tt.msg)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestExtractToolCalls(t *testing.T) {
	tests := []struct {
		name string
		msg  chat.Message
		want []chat.ToolCall
	}{
		{
			name: "single tool call",
			msg: chat.Message{
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:        "tc1",
							Name:      "tool1",
							Arguments: json.RawMessage(`{"arg":"val"}`),
						},
					},
				},
			},
			want: []chat.ToolCall{
				{
					ID:        "tc1",
					Name:      "tool1",
					Arguments: json.RawMessage(`{"arg":"val"}`),
				},
			},
		},
		{
			name: "multiple tool calls",
			msg: chat.Message{
				Contents: []chat.Content{
					{
						ToolCall: &chat.ToolCall{
							ID:   "tc1",
							Name: "tool1",
						},
					},
					{
						ToolCall: &chat.ToolCall{
							ID:   "tc2",
							Name: "tool2",
						},
					},
				},
			},
			want: []chat.ToolCall{
				{ID: "tc1", Name: "tool1"},
				{ID: "tc2", Name: "tool2"},
			},
		},
		{
			name: "no tool calls",
			msg: chat.Message{
				Contents: []chat.Content{
					{Text: "Just text"},
				},
			},
			want: nil,
		},
		{
			name: "tool calls mixed with other content",
			msg: chat.Message{
				Contents: []chat.Content{
					{Text: "Text"},
					{
						ToolCall: &chat.ToolCall{
							ID:   "tc1",
							Name: "tool1",
						},
					},
					{ToolResult: &chat.ToolResult{ToolCallID: "tc1"}},
				},
			},
			want: []chat.ToolCall{
				{ID: "tc1", Name: "tool1"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractToolCalls(tt.msg)
			if tt.want == nil {
				assert.Nil(t, got)
			} else {
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestExtractToolResults(t *testing.T) {
	tests := []struct {
		name string
		msg  chat.Message
		want []chat.ToolResult
	}{
		{
			name: "single tool result",
			msg: chat.Message{
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tc1",
							Name:       "tool1",
							Content:    "result",
						},
					},
				},
			},
			want: []chat.ToolResult{
				{
					ToolCallID: "tc1",
					Name:       "tool1",
					Content:    "result",
				},
			},
		},
		{
			name: "multiple tool results",
			msg: chat.Message{
				Contents: []chat.Content{
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tc1",
							Name:       "tool1",
							Content:    "result1",
						},
					},
					{
						ToolResult: &chat.ToolResult{
							ToolCallID: "tc2",
							Name:       "tool2",
							Error:      "error2",
						},
					},
				},
			},
			want: []chat.ToolResult{
				{ToolCallID: "tc1", Name: "tool1", Content: "result1"},
				{ToolCallID: "tc2", Name: "tool2", Error: "error2"},
			},
		},
		{
			name: "no tool results",
			msg: chat.Message{
				Contents: []chat.Content{
					{Text: "Just text"},
				},
			},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractToolResults(tt.msg)
			if tt.want == nil {
				assert.Nil(t, got)
			} else {
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestMessagesToGemini(t *testing.T) {
	tests := []struct {
		name    string
		msgs    []chat.Message
		want    []*genai.Content
		wantErr bool
		errMsg  string
	}{
		{
			name: "multiple messages conversion",
			msgs: []chat.Message{
				{
					Role: chat.UserRole,
					Contents: []chat.Content{
						{Text: "Hello"},
					},
				},
				{
					Role: chat.AssistantRole,
					Contents: []chat.Content{
						{Text: "Hi there!"},
					},
				},
				{
					Role: chat.UserRole,
					Contents: []chat.Content{
						{Text: "How are you?"},
					},
				},
			},
			want: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "Hello"}},
				},
				{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Hi there!"}},
				},
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "How are you?"}},
				},
			},
		},
		{
			name: "filters out empty assistant messages",
			msgs: []chat.Message{
				{
					Role: chat.UserRole,
					Contents: []chat.Content{
						{Text: "Hello"},
					},
				},
				{
					Role:     chat.AssistantRole,
					Contents: []chat.Content{{Text: ""}}, // Empty assistant message
				},
				{
					Role: chat.UserRole,
					Contents: []chat.Content{
						{Text: "Are you there?"},
					},
				},
			},
			want: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "Hello"}},
				},
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "Are you there?"}},
				},
			},
		},
		{
			name: "error on invalid message",
			msgs: []chat.Message{
				{
					Role: chat.UserRole,
					Contents: []chat.Content{
						{Text: "Hello"},
					},
				},
				{
					Role:     chat.UserRole,
					Contents: []chat.Content{}, // Empty contents
				},
			},
			wantErr: true,
			errMsg:  "converting message 1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := messagesToGemini(tt.msgs)

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

func TestJsonSchemaToGeminiSchema(t *testing.T) {
	tests := []struct {
		name       string
		jsonSchema map[string]interface{}
		want       *genai.Schema
		wantErr    bool
		errMsg     string
	}{
		{
			name: "simple string property",
			jsonSchema: map[string]interface{}{
				"type": "string",
			},
			want: &genai.Schema{
				Type: genai.TypeString,
			},
		},
		{
			name: "string with description",
			jsonSchema: map[string]interface{}{
				"type":        "string",
				"description": "A user's name",
			},
			want: &genai.Schema{
				Type:        genai.TypeString,
				Description: "A user's name",
			},
		},
		{
			name: "integer type",
			jsonSchema: map[string]interface{}{
				"type": "integer",
			},
			want: &genai.Schema{
				Type: genai.TypeInteger,
			},
		},
		{
			name: "number type",
			jsonSchema: map[string]interface{}{
				"type": "number",
			},
			want: &genai.Schema{
				Type: genai.TypeNumber,
			},
		},
		{
			name: "boolean type",
			jsonSchema: map[string]interface{}{
				"type": "boolean",
			},
			want: &genai.Schema{
				Type: genai.TypeBoolean,
			},
		},
		{
			name: "string with enum",
			jsonSchema: map[string]interface{}{
				"type": "string",
				"enum": []interface{}{"red", "green", "blue"},
			},
			want: &genai.Schema{
				Type: genai.TypeString,
				Enum: []string{"red", "green", "blue"},
			},
		},
		{
			name: "array with string items",
			jsonSchema: map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
			want: &genai.Schema{
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type: genai.TypeString,
				},
			},
		},
		{
			name: "array with object items",
			jsonSchema: map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"name": map[string]interface{}{
							"type": "string",
						},
						"age": map[string]interface{}{
							"type": "integer",
						},
					},
					"required": []interface{}{"name"},
				},
			},
			want: &genai.Schema{
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"name": {
							Type: genai.TypeString,
						},
						"age": {
							Type: genai.TypeInteger,
						},
					},
					Required: []string{"name"},
				},
			},
		},
		{
			name: "object with simple properties",
			jsonSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"title": map[string]interface{}{
						"type": "string",
					},
					"count": map[string]interface{}{
						"type": "integer",
					},
				},
				"required": []interface{}{"title"},
			},
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"title": {
						Type: genai.TypeString,
					},
					"count": {
						Type: genai.TypeInteger,
					},
				},
				Required: []string{"title"},
			},
		},
		{
			name: "nested objects",
			jsonSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"user": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"address": map[string]interface{}{
								"type": "object",
								"properties": map[string]interface{}{
									"street": map[string]interface{}{
										"type": "string",
									},
									"city": map[string]interface{}{
										"type": "string",
									},
								},
								"required": []interface{}{"street"},
							},
						},
					},
				},
			},
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"user": {
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"address": {
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"street": {
										Type: genai.TypeString,
									},
									"city": {
										Type: genai.TypeString,
									},
								},
								Required: []string{"street"},
							},
						},
					},
				},
			},
		},
		{
			name: "array of arrays",
			jsonSchema: map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "array",
					"items": map[string]interface{}{
						"type": "number",
					},
				},
			},
			want: &genai.Schema{
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type: genai.TypeNumber,
					},
				},
			},
		},
		{
			name: "complex nested structure matching user's causal_chains",
			jsonSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"causal_chains": map[string]interface{}{
						"type": "array",
						"items": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"initial_variable": map[string]interface{}{
									"type": "string",
								},
								"reasoning": map[string]interface{}{
									"type": "string",
								},
								"relationships": map[string]interface{}{
									"type": "array",
									"items": map[string]interface{}{
										"type": "object",
										"properties": map[string]interface{}{
											"variable": map[string]interface{}{
												"type": "string",
											},
											"polarity": map[string]interface{}{
												"type":        "string",
												"description": "\"+\", or \"-\"",
											},
											"polarity_reasoning": map[string]interface{}{
												"type": "string",
											},
										},
										"required": []interface{}{"variable", "polarity", "polarity_reasoning"},
									},
								},
							},
							"required": []interface{}{"initial_variable", "relationships", "reasoning"},
						},
					},
				},
			},
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"causal_chains": {
						Type: genai.TypeArray,
						Items: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"initial_variable": {
									Type: genai.TypeString,
								},
								"reasoning": {
									Type: genai.TypeString,
								},
								"relationships": {
									Type: genai.TypeArray,
									Items: &genai.Schema{
										Type: genai.TypeObject,
										Properties: map[string]*genai.Schema{
											"variable": {
												Type: genai.TypeString,
											},
											"polarity": {
												Type:        genai.TypeString,
												Description: "\"+\", or \"-\"",
											},
											"polarity_reasoning": {
												Type: genai.TypeString,
											},
										},
										Required: []string{"variable", "polarity", "polarity_reasoning"},
									},
								},
							},
							Required: []string{"initial_variable", "relationships", "reasoning"},
						},
					},
				},
			},
		},
		{
			name: "object with multiple required fields",
			jsonSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"field1": map[string]interface{}{"type": "string"},
					"field2": map[string]interface{}{"type": "string"},
					"field3": map[string]interface{}{"type": "string"},
				},
				"required": []interface{}{"field1", "field2", "field3"},
			},
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"field1": {Type: genai.TypeString},
					"field2": {Type: genai.TypeString},
					"field3": {Type: genai.TypeString},
				},
				Required: []string{"field1", "field2", "field3"},
			},
		},
		{
			name: "array without items field",
			jsonSchema: map[string]interface{}{
				"type": "array",
			},
			want: &genai.Schema{
				Type: genai.TypeArray,
			},
		},
		{
			name: "object without properties",
			jsonSchema: map[string]interface{}{
				"type": "object",
			},
			want: &genai.Schema{
				Type: genai.TypeObject,
			},
		},
		{
			name: "property with description and enum",
			jsonSchema: map[string]interface{}{
				"type":        "string",
				"description": "Status of the item",
				"enum":        []interface{}{"pending", "active", "completed"},
			},
			want: &genai.Schema{
				Type:        genai.TypeString,
				Description: "Status of the item",
				Enum:        []string{"pending", "active", "completed"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := jsonSchemaToGeminiSchema(tt.jsonSchema)

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
