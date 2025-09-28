package openai

import (
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/internal/common"
)

// messageToOpenAI converts a chat.Message to OpenAI message parameters.
// This function handles all message types (User, Assistant, Tool) and content types
// (text, tool calls, tool results) using the unified Contents array approach.
//
// IMPORTANT INVARIANTS for OpenAI:
// - Tool calls must be in Assistant role messages
// - Tool results must be in separate Tool role messages
// - System messages are a distinct role and should not contain tool content
// - OpenAI uses "tool" role for tool results, not "user" like Claude
func messageToOpenAI(msg chat.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(msg.Contents) == 0 {
		return nil, fmt.Errorf("message has no contents")
	}

	switch msg.Role {
	case chat.UserRole:
		// User messages can only contain text content
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("user message has no text content")
		}
		return []openai.ChatCompletionMessageParamUnion{openai.UserMessage(text)}, nil

	case chat.AssistantRole:
		// Assistant messages can contain text and/or tool calls
		assistant := openai.ChatCompletionAssistantMessageParam{}

		// Extract text content if present
		if text := extractText(msg); text != "" {
			assistant.Content.OfString = param.NewOpt(text)
		}

		// Extract tool calls if present
		toolCalls := extractToolCalls(msg)
		if len(toolCalls) > 0 {
			assistant.ToolCalls = buildOpenAIToolCallParams(toolCalls)
		}

		// Validate that we have at least some content
		if assistant.Content.OfString.Value == "" && len(assistant.ToolCalls) == 0 {
			return nil, fmt.Errorf("assistant message has no valid content")
		}

		return []openai.ChatCompletionMessageParamUnion{{OfAssistant: &assistant}}, nil

	case chat.ToolRole:
		// Tool role messages contain tool results
		// OpenAI requires separate messages for each tool result
		toolResults := extractToolResults(msg)
		if len(toolResults) == 0 {
			// Fallback: if no structured tool results but text present
			if text := extractText(msg); text != "" {
				return []openai.ChatCompletionMessageParamUnion{
					openai.ToolMessage(text, ""),
				}, nil
			}
			return nil, fmt.Errorf("tool message has no tool results")
		}

		// Convert each tool result to a separate message
		msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(toolResults))
		for _, tr := range toolResults {
			content := tr.Content
			if tr.Error != "" {
				content = common.FormatToolErrorJSON(tr.Error)
			}
			if content == "" {
				content = "{}"
			}
			msgs = append(msgs, openai.ToolMessage(content, tr.ToolCallID))
		}
		return msgs, nil

	case "system":
		// System messages can only contain text
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("system message has no text content")
		}
		return []openai.ChatCompletionMessageParamUnion{openai.SystemMessage(text)}, nil

	default:
		return nil, fmt.Errorf("unknown message role: %s", msg.Role)
	}
}

// extractText concatenates all text content from a message.
func extractText(msg chat.Message) string {
	var text string
	for _, content := range msg.Contents {
		if content.Text != "" {
			if text != "" {
				text += "\n"
			}
			text += content.Text
		}
	}
	return text
}

// extractToolCalls collects all tool calls from a message.
func extractToolCalls(msg chat.Message) []chat.ToolCall {
	var calls []chat.ToolCall
	for _, content := range msg.Contents {
		if content.ToolCall != nil {
			calls = append(calls, *content.ToolCall)
		}
	}
	return calls
}

// extractToolResults collects all tool results from a message.
func extractToolResults(msg chat.Message) []chat.ToolResult {
	var results []chat.ToolResult
	for _, content := range msg.Contents {
		if content.ToolResult != nil {
			results = append(results, *content.ToolResult)
		}
	}
	return results
}

// buildOpenAIToolCallParams converts chat.ToolCall array to OpenAI tool call params.
func buildOpenAIToolCallParams(toolCalls []chat.ToolCall) []openai.ChatCompletionMessageToolCallParam {
	params := make([]openai.ChatCompletionMessageToolCallParam, len(toolCalls))
	for i, tc := range toolCalls {
		params[i] = openai.ChatCompletionMessageToolCallParam{
			ID: tc.ID,
			Function: openai.ChatCompletionMessageToolCallFunctionParam{
				Name:      tc.Name,
				Arguments: string(tc.Arguments),
			},
		}
	}
	return params
}

// messagesToOpenAI converts a slice of chat messages to OpenAI message parameters.
// This handles the conversion of multiple messages, properly expanding tool results
// which may result in multiple OpenAI messages for a single chat message.
func messagesToOpenAI(msgs []chat.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	var result []openai.ChatCompletionMessageParamUnion

	for i, msg := range msgs {
		converted, err := messageToOpenAI(msg)
		if err != nil {
			return nil, fmt.Errorf("converting message %d: %w", i, err)
		}
		result = append(result, converted...)
	}

	return result, nil
}
