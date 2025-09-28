package gemini

import (
	"encoding/json"
	"fmt"

	"google.golang.org/genai"

	"github.com/bpowers/go-agent/chat"
)

// messageToGemini converts a chat.Message to Gemini Content format.
// This function handles all message types (User, Assistant, Tool) and content types
// (text, tool calls, tool results) using the unified Contents array approach.
//
// IMPORTANT INVARIANTS for Gemini:
// - Tool calls are FunctionCall parts within a Content
// - Tool results are FunctionResponse parts with "function" role
// - Assistant role maps to "model", User role maps to "user", Tool role maps to "function"
// - Multiple content types can be mixed within a single message's Parts array
// - Empty messages should return nil rather than empty Content objects
func messageToGemini(msg chat.Message) ([]*genai.Content, error) {
	if len(msg.Contents) == 0 {
		return nil, fmt.Errorf("message has no contents")
	}

	switch msg.Role {
	case chat.UserRole, "system":
		// User and system messages
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("user/system message has no text content")
		}
		return []*genai.Content{{
			Role:  "user",
			Parts: []*genai.Part{{Text: text}},
		}}, nil

	case chat.AssistantRole:
		// Assistant messages can contain text and/or tool calls
		var parts []*genai.Part

		// Add text content if present
		if text := extractText(msg); text != "" {
			parts = append(parts, &genai.Part{Text: text})
		}

		// Add tool calls if present
		toolCalls := extractToolCalls(msg)
		for _, tc := range toolCalls {
			// Convert arguments from JSON to map
			var args map[string]any
			if len(tc.Arguments) > 0 {
				if err := json.Unmarshal(tc.Arguments, &args); err != nil {
					// If unmarshal fails, wrap as raw argument
					args = map[string]any{"raw": string(tc.Arguments)}
				}
			}

			// Ensure ID is set
			id := tc.ID
			if id == "" {
				id = generateFunctionCallID()
			}

			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   id,
					Name: tc.Name,
					Args: args,
				},
			})
		}

		// Skip empty assistant messages
		if len(parts) == 0 {
			return nil, nil
		}

		return []*genai.Content{{
			Role:  "model",
			Parts: parts,
		}}, nil

	case chat.ToolRole:
		// Tool role messages contain tool results
		// Gemini uses "function" role for these
		toolResults := extractToolResults(msg)
		if len(toolResults) == 0 {
			return nil, fmt.Errorf("tool message has no tool results")
		}

		// Convert tool results to function response parts
		parts := make([]*genai.Part, 0, len(toolResults))
		for _, tr := range toolResults {
			response := make(map[string]any)

			if tr.Error != "" {
				response["error"] = tr.Error
			} else if tr.Content != "" {
				// Try to unmarshal as JSON first
				if err := json.Unmarshal([]byte(tr.Content), &response); err != nil {
					// If not valid JSON, wrap as result
					response["result"] = tr.Content
				}
			} else {
				// Empty result - provide a non-empty response for Gemini
				response["result"] = "success"
			}

			parts = append(parts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:       tr.ToolCallID,
					Name:     tr.Name,
					Response: response,
				},
			})
		}

		return []*genai.Content{{
			Role:  "function",
			Parts: parts,
		}}, nil

	default:
		// Unknown role, treat as user message
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("message with unknown role has no text content")
		}
		return []*genai.Content{{
			Role:  "user",
			Parts: []*genai.Part{{Text: text}},
		}}, nil
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

// messagesToGemini converts a slice of chat messages to Gemini Content format.
// This handles the conversion of multiple messages, filtering out any nil results.
func messagesToGemini(msgs []chat.Message) ([]*genai.Content, error) {
	var result []*genai.Content

	for i, msg := range msgs {
		converted, err := messageToGemini(msg)
		if err != nil {
			return nil, fmt.Errorf("converting message %d: %w", i, err)
		}
		// Filter out nil results (e.g., empty assistant messages)
		for _, content := range converted {
			if content != nil && len(content.Parts) > 0 {
				result = append(result, content)
			}
		}
	}

	return result, nil
}
