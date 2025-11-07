package claude

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/internal/logging"
	"github.com/bpowers/go-agent/llm/internal/common"
)

// logger is the package-level structured logger with provider context.
// Log levels used in this package:
//   - Info: Client creation, API selection, model warnings
//   - Debug: Stream events, tool calls, token updates, raw API data
//   - Warn: Missing token usage, unknown models, fallback behavior
//   - Error: Should never occur (indicates bugs)
var logger = logging.Logger().With("provider", "claude")

const (
	AnthropicURL = "https://api.anthropic.com/v1"
)

type client struct {
	anthropicClient anthropic.Client
	modelName       string
	baseURL         string            // Store base URL for testing
	headers         map[string]string // Custom HTTP headers
	logger          *slog.Logger
}

var _ chat.Client = &client{}

type Option func(*client)

func WithModel(modelName string) Option {
	return func(c *client) {
		c.modelName = strings.TrimSpace(modelName)
	}
}

func WithHeaders(headers map[string]string) Option {
	return func(c *client) {
		c.headers = headers
	}
}

// NewClient returns a chat client that can begin chat sessions with Claude's Messages API.
func NewClient(apiBase string, apiKey string, opts ...Option) (chat.Client, error) {
	c := &client{
		baseURL: apiBase, // Store for testing
		logger:  logger,
	}

	// Use default if empty
	if c.baseURL == "" {
		c.baseURL = AnthropicURL
	}

	for _, opt := range opts {
		opt(c)
	}

	if c.modelName == "" {
		return nil, fmt.Errorf("WithModel is a required option")
	}

	if apiKey == "" {
		return nil, fmt.Errorf("API key is required for Claude API")
	}

	// Build Anthropic client options
	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if apiBase != "" && apiBase != AnthropicURL {
		clientOpts = append(clientOpts, option.WithBaseURL(apiBase))
	}

	// Add custom headers if provided
	for key, value := range c.headers {
		clientOpts = append(clientOpts, option.WithHeader(key, value))
	}

	c.anthropicClient = anthropic.NewClient(clientOpts...)

	return c, nil
}

// BaseURL returns the base URL for testing purposes.
// This is exported for integration testing only.
func (c *client) BaseURL() string {
	return c.baseURL
}

// Headers returns the custom headers for testing purposes.
// This is exported for integration testing only.
func (c *client) Headers() map[string]string {
	return c.headers
}

// NewChat returns a chat instance.
func (c client) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	// Determine max tokens based on model
	maxTokens := getModelMaxTokens(c.modelName)

	return &chatClient{
		client:    c,
		state:     common.NewState(systemPrompt, initialMsgs),
		tools:     common.NewTools(),
		maxTokens: maxTokens,
	}
}

var modelMaxOutputTokens = map[string]int64{
	"claude-opus-4-1":   32000,
	"claude-opus-4":     32000,
	"claude-sonnet-4-5": 64000,
	"claude-sonnet-4":   64000,
	"claude-3-7-sonnet": 64000,
	"claude-3-5-haiku":  8192,
	"claude-3-haiku":    4096,
}

func getMaxOutputTokens(modelName string) int64 {
	t, ok := modelMaxOutputTokens[modelName]
	if !ok {
		logger.Warn("model not found in model library, using default", "model", modelName, "default", 4096)
		t = 4096
	}
	return t
}

// set represents a generic set of comparable items
type set[T comparable] struct {
	items map[T]struct{}
}

// newSet creates a new set with the given items
func newSet[T comparable](items ...T) *set[T] {
	s := &set[T]{
		items: make(map[T]struct{}),
	}
	for _, item := range items {
		s.items[item] = struct{}{}
	}
	return s
}

// contains checks if the set contains the exact item
func (s *set[T]) contains(item T) bool {
	_, ok := s.items[item]
	return ok
}

// containsWithPredicate checks if any item in the set satisfies the predicate
func (s *set[T]) containsWithPredicate(predicate func(T) bool) bool {
	for item := range s.items {
		if predicate(item) {
			return true
		}
	}
	return false
}

// modelsWithThinking defines which Claude models support thinking/reasoning capabilities
var modelsWithThinking = newSet(
	"claude-opus-4-1",
	"claude-opus-4",
	"claude-sonnet-4-5",
	"claude-sonnet-4",
	"claude-3-7-sonnet",
	"claude-3-5-sonnet", // Legacy naming, keep for compatibility
	// Note: haiku models do not support thinking
)

// supportsThinking checks if a model supports thinking/reasoning capabilities
func supportsThinking(modelName string) bool {
	// Check exact match first
	if modelsWithThinking.contains(modelName) {
		return true
	}

	// Check for partial matches for versioned models (e.g., "claude-sonnet-4-5-20250929")
	return modelsWithThinking.containsWithPredicate(func(model string) bool {
		return strings.Contains(modelName, model)
	})
}

var modelLimits = []chat.ModelTokenLimits{
	{Model: "claude-opus-4-1", TokenLimits: chat.TokenLimits{Context: 200000, Output: 32000}},
	{Model: "claude-opus-4", TokenLimits: chat.TokenLimits{Context: 200000, Output: 32000}},
	{Model: "claude-sonnet-4-5", TokenLimits: chat.TokenLimits{Context: 200000, Output: 64000}},
	{Model: "claude-sonnet-4", TokenLimits: chat.TokenLimits{Context: 200000, Output: 64000}},
	{Model: "claude-3-7-sonnet", TokenLimits: chat.TokenLimits{Context: 200000, Output: 64000}},
	{Model: "claude-3-5-haiku", TokenLimits: chat.TokenLimits{Context: 200000, Output: 8192}},
	{Model: "claude-3-haiku", TokenLimits: chat.TokenLimits{Context: 200000, Output: 4096}},
}

// getModelMaxTokens returns the maximum token limit for known models
func getModelMaxTokens(model string) int {
	modelLower := strings.ToLower(model)

	for _, m := range modelLimits {
		if strings.HasPrefix(modelLower, m.Model) {
			return m.TokenLimits.Output
		}
	}

	panic(fmt.Errorf("unknown model %q", model))
}

// getSystemReminderText retrieves and executes system reminder function if present
func getSystemReminderText(ctx context.Context) string {
	if reminderFunc := chat.GetSystemReminder(ctx); reminderFunc != nil {
		return reminderFunc()
	}
	return ""
}

type chatClient struct {
	client
	state     *common.State
	tools     *common.Tools
	maxTokens int
}

func (c *chatClient) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	// Apply options to get callback if provided
	reqMsg := msg
	reqOpts := chat.ApplyOptions(opts...)
	callback := reqOpts.StreamingCb

	// Build message list for Claude
	var msgs []anthropic.MessageParam

	// Snapshot history with minimal lock
	systemPrompt, history := c.state.Snapshot()

	// Add history using the proper conversion function
	for _, m := range history {
		// Skip system messages as Claude handles them separately
		if m.Role == "system" {
			continue
		}
		param, err := messageParam(m)
		if err != nil {
			return chat.Message{}, fmt.Errorf("converting history message to param: %w", err)
		}
		msgs = append(msgs, param)
	}

	// Add current message using the proper conversion function
	currentParam, err := messageParam(msg)
	if err != nil {
		return chat.Message{}, fmt.Errorf("converting current message to param: %w", err)
	}
	msgs = append(msgs, currentParam)

	// Build request parameters
	params := anthropic.MessageNewParams{
		Messages:  msgs,
		Model:     anthropic.Model(c.modelName),
		MaxTokens: getMaxOutputTokens(c.modelName), // Claude requires this
	}

	// Add tools if registered
	allTools := c.tools.GetAll()
	if len(allTools) > 0 {
		tools := make([]anthropic.ToolUnionParam, 0, len(allTools))
		for _, tool := range allTools {
			toolParam, err := c.mcpToClaudeTool(tool)
			if err != nil {
				return chat.Message{}, fmt.Errorf("failed to convert tool: %w", err)
			}
			tools = append(tools, toolParam)
		}
		params.Tools = tools
	}

	// Add system prompt if present
	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{
				Text: systemPrompt,
				Type: "text",
			},
		}
	}

	if reqOpts.Temperature != nil {
		params.Temperature = anthropic.Float(*reqOpts.Temperature)
	}

	if reqOpts.MaxTokens > 0 {
		params.MaxTokens = int64(reqOpts.MaxTokens)
	}

	// Handle response format if provided
	// Claude doesn't have a direct equivalent to OpenAI's response_format
	// but we can append instructions to the system prompt
	if reqOpts.ResponseFormat != nil && reqOpts.ResponseFormat.Schema != nil {
		systemText := systemPrompt
		if systemText != "" {
			systemText += "\n\n"
		}
		systemText += fmt.Sprintf("You must respond with valid JSON that conforms to the schema named: %s", reqOpts.ResponseFormat.Name)
		params.System = []anthropic.TextBlockParam{
			{
				Text: systemText,
				Type: "text",
			},
		}
	}

	// Streaming implementation
	stream := c.anthropicClient.Messages.NewStreaming(ctx, params)

	var respContent strings.Builder
	var inThinking bool
	var thinkingContent strings.Builder
	var thinkingSignature strings.Builder
	var toolCalls []anthropic.ToolUseBlock
	var currentToolCall *anthropic.ToolUseBlock
	var toolCallArgs strings.Builder

	for stream.Next() {
		event := stream.Current()
		c.logger.Debug("stream event", "type", event.Type)
		// Handle different event types
		switch event.Type {
		case "message_start":
			// Check if this is a model that supports thinking
			if supportsThinking(c.modelName) && callback != nil {
				// Emit initial thinking event for models that support it
				thinkingEvent := chat.StreamEvent{
					Type:           chat.StreamEventTypeThinking,
					ThinkingStatus: &chat.ThinkingStatus{},
				}
				if err := callback(thinkingEvent); err != nil {
					return chat.Message{}, err
				}
				inThinking = true
			}
		case "content_block_start":
			// Check if this is the start of thinking content
			if event.ContentBlock.Type == "thinking" {
				inThinking = true
				if callback != nil {
					thinkingEvent := chat.StreamEvent{
						Type:           chat.StreamEventTypeThinking,
						ThinkingStatus: &chat.ThinkingStatus{},
					}
					if err := callback(thinkingEvent); err != nil {
						return chat.Message{}, err
					}
				}
			} else if event.ContentBlock.Type == "tool_use" {
				// Start of a tool use block
				currentToolCall = &anthropic.ToolUseBlock{
					ID:   event.ContentBlock.ID,
					Name: event.ContentBlock.Name,
				}
				toolCallArgs.Reset()
				c.logger.Debug("tool use start", "id", event.ContentBlock.ID, "name", event.ContentBlock.Name, "input", event.ContentBlock.Input)

				// Don't emit tool call event yet - wait for arguments to be accumulated
				if event.ContentBlock.Input != nil {
					// Input is sometimes provided in the start event
					inputBytes, err := json.Marshal(event.ContentBlock.Input)
					if err == nil {
						currentToolCall.Input = json.RawMessage(inputBytes)
						c.logger.Debug("set tool input from start event", "input", string(inputBytes))
					}
				}
			} else if event.ContentBlock.Type == "redacted_thinking" {
				// Redacted thinking block (safety-flagged)
				c.logger.Debug("redacted thinking block detected", "data", event.ContentBlock.Data)
				if callback != nil {
					redactedEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeRedactedThinking,
						ThinkingStatus: &chat.ThinkingStatus{
							RedactedData: event.ContentBlock.Data,
						},
					}
					if err := callback(redactedEvent); err != nil {
						return chat.Message{}, err
					}
				}
			} else if event.ContentBlock.Type == "server_tool_use" {
				// Server-side tool invocation (e.g., web search)
				c.logger.Debug("server tool use", "id", event.ContentBlock.ID, "name", event.ContentBlock.Name, "input", event.ContentBlock.Input)
				if callback != nil {
					serverToolEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeServerToolUse,
						ToolCalls: []chat.ToolCall{
							{
								ID:        event.ContentBlock.ID,
								Name:      event.ContentBlock.Name,
								Arguments: nil, // Server tools may not have arguments
							},
						},
					}
					if err := callback(serverToolEvent); err != nil {
						return chat.Message{}, err
					}
				}
			} else if event.ContentBlock.Type == "web_search_tool_result" {
				// Web search results from server-side search
				c.logger.Debug("web search result", "tool_use_id", event.ContentBlock.ToolUseID, "content", event.ContentBlock.Content)
				if callback != nil {
					webSearchEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeWebSearchResult,
						// TODO: Parse and forward the actual search results
						Content: "Web search results received",
					}
					if err := callback(webSearchEvent); err != nil {
						return chat.Message{}, err
					}
				}
			} else if inThinking && event.ContentBlock.Type == "text" {
				// End of thinking, start of actual response
				inThinking = false
				if callback != nil && thinkingContent.Len() > 0 {
					// Send thinking summary event
					thinkingSummaryEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeThinkingSummary,
						ThinkingStatus: &chat.ThinkingStatus{
							Summary:   thinkingContent.String(),
							Signature: thinkingSignature.String(),
						},
					}
					if err := callback(thinkingSummaryEvent); err != nil {
						return chat.Message{}, err
					}
				}
			}
		case "content_block_delta":
			// Handle different delta types
			switch event.Delta.Type {
			case "text_delta":
				content := event.Delta.Text
				if inThinking {
					// Accumulate thinking content
					thinkingContent.WriteString(content)
					if callback != nil {
						// Stream thinking updates
						thinkingEvent := chat.StreamEvent{
							Type:           chat.StreamEventTypeThinking,
							Content:        content,
							ThinkingStatus: &chat.ThinkingStatus{},
						}
						if err := callback(thinkingEvent); err != nil {
							return chat.Message{}, err
						}
					}
				} else {
					// Regular content
					respContent.WriteString(content)
					if callback != nil {
						event := chat.StreamEvent{
							Type:    chat.StreamEventTypeContent,
							Content: content,
						}
						if err := callback(event); err != nil {
							return chat.Message{}, err
						}
					}
				}
			case "thinking_delta":
				// Direct thinking delta (not via text when inThinking)
				thinking := event.Delta.Thinking
				thinkingContent.WriteString(thinking)
				if callback != nil {
					thinkingEvent := chat.StreamEvent{
						Type:           chat.StreamEventTypeThinking,
						Content:        thinking,
						ThinkingStatus: &chat.ThinkingStatus{},
					}
					if err := callback(thinkingEvent); err != nil {
						return chat.Message{}, err
					}
				}
			case "signature_delta":
				// Thinking block signature
				signature := event.Delta.Signature
				thinkingSignature.WriteString(signature)
				c.logger.Debug("signature_delta", "signature", signature)
			case "citations_delta":
				// Citation updates
				c.logger.Debug("citations_delta", "citation", event.Delta.Citation)
				// TODO: Handle citation updates
			case "input_json_delta":
				// Tool use input delta
				if currentToolCall != nil {
					if partialJSON := event.Delta.PartialJSON; partialJSON != "" {
						c.logger.Debug("input_json_delta", "partial_json", partialJSON)
						toolCallArgs.WriteString(partialJSON)
					}
				}
			default:
				// Also handle the case where Delta.Text is set but Type is empty (backwards compatibility)
				if event.Delta.Text != "" && event.Delta.Type == "" {
					content := event.Delta.Text
					if inThinking {
						thinkingContent.WriteString(content)
						if callback != nil {
							thinkingEvent := chat.StreamEvent{
								Type:           chat.StreamEventTypeThinking,
								Content:        content,
								ThinkingStatus: &chat.ThinkingStatus{},
							}
							if err := callback(thinkingEvent); err != nil {
								return chat.Message{}, err
							}
						}
					} else {
						respContent.WriteString(content)
						if callback != nil {
							event := chat.StreamEvent{
								Type:    chat.StreamEventTypeContent,
								Content: content,
							}
							if err := callback(event); err != nil {
								return chat.Message{}, err
							}
						}
					}
				} else if event.Delta.Type != "" {
					c.logger.Debug("unhandled delta type", "type", event.Delta.Type, "delta", event.Delta)
				}
			}
		case "content_block_stop":
			if inThinking {
				inThinking = false
				if callback != nil && thinkingContent.Len() > 0 {
					// Send thinking summary event
					thinkingSummaryEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeThinkingSummary,
						ThinkingStatus: &chat.ThinkingStatus{
							Summary:   thinkingContent.String(),
							Signature: thinkingSignature.String(),
						},
					}
					if err := callback(thinkingSummaryEvent); err != nil {
						return chat.Message{}, err
					}
				}
			}
			// Finalize current tool call if we have one
			if currentToolCall != nil {
				// Prefer accumulated deltas over start event input
				if toolCallArgs.Len() > 0 {
					currentToolCall.Input = json.RawMessage(toolCallArgs.String())
					c.logger.Debug("set tool input from deltas", "input", toolCallArgs.String())
				}

				// Now emit the tool call event with complete arguments
				if callback != nil {
					toolCallEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeToolCall,
						ToolCalls: []chat.ToolCall{
							{
								ID:        currentToolCall.ID,
								Name:      currentToolCall.Name,
								Arguments: currentToolCall.Input,
							},
						},
					}
					if err := callback(toolCallEvent); err != nil {
						return chat.Message{}, err
					}
				}

				c.logger.Debug("finalizing tool call", "id", currentToolCall.ID, "name", currentToolCall.Name, "input", string(currentToolCall.Input))
				toolCalls = append(toolCalls, *currentToolCall)
				currentToolCall = nil
				toolCallArgs.Reset()
			}
		case "message_delta":
			// Check for usage information in message delta
			if event.Usage.InputTokens > 0 || event.Usage.OutputTokens > 0 {
				usage := chat.TokenUsageDetails{
					InputTokens:  int(event.Usage.InputTokens),
					OutputTokens: int(event.Usage.OutputTokens),
					TotalTokens:  int(event.Usage.InputTokens + event.Usage.OutputTokens),
				}

				// Update usage
				c.state.UpdateUsage(usage)

				totalUsage, _ := c.state.TokenUsage()
				c.logger.Debug("usage from message_delta", "input", usage.InputTokens, "output", usage.OutputTokens, "total", usage.TotalTokens,
					"cumulative_input", totalUsage.Cumulative.InputTokens, "cumulative_output", totalUsage.Cumulative.OutputTokens, "cumulative_total", totalUsage.Cumulative.TotalTokens)
			}
		case "message_stop":
			// Message stream completed
			c.logger.Debug("stream completed via message_stop")
		default:
			// Log unhandled event types at debug level
			c.logger.Debug("unhandled stream event type", "type", event.Type, "event", event)
		}
	}

	if err := stream.Err(); err != nil {
		return chat.Message{}, fmt.Errorf("streaming error: %w", err)
	}

	// Handle tool calls with multiple rounds if needed
	if len(toolCalls) > 0 {
		c.logger.Debug("initial response has tool calls, entering tool call handler", "count", len(toolCalls), "initial_text", respContent.String())
		return c.handleToolCallRounds(ctx, reqMsg, respContent.String(), thinkingContent.String(), thinkingSignature.String(), toolCalls, reqOpts, callback)
	}

	c.logger.Debug("initial response has no tool calls, returning content", "content", respContent.String())

	// Build response message, avoiding empty text content blocks
	respMsg := chat.Message{Role: chat.AssistantRole}
	if respContent.Len() > 0 {
		respMsg.AddText(respContent.String())
	}

	// Add thinking content if present
	if thinkingContent.Len() > 0 {
		respMsg.AddThinking(thinkingContent.String(), thinkingSignature.String())
	}

	// Update history
	c.state.AppendMessages([]chat.Message{reqMsg, respMsg}, nil)

	// Token usage is extracted from message_delta events during streaming

	return respMsg, nil
}

func (c *chatClient) History() (systemPrompt string, msgs []chat.Message) {
	return c.state.History()
}

// TokenUsage returns token usage for both the last message and cumulative session
func (c *chatClient) TokenUsage() (chat.TokenUsage, error) {
	return c.state.TokenUsage()
}

// MaxTokens returns the maximum token limit for the model
func (c *chatClient) MaxTokens() int {
	return c.maxTokens
}

// RegisterTool registers a tool that can be called by the LLM
func (c *chatClient) RegisterTool(tool chat.Tool) error {
	return c.tools.Register(tool)
}

// DeregisterTool removes a tool by name
func (c *chatClient) DeregisterTool(name string) {
	c.tools.Deregister(name)
}

// ListTools returns the names of all registered tools
func (c *chatClient) ListTools() []string {
	return c.tools.List()
}

// mcpToClaudeTool converts an MCP tool definition to Claude format
func (c *chatClient) mcpToClaudeTool(mcpDef chat.ToolDef) (anthropic.ToolUnionParam, error) {
	// Parse the MCP JSON schema to extract the inputSchema
	var mcp struct {
		InputSchema json.RawMessage `json:"inputSchema"`
	}

	jsonSchema := mcpDef.MCPJsonSchema()
	if err := json.Unmarshal([]byte(jsonSchema), &mcp); err != nil {
		return anthropic.ToolUnionParam{}, fmt.Errorf("failed to parse MCP definition: %w", err)
	}

	// Convert inputSchema to ToolInputSchemaParam
	var inputSchema anthropic.ToolInputSchemaParam
	if len(mcp.InputSchema) > 0 {
		// The inputSchema is already in JSON Schema format, which Claude expects
		if err := json.Unmarshal(mcp.InputSchema, &inputSchema); err != nil {
			return anthropic.ToolUnionParam{}, fmt.Errorf("failed to parse input schema: %w", err)
		}
	}

	toolParam := anthropic.ToolParam{
		Name:        mcpDef.Name(),
		InputSchema: inputSchema,
		Type:        anthropic.ToolTypeCustom,
	}

	description := mcpDef.Description()
	if description != "" {
		toolParam.Description = anthropic.String(description)
	}

	return anthropic.ToolUnionParam{
		OfTool: &toolParam,
	}, nil
}

// handleToolCalls processes tool calls from the model and returns tool result content blocks
func (c *chatClient) handleToolCalls(ctx context.Context, toolCalls []anthropic.ToolUseBlock, callback chat.StreamCallback) ([]anthropic.ContentBlockParamUnion, []chat.ToolResult, error) {
	if len(toolCalls) == 0 {
		return nil, nil, nil
	}

	var toolResults []anthropic.ContentBlockParamUnion
	var chatResults []chat.ToolResult

	for _, toolCall := range toolCalls {
		argsStr := string(toolCall.Input)
		result, err := c.tools.Execute(ctx, toolCall.Name, argsStr)

		toolResult := chat.ToolResult{
			ToolCallID: toolCall.ID,
			Name:       toolCall.Name,
		}

		var resultContent string
		if err != nil {
			resultContent = common.FormatToolErrorJSON(err.Error())
			toolResult.Error = err.Error()
		} else {
			resultContent = result
			toolResult.Content = result
		}

		if callback != nil {
			toolResultEvent := chat.StreamEvent{
				Type:        chat.StreamEventTypeToolResult,
				ToolResults: []chat.ToolResult{toolResult},
			}
			if callbackErr := callback(toolResultEvent); callbackErr != nil {
				return nil, nil, fmt.Errorf("callback error: %w", callbackErr)
			}
		}

		if err != nil {
			errorResult := anthropic.NewToolResultBlock(toolCall.ID, resultContent, true)
			toolResults = append(toolResults, errorResult)
			chatResults = append(chatResults, toolResult)
			continue
		}

		c.logger.Debug("tool executed", "name", toolCall.Name, "args", argsStr, "result", result)

		resultBlock := anthropic.NewToolResultBlock(toolCall.ID, result, false)
		toolResults = append(toolResults, resultBlock)
		chatResults = append(chatResults, toolResult)
	}

	return toolResults, chatResults, nil
}

func claudeToolUseToChat(tool anthropic.ToolUseBlock) chat.ToolCall {
	var args json.RawMessage
	if len(tool.Input) > 0 {
		args = append(json.RawMessage(nil), tool.Input...)
	}
	return chat.ToolCall{
		ID:        tool.ID,
		Name:      tool.Name,
		Arguments: args,
	}
}

func claudeToolResultBlock(tr chat.ToolResult) anthropic.ContentBlockParamUnion {
	content := tr.Content
	isError := false
	if tr.Error != "" {
		isError = true
		content = common.FormatToolErrorJSON(tr.Error)
	}
	if content == "" {
		content = "{}"
	}
	return anthropic.NewToolResultBlock(tr.ToolCallID, content, isError)
}

// messageParam converts a chat.Message to an anthropic.MessageParam.
//
// IMPORTANT INVARIANT: Tool results must NEVER be stored in assistant messages.
// - Assistant messages contain only text content and tool calls (ToolUseBlock)
// - Tool results must be in separate ToolRole messages (converted to User role by messageParam)
// This separation is enforced throughout the codebase when constructing messages.
//
// Returns an error if the message has no contents or no valid content blocks.
func messageParam(msg chat.Message) (anthropic.MessageParam, error) {
	if len(msg.Contents) == 0 {
		return anthropic.MessageParam{}, fmt.Errorf("message has no contents")
	}

	var blocks []anthropic.ContentBlockParamUnion

	// Build content blocks from all contents
	for _, content := range msg.Contents {
		// Handle text content
		if content.Text != "" {
			blocks = append(blocks, anthropic.NewTextBlock(content.Text))
		}

		// Handle tool call content
		if content.ToolCall != nil {
			blocks = append(blocks, anthropic.NewToolUseBlock(
				content.ToolCall.ID,
				content.ToolCall.Arguments,
				content.ToolCall.Name,
			))
		}

		// Handle tool result content
		if content.ToolResult != nil {
			blocks = append(blocks, claudeToolResultBlock(*content.ToolResult))
		}
	}

	// Check if we have any valid blocks
	if len(blocks) == 0 {
		return anthropic.MessageParam{}, fmt.Errorf("message has no valid content blocks")
	}

	// Convert based on role
	switch msg.Role {
	case chat.UserRole, "system", chat.ToolRole:
		// Claude API requirement: Tool results must be in User role messages, not a separate tool role.
		// System messages and ToolRole messages are converted to User messages.
		// This is different from OpenAI (which has a "tool" role) and Gemini (which uses "function" role).
		return anthropic.NewUserMessage(blocks...), nil
	case chat.AssistantRole:
		return anthropic.NewAssistantMessage(blocks...), nil
	default:
		// Unknown role, treat as user message
		return anthropic.NewUserMessage(blocks...), nil
	}
}

// handleToolCallRounds handles potentially multiple rounds of tool calls
func (c *chatClient) handleToolCallRounds(ctx context.Context, initialMsg chat.Message, initialContent string, initialThinkingText string, initialThinkingSignature string, initialToolCalls []anthropic.ToolUseBlock, reqOpts chat.Options, callback chat.StreamCallback) (chat.Message, error) {
	// Keep track of all content blocks for the conversation
	var msgs []anthropic.MessageParam

	// Build initial conversation with system prompt and history
	// Snapshot history with minimal lock
	systemPrompt, history := c.state.Snapshot()

	// Add history
	for _, m := range history {
		param, err := messageParam(m)
		if err != nil {
			return chat.Message{}, fmt.Errorf("converting history message to param: %w", err)
		}
		msgs = append(msgs, param)
	}

	// Add the initial user message with system reminder prepended if present
	userBlocks := []anthropic.ContentBlockParamUnion{}
	chatInitialMsg := chat.Message{Role: chat.UserRole}
	if reminder := getSystemReminderText(ctx); reminder != "" {
		userBlocks = append(userBlocks, anthropic.NewTextBlock(reminder))
		// Add system reminder to the message we'll persist
		chatInitialMsg.Contents = append(chatInitialMsg.Contents, chat.Content{SystemReminder: reminder})
	}
	userBlocks = append(userBlocks, anthropic.NewTextBlock(initialMsg.GetText()))
	chatInitialMsg.Contents = append(chatInitialMsg.Contents, chat.Content{Text: initialMsg.GetText()})
	msgs = append(msgs, anthropic.NewUserMessage(userBlocks...))

	// Persist the user message WITH system reminder for complete audit trail
	c.state.AppendMessages([]chat.Message{chatInitialMsg}, nil)

	// Process tool calls in a loop until we get a final response
	toolCalls := initialToolCalls

	c.logger.Debug("starting tool call rounds", "initial_tool_count", len(initialToolCalls))

	for len(toolCalls) > 0 {
		c.logger.Debug("tool execution round", "tool_count", len(toolCalls))
		for i, tc := range toolCalls {
			c.logger.Debug("tool call", "index", i+1, "name", tc.Name, "input", string(tc.Input))
		}
		// Execute tool calls
		toolResults, chatToolResults, err := c.handleToolCalls(ctx, toolCalls, callback)
		if err != nil {
			return chat.Message{}, fmt.Errorf("failed to execute tool calls: %w", err)
		}

		var assistantContentBlocks []anthropic.ContentBlockParamUnion
		if initialContent != "" {
			assistantContentBlocks = append(assistantContentBlocks, anthropic.NewTextBlock(initialContent))
		}
		chatToolCalls := make([]chat.ToolCall, len(toolCalls))
		for i, toolCall := range toolCalls {
			toolUseBlock := anthropic.NewToolUseBlock(toolCall.ID, toolCall.Input, toolCall.Name)
			assistantContentBlocks = append(assistantContentBlocks, toolUseBlock)
			chatToolCalls[i] = claudeToolUseToChat(toolCall)
		}

		assistantMsg := anthropic.NewAssistantMessage(assistantContentBlocks...)
		msgs = append(msgs, assistantMsg)

		// Build assistant message, avoiding empty text content blocks
		chatAssistantMsg := chat.Message{Role: chat.AssistantRole}
		if initialContent != "" {
			chatAssistantMsg.AddText(initialContent)
		}
		if initialThinkingText != "" {
			chatAssistantMsg.AddThinking(initialThinkingText, initialThinkingSignature)
		}
		for _, tc := range chatToolCalls {
			chatAssistantMsg.AddToolCall(tc)
		}
		stateMessages := []chat.Message{chatAssistantMsg}
		if len(chatToolResults) > 0 {
			toolMsg := chat.Message{Role: chat.ToolRole}
			for _, tr := range chatToolResults {
				toolMsg.AddToolResult(tr)
			}
			// Add system reminder AFTER tool results (Claude's ordering requirement)
			// This ensures the complete message is persisted for audit trail
			if reminder := getSystemReminderText(ctx); reminder != "" {
				toolMsg.Contents = append(toolMsg.Contents, chat.Content{SystemReminder: reminder})
			}
			stateMessages = append(stateMessages, toolMsg)
		}
		c.state.AppendMessages(stateMessages, nil)
		initialContent = ""

		// Only create user message if we have tool results
		// Empty messages would cause "text content blocks must be non-empty" error
		if len(toolResults) > 0 {
			// Build message with tool results first, then system reminder
			// Claude requires tool_result blocks to immediately follow tool_use blocks
			resultBlocks := []anthropic.ContentBlockParamUnion{}
			resultBlocks = append(resultBlocks, toolResults...)
			if reminder := getSystemReminderText(ctx); reminder != "" {
				resultBlocks = append(resultBlocks, anthropic.NewTextBlock(reminder))
			}
			userMsg := anthropic.NewUserMessage(resultBlocks...)
			msgs = append(msgs, userMsg)
		}

		// Make another API call with tool results
		followUpParams := anthropic.MessageNewParams{
			Messages:  msgs,
			Model:     anthropic.Model(c.modelName),
			MaxTokens: 4096,
		}

		// Add system prompt if present
		var systemBlocks []anthropic.TextBlockParam
		if systemPrompt != "" {
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
				Text: systemPrompt,
				Type: "text",
			})
		}

		if len(systemBlocks) > 0 {
			followUpParams.System = systemBlocks
		}

		if reqOpts.Temperature != nil {
			followUpParams.Temperature = anthropic.Float(*reqOpts.Temperature)
		}

		if reqOpts.MaxTokens > 0 {
			followUpParams.MaxTokens = int64(reqOpts.MaxTokens)
		}

		// Add tools if registered (for follow-up after tool execution)
		allTools := c.tools.GetAll()
		if len(allTools) > 0 {
			tools := make([]anthropic.ToolUnionParam, 0, len(allTools))
			for _, tool := range allTools {
				toolParam, err := c.mcpToClaudeTool(tool)
				if err != nil {
					// Skip this tool on error
					continue
				}
				tools = append(tools, toolParam)
			}
			followUpParams.Tools = tools
		}

		// Create a new stream for the follow-up request
		followUpStream := c.anthropicClient.Messages.NewStreaming(ctx, followUpParams)

		// Process the follow-up stream
		var respContent strings.Builder
		var followUpThinkingContent strings.Builder
		var followUpThinkingSignature strings.Builder
		// Preserve any initial content from before the tool calls
		if initialContent != "" {
			respContent.WriteString(initialContent)
			initialContent = "" // Only use it once
		}
		toolCalls = nil // Reset for next round
		var currentToolCall *anthropic.ToolUseBlock
		var toolCallArgs strings.Builder

		for followUpStream.Next() {
			event := followUpStream.Current()

			// Handle different event types similar to main streaming logic
			switch event.Type {
			case "content_block_start":
				if event.ContentBlock.Type == "tool_use" {
					// Start of a tool use block
					currentToolCall = &anthropic.ToolUseBlock{
						ID:   event.ContentBlock.ID,
						Name: event.ContentBlock.Name,
					}
					toolCallArgs.Reset()
					c.logger.Debug("follow-up tool use start", "id", event.ContentBlock.ID, "name", event.ContentBlock.Name, "input", event.ContentBlock.Input)

					// Don't emit tool call event yet - wait for arguments to be accumulated
					if event.ContentBlock.Input != nil {
						// Input is sometimes provided in the start event
						inputBytes, err := json.Marshal(event.ContentBlock.Input)
						if err == nil {
							currentToolCall.Input = json.RawMessage(inputBytes)
							c.logger.Debug("follow-up set tool input from start event", "input", string(inputBytes))
						}
					}
				} else if event.ContentBlock.Type == "thinking" {
					// Thinking block in follow-up
					followUpThinkingContent.Reset()
					followUpThinkingSignature.Reset()
					if callback != nil {
						thinkingEvent := chat.StreamEvent{
							Type:           chat.StreamEventTypeThinking,
							ThinkingStatus: &chat.ThinkingStatus{},
						}
						if err := callback(thinkingEvent); err != nil {
							return chat.Message{}, err
						}
					}
				} else if event.ContentBlock.Type == "redacted_thinking" {
					// Redacted thinking block in follow-up
					c.logger.Debug("follow-up redacted thinking block detected", "data", event.ContentBlock.Data)
					if callback != nil {
						redactedEvent := chat.StreamEvent{
							Type: chat.StreamEventTypeRedactedThinking,
							ThinkingStatus: &chat.ThinkingStatus{
								RedactedData: event.ContentBlock.Data,
							},
						}
						if err := callback(redactedEvent); err != nil {
							return chat.Message{}, err
						}
					}
				} else if event.ContentBlock.Type == "server_tool_use" {
					// Server-side tool invocation in follow-up
					c.logger.Debug("follow-up server tool use", "id", event.ContentBlock.ID, "name", event.ContentBlock.Name, "input", event.ContentBlock.Input)
					if callback != nil {
						serverToolEvent := chat.StreamEvent{
							Type: chat.StreamEventTypeServerToolUse,
							ToolCalls: []chat.ToolCall{
								{
									ID:        event.ContentBlock.ID,
									Name:      event.ContentBlock.Name,
									Arguments: nil,
								},
							},
						}
						if err := callback(serverToolEvent); err != nil {
							return chat.Message{}, err
						}
					}
				} else if event.ContentBlock.Type == "web_search_tool_result" {
					// Web search results in follow-up
					c.logger.Debug("follow-up web search result", "tool_use_id", event.ContentBlock.ToolUseID, "content", event.ContentBlock.Content)
					if callback != nil {
						webSearchEvent := chat.StreamEvent{
							Type:    chat.StreamEventTypeWebSearchResult,
							Content: "Web search results received in follow-up",
						}
						if err := callback(webSearchEvent); err != nil {
							return chat.Message{}, err
						}
					}
				}
			case "content_block_delta":
				// Handle different delta types similar to main streaming
				switch event.Delta.Type {
				case "text_delta":
					content := event.Delta.Text
					respContent.WriteString(content)
					if callback != nil {
						streamEvent := chat.StreamEvent{
							Type:    chat.StreamEventTypeContent,
							Content: content,
						}
						if err := callback(streamEvent); err != nil {
							return chat.Message{}, err
						}
					}
				case "thinking_delta":
					// Direct thinking delta in follow-up
					followUpThinkingContent.WriteString(event.Delta.Thinking)
					if callback != nil {
						thinkingEvent := chat.StreamEvent{
							Type:           chat.StreamEventTypeThinking,
							Content:        event.Delta.Thinking,
							ThinkingStatus: &chat.ThinkingStatus{},
						}
						if err := callback(thinkingEvent); err != nil {
							return chat.Message{}, err
						}
					}
				case "signature_delta":
					// Thinking block signature in follow-up
					followUpThinkingSignature.WriteString(event.Delta.Signature)
					c.logger.Debug("follow-up got signature_delta", "signature", event.Delta.Signature)
				case "citations_delta":
					// Citation updates in follow-up
					c.logger.Debug("follow-up got citations_delta", "citation", event.Delta.Citation)
				case "input_json_delta":
					// Tool use input delta
					if currentToolCall != nil {
						if partialJSON := event.Delta.PartialJSON; partialJSON != "" {
							toolCallArgs.WriteString(partialJSON)
						}
					}
				default:
					// Handle backwards compatibility
					if event.Delta.Text != "" && event.Delta.Type == "" {
						content := event.Delta.Text
						respContent.WriteString(content)
						if callback != nil {
							streamEvent := chat.StreamEvent{
								Type:    chat.StreamEventTypeContent,
								Content: content,
							}
							if err := callback(streamEvent); err != nil {
								return chat.Message{}, err
							}
						}
					} else if event.Delta.Type != "" {
						c.logger.Debug("follow-up unhandled delta type", "type", event.Delta.Type, "delta", event.Delta)
					}
				}
			case "content_block_stop":
				// Finalize current tool call if we have one
				if currentToolCall != nil {
					// Prefer accumulated deltas over start event input
					if toolCallArgs.Len() > 0 {
						currentToolCall.Input = json.RawMessage(toolCallArgs.String())
						c.logger.Debug("follow-up set tool input from deltas", "input", toolCallArgs.String())
					}

					// Now emit the tool call event with complete arguments
					if callback != nil {
						toolCallEvent := chat.StreamEvent{
							Type: chat.StreamEventTypeToolCall,
							ToolCalls: []chat.ToolCall{
								{
									ID:        currentToolCall.ID,
									Name:      currentToolCall.Name,
									Arguments: currentToolCall.Input,
								},
							},
						}
						if err := callback(toolCallEvent); err != nil {
							return chat.Message{}, err
						}
					}

					c.logger.Debug("follow-up finalizing tool call", "id", currentToolCall.ID, "name", currentToolCall.Name, "input", string(currentToolCall.Input))
					toolCalls = append(toolCalls, *currentToolCall)
					currentToolCall = nil
					toolCallArgs.Reset()
				}
			case "message_delta":
				// Check for usage information in follow-up message delta
				if event.Usage.InputTokens > 0 || event.Usage.OutputTokens > 0 {
					usage := chat.TokenUsageDetails{
						InputTokens:  int(event.Usage.InputTokens),
						OutputTokens: int(event.Usage.OutputTokens),
						TotalTokens:  int(event.Usage.InputTokens + event.Usage.OutputTokens),
					}

					// Update usage
					c.state.UpdateUsage(usage)

					totalUsage, _ := c.state.TokenUsage()
					c.logger.Debug("follow-up usage from message_delta", "input", usage.InputTokens, "output", usage.OutputTokens, "total", usage.TotalTokens,
						"cumulative_input", totalUsage.Cumulative.InputTokens, "cumulative_output", totalUsage.Cumulative.OutputTokens, "cumulative_total", totalUsage.Cumulative.TotalTokens)
				}
			case "message_stop":
				// Follow-up message stream completed
				c.logger.Debug("follow-up stream completed via message_stop")
			default:
				// Log unhandled event types at debug level
				c.logger.Debug("follow-up unhandled stream event type", "type", event.Type, "event", event)
			}
		}

		if err := followUpStream.Err(); err != nil {
			return chat.Message{}, fmt.Errorf("follow-up streaming error: %w", err)
		}

		// If we got more tool calls, continue the loop
		if len(toolCalls) > 0 {
			c.logger.Debug("got more tool calls, continuing", "count", len(toolCalls))
			continue
		}

		c.logger.Debug("no more tool calls, got final response", "response", respContent.String())

		// No more tool calls, we have the final response
		// Build final message, avoiding empty text content blocks
		finalMsg := chat.Message{Role: chat.AssistantRole}
		if respContent.Len() > 0 {
			finalMsg.AddText(respContent.String())
		}

		// Add thinking content if present from follow-up rounds
		if followUpThinkingContent.Len() > 0 {
			finalMsg.AddThinking(followUpThinkingContent.String(), followUpThinkingSignature.String())
		}

		c.logger.Debug("returning final response from tool handler", "content_length", len(finalMsg.GetText()))

		// Update history with final assistant response (user message already persisted)
		c.state.AppendMessages([]chat.Message{finalMsg}, nil)

		return finalMsg, nil
	}

	// This should never be reached since the loop continues until no tool calls
	c.logger.Error("unexpected end of tool call processing", "initial_tool_count", len(initialToolCalls))
	return chat.Message{}, fmt.Errorf("unexpected end of tool call processing")
}
