package claude

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/internal/common"
)

const (
	ClaudeURL = "https://api.anthropic.com/v1"
)

type client struct {
	anthropicClient anthropic.Client
	modelName       string
	debug           bool
}

var _ chat.Client = &client{}

type Option func(*client)

func WithModel(modelName string) Option {
	return func(c *client) {
		c.modelName = strings.TrimSpace(modelName)
	}
}

func WithDebug(debug bool) Option {
	return func(c *client) {
		c.debug = debug
	}
}

// NewClient returns a chat client that can begin chat sessions with Claude's Messages API.
func NewClient(apiBase string, apiKey string, opts ...Option) (chat.Client, error) {
	c := &client{}

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

	if apiBase != "" && apiBase != ClaudeURL {
		clientOpts = append(clientOpts, option.WithBaseURL(apiBase))
	}

	c.anthropicClient = anthropic.NewClient(clientOpts...)

	return c, nil
}

// NewChat returns a chat instance.
func (c client) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	// Determine max tokens based on model
	maxTokens := getModelMaxTokens(c.modelName)

	return &chatClient{
		client:       c,
		systemPrompt: systemPrompt,
		msgs:         initialMsgs,
		maxTokens:    maxTokens,
		tools:        make(map[string]common.RegisteredTool),
	}
}

var modelLimits = []chat.ModelTokenLimits{
	{Model: "claude-opus-4-1", TokenLimits: chat.TokenLimits{Context: 200000, Output: 32000}},
	{Model: "claude-opus-4", TokenLimits: chat.TokenLimits{Context: 200000, Output: 32000}},
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

type chatClient struct {
	client
	systemPrompt string

	mu   sync.Mutex
	msgs []chat.Message

	// Token tracking
	cumulativeUsage  chat.TokenUsageDetails
	lastMessageUsage chat.TokenUsageDetails
	maxTokens        int

	// Tool support
	tools     map[string]common.RegisteredTool
	toolsLock sync.RWMutex
}

func (c *chatClient) MessageStream(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	reqMsg := msg
	reqOpts := chat.ApplyOptions(opts...)

	c.mu.Lock()
	defer c.mu.Unlock()

	// Build message list for Claude
	var messages []anthropic.MessageParam

	// Add history
	for _, m := range c.msgs {
		switch m.Role {
		case chat.UserRole:
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(m.Content),
			))
		case chat.AssistantRole:
			messages = append(messages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(m.Content),
			))
		default:
			// Claude doesn't support system role in messages, only as a separate field
			continue
		}
	}

	// Add current message
	switch msg.Role {
	case chat.UserRole:
		messages = append(messages, anthropic.NewUserMessage(
			anthropic.NewTextBlock(msg.Content),
		))
	case chat.AssistantRole:
		messages = append(messages, anthropic.NewAssistantMessage(
			anthropic.NewTextBlock(msg.Content),
		))
	default:
		// If it's a system message, convert to user message
		messages = append(messages, anthropic.NewUserMessage(
			anthropic.NewTextBlock(msg.Content),
		))
	}

	// Build request parameters
	params := anthropic.MessageNewParams{
		Messages:  messages,
		Model:     anthropic.Model(c.modelName),
		MaxTokens: 4096, // Claude requires this
	}

	// Add tools if registered
	var toolConversionErr error
	func() {
		c.toolsLock.RLock()
		defer c.toolsLock.RUnlock()

		if len(c.tools) > 0 {
			tools := make([]anthropic.ToolUnionParam, 0, len(c.tools))
			for _, tool := range c.tools {
				toolParam, err := c.mcpToClaudeTool(tool.Definition)
				if err != nil {
					toolConversionErr = fmt.Errorf("failed to convert tool: %w", err)
					return
				}
				tools = append(tools, toolParam)
			}
			params.Tools = tools
		}
	}()

	if toolConversionErr != nil {
		return chat.Message{}, toolConversionErr
	}

	// Add system prompt if present
	if c.systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{
				Text: c.systemPrompt,
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
		systemText := c.systemPrompt
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
	var toolCalls []anthropic.ToolUseBlock
	var currentToolCall *anthropic.ToolUseBlock
	var toolCallArgs strings.Builder

	for stream.Next() {
		event := stream.Current()
		if c.debug {
			fmt.Fprintf(os.Stderr, "[Claude] Stream event type: %s\n", event.Type)
		}
		// Handle different event types
		switch event.Type {
		case "message_start":
			// Check if this is a model that supports thinking (e.g., claude-3-5-sonnet-20241022)
			if strings.Contains(c.modelName, "claude-3-5-sonnet") && callback != nil {
				// Some Claude models might support thinking, emit initial thinking event
				thinkingEvent := chat.StreamEvent{
					Type: chat.StreamEventTypeThinking,
					ThinkingStatus: &chat.ThinkingStatus{
						IsThinking: true,
					},
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
						Type: chat.StreamEventTypeThinking,
						ThinkingStatus: &chat.ThinkingStatus{
							IsThinking: true,
						},
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
				if c.debug {
					fmt.Fprintf(os.Stderr, "[Claude] Tool use start: ID=%s, Name=%s, Input=%v\n",
						event.ContentBlock.ID, event.ContentBlock.Name, event.ContentBlock.Input)
				}

				// Emit tool call event
				if callback != nil {
					toolCallEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeToolCall,
						ToolCalls: []chat.ToolCall{
							{
								ID:        event.ContentBlock.ID,
								Name:      event.ContentBlock.Name,
								Arguments: json.RawMessage("{}"), // Initial empty arguments
							},
						},
					}
					if err := callback(toolCallEvent); err != nil {
						return chat.Message{}, err
					}
				}
				if event.ContentBlock.Input != nil {
					// Input is provided in the start event - Claude usually provides complete input here
					inputBytes, err := json.Marshal(event.ContentBlock.Input)
					if err == nil {
						currentToolCall.Input = json.RawMessage(inputBytes)
						if c.debug {
							fmt.Fprintf(os.Stderr, "[Claude] Set tool input from start event: %s\n", string(inputBytes))
						}
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
							IsThinking: false,
							Summary:    thinkingContent.String(),
						},
					}
					if err := callback(thinkingSummaryEvent); err != nil {
						return chat.Message{}, err
					}
				}
			}
		case "content_block_delta":
			// Check if Delta has text content
			if event.Delta.Text != "" {
				content := event.Delta.Text

				if inThinking {
					// Accumulate thinking content
					thinkingContent.WriteString(content)
					if callback != nil {
						// Stream thinking updates
						thinkingEvent := chat.StreamEvent{
							Type:    chat.StreamEventTypeThinking,
							Content: content,
							ThinkingStatus: &chat.ThinkingStatus{
								IsThinking: true,
							},
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
			}
			// Check if this is a tool use input delta
			if currentToolCall != nil && event.Delta.Type == "input_json_delta" {
				// Accumulate tool call arguments from streaming delta
				if partialJSON := event.Delta.PartialJSON; partialJSON != "" {
					if c.debug {
						fmt.Fprintf(os.Stderr, "[Claude] Got input_json_delta: %s\n", partialJSON)
					}
					toolCallArgs.WriteString(partialJSON)
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
							IsThinking: false,
							Summary:    thinkingContent.String(),
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
					if c.debug {
						fmt.Fprintf(os.Stderr, "[Claude] Set tool input from deltas: %s\n", toolCallArgs.String())
					}
				}
				if c.debug {
					fmt.Fprintf(os.Stderr, "[Claude] Finalizing tool call: ID=%s, Name=%s, Input=%s\n",
						currentToolCall.ID, currentToolCall.Name, string(currentToolCall.Input))
				}
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
				c.lastMessageUsage = usage
				c.cumulativeUsage.InputTokens += usage.InputTokens
				c.cumulativeUsage.OutputTokens += usage.OutputTokens
				c.cumulativeUsage.TotalTokens += usage.TotalTokens
				if c.debug {
					fmt.Fprintf(os.Stderr, "[Claude] Usage from message_delta - Input: %d, Output: %d, Total: %d (Cumulative - Input: %d, Output: %d, Total: %d)\n",
						usage.InputTokens, usage.OutputTokens, usage.TotalTokens,
						c.cumulativeUsage.InputTokens, c.cumulativeUsage.OutputTokens, c.cumulativeUsage.TotalTokens)
				}
			}
		case "message_stop":
			// Message stream completed
			if c.debug {
				fmt.Fprintf(os.Stderr, "[Claude] Stream completed via message_stop\n")
			}
		}
	}

	if err := stream.Err(); err != nil {
		return chat.Message{}, fmt.Errorf("streaming error: %w", err)
	}

	// Handle tool calls with multiple rounds if needed
	if len(toolCalls) > 0 {
		return c.handleToolCallRounds(ctx, reqMsg, toolCalls, reqOpts, callback)
	}

	respMsg := chat.Message{
		Role:    chat.AssistantRole,
		Content: respContent.String(),
	}

	// Update history
	c.msgs = append(c.msgs, reqMsg)
	c.msgs = append(c.msgs, respMsg)

	// Token usage is extracted from message_delta events during streaming

	return respMsg, nil
}

func (c *chatClient) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	// Use MessageStream with nil callback
	return c.MessageStream(ctx, msg, nil, opts...)
}

func (c *chatClient) History() (systemPrompt string, msgs []chat.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	msgs = make([]chat.Message, len(c.msgs))
	copy(msgs, c.msgs)

	return c.systemPrompt, msgs
}

// TokenUsage returns token usage for both the last message and cumulative session
func (c *chatClient) TokenUsage() (chat.TokenUsage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return chat.TokenUsage{
		LastMessage: c.lastMessageUsage,
		Cumulative:  c.cumulativeUsage,
	}, nil
}

// MaxTokens returns the maximum token limit for the model
func (c *chatClient) MaxTokens() int {
	return c.maxTokens
}

// RegisterTool registers a tool with its MCP definition and handler function
func (c *chatClient) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
	c.toolsLock.Lock()
	defer c.toolsLock.Unlock()

	toolName := def.Name()
	if toolName == "" {
		return fmt.Errorf("tool definition missing name")
	}

	c.tools[toolName] = common.RegisteredTool{
		Definition: def,
		Handler:    fn,
	}

	return nil
}

// DeregisterTool removes a tool by name
func (c *chatClient) DeregisterTool(name string) {
	c.toolsLock.Lock()
	defer c.toolsLock.Unlock()
	delete(c.tools, name)
}

// ListTools returns the names of all registered tools
func (c *chatClient) ListTools() []string {
	c.toolsLock.RLock()
	defer c.toolsLock.RUnlock()

	names := make([]string, 0, len(c.tools))
	for name := range c.tools {
		names = append(names, name)
	}
	return names
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
func (c *chatClient) handleToolCalls(ctx context.Context, toolCalls []anthropic.ToolUseBlock) ([]anthropic.ContentBlockParamUnion, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}

	var toolResults []anthropic.ContentBlockParamUnion

	c.toolsLock.RLock()
	defer c.toolsLock.RUnlock()

	for _, toolCall := range toolCalls {
		tool, exists := c.tools[toolCall.Name]
		if !exists {
			// Tool not found, return error message
			errorResult := anthropic.NewToolResultBlock(toolCall.ID, fmt.Sprintf(`{"error": "Tool %s not found"}`, toolCall.Name), true)
			toolResults = append(toolResults, errorResult)
			continue
		}

		// Execute the tool
		argsStr := string(toolCall.Input)
		result := tool.Handler(ctx, argsStr)

		if c.debug {
			fmt.Fprintf(os.Stderr, "[Claude] Tool %s executed with args %s, result: %s\n",
				toolCall.Name, argsStr, result)
		}

		// Add tool result content block
		resultBlock := anthropic.NewToolResultBlock(toolCall.ID, result, false)
		toolResults = append(toolResults, resultBlock)
	}

	return toolResults, nil
}

// handleToolCallRounds handles potentially multiple rounds of tool calls
func (c *chatClient) handleToolCallRounds(ctx context.Context, initialMsg chat.Message, initialToolCalls []anthropic.ToolUseBlock, reqOpts chat.Options, callback chat.StreamCallback) (chat.Message, error) {
	// Keep track of all content blocks for the conversation
	var conversationMessages []anthropic.MessageParam

	// Build initial conversation with system prompt and history
	// Add history
	for _, m := range c.msgs {
		switch m.Role {
		case chat.UserRole:
			conversationMessages = append(conversationMessages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(m.Content),
			))
		case chat.AssistantRole:
			conversationMessages = append(conversationMessages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(m.Content),
			))
		default:
			// Convert system messages to user messages for Claude
			conversationMessages = append(conversationMessages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(m.Content),
			))
		}
	}

	// Add the initial user message
	c.msgs = append(c.msgs, initialMsg)
	conversationMessages = append(conversationMessages, anthropic.NewUserMessage(
		anthropic.NewTextBlock(initialMsg.Content),
	))

	// Process tool calls in a loop until we get a final response
	toolCalls := initialToolCalls
	maxRounds := 10 // Prevent infinite loops

	for round := 0; round < maxRounds && len(toolCalls) > 0; round++ {
		// Debug logging
		if c.debug {
			fmt.Fprintf(os.Stderr, "[Claude] Tool execution round %d with %d tool calls\n", round+1, len(toolCalls))
			for i, tc := range toolCalls {
				fmt.Fprintf(os.Stderr, "[Claude] Tool %d: %s with input: %s\n", i+1, tc.Name, string(tc.Input))
			}
		}
		// Execute tool calls
		toolResults, err := c.handleToolCalls(ctx, toolCalls)
		if err != nil {
			return chat.Message{}, fmt.Errorf("failed to execute tool calls: %w", err)
		}

		// Create assistant message with tool calls
		var assistantContentBlocks []anthropic.ContentBlockParamUnion
		for _, toolCall := range toolCalls {
			toolUseBlock := anthropic.NewToolUseBlock(toolCall.ID, toolCall.Input, toolCall.Name)
			assistantContentBlocks = append(assistantContentBlocks, toolUseBlock)
		}

		// Add assistant message with tool calls
		assistantMsg := anthropic.NewAssistantMessage(assistantContentBlocks...)
		conversationMessages = append(conversationMessages, assistantMsg)

		// Add tool results as user message
		userMsg := anthropic.NewUserMessage(toolResults...)
		conversationMessages = append(conversationMessages, userMsg)

		// Make another API call with tool results
		followUpParams := anthropic.MessageNewParams{
			Messages:  conversationMessages,
			Model:     anthropic.Model(c.modelName),
			MaxTokens: 4096,
		}

		// Add system prompt if present
		if c.systemPrompt != "" {
			followUpParams.System = []anthropic.TextBlockParam{
				{
					Text: c.systemPrompt,
					Type: "text",
				},
			}
		}

		if reqOpts.Temperature != nil {
			followUpParams.Temperature = anthropic.Float(*reqOpts.Temperature)
		}

		if reqOpts.MaxTokens > 0 {
			followUpParams.MaxTokens = int64(reqOpts.MaxTokens)
		}

		// Add tools if registered (for follow-up after tool execution)
		func() {
			c.toolsLock.RLock()
			defer c.toolsLock.RUnlock()

			if len(c.tools) > 0 {
				tools := make([]anthropic.ToolUnionParam, 0, len(c.tools))
				for _, tool := range c.tools {
					toolParam, err := c.mcpToClaudeTool(tool.Definition)
					if err != nil {
						// Skip this tool on error
						continue
					}
					tools = append(tools, toolParam)
				}
				followUpParams.Tools = tools
			}
		}()

		// Create a new stream for the follow-up request
		followUpStream := c.anthropicClient.Messages.NewStreaming(ctx, followUpParams)

		// Process the follow-up stream
		var respContent strings.Builder
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
					if c.debug {
						fmt.Fprintf(os.Stderr, "[Claude] Follow-up tool use start: ID=%s, Name=%s, Input=%v\n",
							event.ContentBlock.ID, event.ContentBlock.Name, event.ContentBlock.Input)
					}

					// Emit tool call event
					if callback != nil {
						toolCallEvent := chat.StreamEvent{
							Type: chat.StreamEventTypeToolCall,
							ToolCalls: []chat.ToolCall{
								{
									ID:        event.ContentBlock.ID,
									Name:      event.ContentBlock.Name,
									Arguments: json.RawMessage("{}"), // Initial empty arguments
								},
							},
						}
						if err := callback(toolCallEvent); err != nil {
							return chat.Message{}, err
						}
					}
					if event.ContentBlock.Input != nil {
						// Input is provided in the start event - Claude usually provides complete input here
						inputBytes, err := json.Marshal(event.ContentBlock.Input)
						if err == nil {
							currentToolCall.Input = json.RawMessage(inputBytes)
							if c.debug {
								fmt.Fprintf(os.Stderr, "[Claude] Follow-up set tool input from start event: %s\n", string(inputBytes))
							}
						}
					}
				}
			case "content_block_delta":
				if event.Delta.Text != "" {
					content := event.Delta.Text
					respContent.WriteString(content)

					// Call the callback with the content event
					if callback != nil {
						streamEvent := chat.StreamEvent{
							Type:    chat.StreamEventTypeContent,
							Content: content,
						}
						if err := callback(streamEvent); err != nil {
							return chat.Message{}, err
						}
					}
				}
				// Check if this is a tool use input delta
				if currentToolCall != nil && event.Delta.Type == "input_json_delta" {
					if partialJSON := event.Delta.PartialJSON; partialJSON != "" {
						toolCallArgs.WriteString(partialJSON)
					}
				}
			case "content_block_stop":
				// Finalize current tool call if we have one
				if currentToolCall != nil {
					// Prefer accumulated deltas over start event input
					if toolCallArgs.Len() > 0 {
						currentToolCall.Input = json.RawMessage(toolCallArgs.String())
						if c.debug {
							fmt.Fprintf(os.Stderr, "[Claude] Follow-up set tool input from deltas: %s\n", toolCallArgs.String())
						}
					}
					if c.debug {
						fmt.Fprintf(os.Stderr, "[Claude] Follow-up finalizing tool call: ID=%s, Name=%s, Input=%s\n",
							currentToolCall.ID, currentToolCall.Name, string(currentToolCall.Input))
					}
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
					c.lastMessageUsage = usage
					c.cumulativeUsage.InputTokens += usage.InputTokens
					c.cumulativeUsage.OutputTokens += usage.OutputTokens
					c.cumulativeUsage.TotalTokens += usage.TotalTokens
					if c.debug {
						fmt.Fprintf(os.Stderr, "[Claude] Follow-up usage from message_delta - Input: %d, Output: %d, Total: %d (Cumulative - Input: %d, Output: %d, Total: %d)\n",
							usage.InputTokens, usage.OutputTokens, usage.TotalTokens,
							c.cumulativeUsage.InputTokens, c.cumulativeUsage.OutputTokens, c.cumulativeUsage.TotalTokens)
					}
				}
			case "message_stop":
				// Follow-up message stream completed
				if c.debug {
					fmt.Fprintf(os.Stderr, "[Claude] Follow-up stream completed via message_stop\n")
				}
			}
		}

		if err := followUpStream.Err(); err != nil {
			return chat.Message{}, fmt.Errorf("follow-up streaming error: %w", err)
		}

		// If we got more tool calls, continue the loop
		if len(toolCalls) > 0 {
			if c.debug {
				fmt.Fprintf(os.Stderr, "[Claude] Got %d more tool calls in round %d, continuing\n", len(toolCalls), round+1)
			}
			continue
		}

		if c.debug {
			fmt.Fprintf(os.Stderr, "[Claude] No more tool calls, got final response: %q\n", respContent.String())
		}

		// No more tool calls, we have the final response
		finalMsg := chat.Message{
			Role:    chat.AssistantRole,
			Content: respContent.String(),
		}

		// Update history with the final response
		c.msgs = append(c.msgs, finalMsg)

		return finalMsg, nil
	}

	// If we get here, we exceeded maxRounds
	return chat.Message{}, fmt.Errorf("exceeded maximum rounds of tool calls (%d)", maxRounds)
}
