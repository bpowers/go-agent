package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"google.golang.org/genai"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/internal/common"
)

type client struct {
	genaiClient *genai.Client
	modelName   string
	debug       bool
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

// NewClient returns a chat client that can begin chat sessions with Google's Gemini API.
func NewClient(apiKey string, opts ...Option) (chat.Client, error) {
	c := &client{}

	for _, opt := range opts {
		opt(c)
	}

	if c.modelName == "" {
		return nil, fmt.Errorf("WithModel is a required option")
	}

	if apiKey == "" {
		return nil, fmt.Errorf("API key is required for Gemini API")
	}

	ctx := context.Background()
	genaiClient, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	c.genaiClient = genaiClient

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
	{Model: "gemini-2.5-pro", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 65536}},
	{Model: "gemini-2.5-flash", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 65536}},
	{Model: "gemini-2.5-flash-lite", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 65536}},
	{Model: "gemini-2.0-flash", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
	{Model: "gemini-2.0-flash-lite", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
	{Model: "gemini-1.5-pro", TokenLimits: chat.TokenLimits{Context: 2097152, Output: 8192}},
	{Model: "gemini-1.5-flash", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
	{Model: "gemini-1.5-flash-8b", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
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
	cumulativeUsage chat.TokenUsage
	maxTokens       int

	// Tool support
	tools     map[string]common.RegisteredTool
	toolsLock sync.RWMutex
}

func (c *chatClient) MessageStream(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	reqMsg := msg
	reqOpts := chat.ApplyOptions(opts...)

	c.mu.Lock()
	defer c.mu.Unlock()

	// Build content for all messages
	var contents []*genai.Content

	// Add system instruction as first content if present
	if c.systemPrompt != "" {
		systemText := c.systemPrompt
		// Handle response format if provided
		if reqOpts.ResponseFormat != nil && reqOpts.ResponseFormat.Schema != nil {
			systemText += fmt.Sprintf("\n\nYou must respond with valid JSON that conforms to the schema named: %s", reqOpts.ResponseFormat.Name)
		}
		contents = append(contents, &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{Text: systemText},
			},
		})
		// Add a placeholder assistant response to maintain conversation flow
		contents = append(contents, &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "I understand and will follow these instructions."},
			},
		})
	}

	// Add history messages
	for _, m := range c.msgs {
		var role string
		switch m.Role {
		case chat.UserRole:
			role = "user"
		case chat.AssistantRole:
			role = "model"
		default:
			// Skip system messages as they're handled separately
			continue
		}

		contents = append(contents, &genai.Content{
			Role: role,
			Parts: []*genai.Part{
				{Text: m.Content},
			},
		})
	}

	// Add current message
	var currentRole string
	switch msg.Role {
	case chat.UserRole:
		currentRole = "user"
	case chat.AssistantRole:
		currentRole = "model"
	default:
		currentRole = "user"
	}

	contents = append(contents, &genai.Content{
		Role: currentRole,
		Parts: []*genai.Part{
			{Text: msg.Content},
		},
	})

	// Configure generation settings
	config := &genai.GenerateContentConfig{}

	if reqOpts.Temperature != nil {
		temp := float32(*reqOpts.Temperature)
		config.Temperature = &temp
	}

	if reqOpts.MaxTokens > 0 {
		config.MaxOutputTokens = int32(reqOpts.MaxTokens)
	}

	// Add tools if registered
	var toolConversionErr error
	func() {
		c.toolsLock.RLock()
		defer c.toolsLock.RUnlock()

		if len(c.tools) > 0 {
			tools := make([]*genai.Tool, 0, 1)
			functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(c.tools))
			for _, tool := range c.tools {
				funcDecl, err := c.mcpToGeminiFunctionDeclaration(tool.Definition)
				if err != nil {
					toolConversionErr = fmt.Errorf("failed to convert tool: %w", err)
					return
				}
				functionDeclarations = append(functionDeclarations, funcDecl)
			}
			// Create a single Tool with all function declarations
			tools = append(tools, &genai.Tool{
				FunctionDeclarations: functionDeclarations,
			})
			config.Tools = tools
		}
	}()

	if toolConversionErr != nil {
		return chat.Message{}, toolConversionErr
	}

	// Stream content
	stream := c.genaiClient.Models.GenerateContentStream(ctx, c.modelName, contents, config)

	var respContent strings.Builder
	var functionCalls []*genai.FunctionCall
	for chunk, err := range stream {
		if err != nil {
			return chat.Message{}, fmt.Errorf("streaming error: %w", err)
		}

		// Extract text and function calls from chunk
		for _, candidate := range chunk.Candidates {
			if candidate.Content != nil {
				for _, part := range candidate.Content.Parts {
					if part.Text != "" {
						content := part.Text
						respContent.WriteString(content)

						// Call the callback with the content event
						if callback != nil {
							event := chat.StreamEvent{
								Type:    chat.StreamEventTypeContent,
								Content: content,
							}
							if err := callback(event); err != nil {
								// User requested to stop streaming
								return chat.Message{}, err
							}
						}
					}
					// Check for function calls
					if part.FunctionCall != nil {
						functionCalls = append(functionCalls, part.FunctionCall)

						// Emit tool call event
						if callback != nil {
							// Convert arguments to JSON
							argsJSON, _ := json.Marshal(part.FunctionCall.Args)
							toolCallEvent := chat.StreamEvent{
								Type: chat.StreamEventTypeToolCall,
								ToolCalls: []chat.ToolCall{
									{
										ID:        part.FunctionCall.ID,
										Name:      part.FunctionCall.Name,
										Arguments: json.RawMessage(argsJSON),
									},
								},
							}
							if err := callback(toolCallEvent); err != nil {
								return chat.Message{}, err
							}
						}
					}
				}
			}
			// Extract token usage if available
			if chunk.UsageMetadata != nil {
				usage := chat.TokenUsage{
					InputTokens:  int(chunk.UsageMetadata.PromptTokenCount),
					OutputTokens: int(chunk.UsageMetadata.CandidatesTokenCount),
					TotalTokens:  int(chunk.UsageMetadata.TotalTokenCount),
					CachedTokens: int(chunk.UsageMetadata.CachedContentTokenCount),
				}
				c.cumulativeUsage.InputTokens += usage.InputTokens
				c.cumulativeUsage.OutputTokens += usage.OutputTokens
				c.cumulativeUsage.TotalTokens += usage.TotalTokens
				c.cumulativeUsage.CachedTokens += usage.CachedTokens
			}
		}
	}

	// Handle tool calls with multiple rounds if needed
	if len(functionCalls) > 0 {
		return c.handleToolCallRounds(ctx, reqMsg, functionCalls, reqOpts, callback)
	}

	respMsg := chat.Message{
		Role:    chat.AssistantRole,
		Content: respContent.String(),
	}

	// Update history
	c.msgs = append(c.msgs, reqMsg)
	c.msgs = append(c.msgs, respMsg)

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

// TokenUsage returns the token usage for the last message exchange
func (c *chatClient) TokenUsage() (chat.TokenUsage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cumulativeUsage, nil
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

// mcpToGeminiFunctionDeclaration converts an MCP tool definition to Gemini FunctionDeclaration format
func (c *chatClient) mcpToGeminiFunctionDeclaration(mcpDef chat.ToolDef) (*genai.FunctionDeclaration, error) {
	// Parse the MCP JSON schema to extract the inputSchema
	var mcp struct {
		InputSchema json.RawMessage `json:"inputSchema"`
	}

	jsonSchema := mcpDef.MCPJsonSchema()
	if err := json.Unmarshal([]byte(jsonSchema), &mcp); err != nil {
		return nil, fmt.Errorf("failed to parse MCP definition: %w", err)
	}

	// Convert inputSchema to Gemini Schema format
	var parameters *genai.Schema
	if len(mcp.InputSchema) > 0 {
		// Parse the JSON schema into a map first
		var schemaMap map[string]interface{}
		if err := json.Unmarshal(mcp.InputSchema, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse input schema: %w", err)
		}

		// Create a Gemini Schema from the parsed JSON schema
		parameters = &genai.Schema{
			Type:       genai.TypeObject, // MCP tools typically use object schemas
			Properties: make(map[string]*genai.Schema),
		}

		// Extract properties if they exist
		if props, ok := schemaMap["properties"].(map[string]interface{}); ok {
			for propName, propSchema := range props {
				if propMap, ok := propSchema.(map[string]interface{}); ok {
					geminiProp := &genai.Schema{}

					// Convert basic type
					if typeStr, ok := propMap["type"].(string); ok {
						switch typeStr {
						case "string":
							geminiProp.Type = genai.TypeString
						case "integer":
							geminiProp.Type = genai.TypeInteger
						case "number":
							geminiProp.Type = genai.TypeNumber
						case "boolean":
							geminiProp.Type = genai.TypeBoolean
						case "array":
							geminiProp.Type = genai.TypeArray
						case "object":
							geminiProp.Type = genai.TypeObject
						}
					}

					// Add description if available
					if desc, ok := propMap["description"].(string); ok {
						geminiProp.Description = desc
					}

					parameters.Properties[propName] = geminiProp
				}
			}
		}

		// Extract required fields if they exist
		if required, ok := schemaMap["required"].([]interface{}); ok {
			requiredFields := make([]string, 0, len(required))
			for _, field := range required {
				if fieldName, ok := field.(string); ok {
					requiredFields = append(requiredFields, fieldName)
				}
			}
			parameters.Required = requiredFields
		}
	}

	return &genai.FunctionDeclaration{
		Name:        mcpDef.Name(),
		Description: mcpDef.Description(),
		Parameters:  parameters,
	}, nil
}

// handleToolCallRounds handles potentially multiple rounds of tool calls
func (c *chatClient) handleToolCallRounds(ctx context.Context, initialMsg chat.Message, initialFunctionCalls []*genai.FunctionCall, reqOpts chat.Options, callback chat.StreamCallback) (chat.Message, error) {
	// Keep track of all messages for the conversation
	var conversationContents []*genai.Content

	// Build initial conversation with system prompt and history
	if c.systemPrompt != "" {
		conversationContents = append(conversationContents, &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{Text: c.systemPrompt},
			},
		})
		// Add a placeholder assistant response to maintain conversation flow
		conversationContents = append(conversationContents, &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "I understand and will follow these instructions."},
			},
		})
	}

	// Add history messages
	for _, m := range c.msgs {
		var role string
		switch m.Role {
		case chat.UserRole:
			role = "user"
		case chat.AssistantRole:
			role = "model"
		default:
			continue
		}

		conversationContents = append(conversationContents, &genai.Content{
			Role: role,
			Parts: []*genai.Part{
				{Text: m.Content},
			},
		})
	}

	// Add the initial user message
	c.msgs = append(c.msgs, initialMsg)
	conversationContents = append(conversationContents, &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: initialMsg.Content},
		},
	})

	// Process tool calls in a loop until we get a final response
	functionCalls := initialFunctionCalls
	maxRounds := 10 // Prevent infinite loops

	for round := 0; round < maxRounds && len(functionCalls) > 0; round++ {
		// Execute tool calls
		functionResults, err := c.handleFunctionCalls(ctx, functionCalls)
		if err != nil {
			return chat.Message{}, fmt.Errorf("failed to execute function calls: %w", err)
		}

		// Add assistant message with function calls
		assistantParts := make([]*genai.Part, len(functionCalls))
		for i, fc := range functionCalls {
			assistantParts[i] = &genai.Part{
				FunctionCall: fc,
			}
		}

		conversationContents = append(conversationContents, &genai.Content{
			Role:  "model",
			Parts: assistantParts,
		})

		// Add function results to messages
		resultParts := make([]*genai.Part, len(functionResults))
		for i, fr := range functionResults {
			resultParts[i] = &genai.Part{
				FunctionResponse: fr,
			}
		}

		conversationContents = append(conversationContents, &genai.Content{
			Role:  "function",
			Parts: resultParts,
		})

		// Make another API call with tool results
		followUpConfig := &genai.GenerateContentConfig{}
		if reqOpts.Temperature != nil {
			temp := float32(*reqOpts.Temperature)
			followUpConfig.Temperature = &temp
		}
		if reqOpts.MaxTokens > 0 {
			followUpConfig.MaxOutputTokens = int32(reqOpts.MaxTokens)
		}

		// Add tools again for follow-up after tool execution
		func() {
			c.toolsLock.RLock()
			defer c.toolsLock.RUnlock()

			if len(c.tools) > 0 {
				tools := make([]*genai.Tool, 0, 1)
				functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(c.tools))
				for _, tool := range c.tools {
					funcDecl, err := c.mcpToGeminiFunctionDeclaration(tool.Definition)
					if err != nil {
						// Skip this tool on error
						continue
					}
					functionDeclarations = append(functionDeclarations, funcDecl)
				}
				// Create a single Tool with all function declarations
				tools = append(tools, &genai.Tool{
					FunctionDeclarations: functionDeclarations,
				})
				followUpConfig.Tools = tools
			}
		}()

		// Create a new stream for the follow-up request
		followUpStream := c.genaiClient.Models.GenerateContentStream(ctx, c.modelName, conversationContents, followUpConfig)

		// Process the follow-up stream
		var respContent strings.Builder
		functionCalls = nil // Reset for next round

		for chunk, err := range followUpStream {
			if err != nil {
				return chat.Message{}, fmt.Errorf("follow-up streaming error: %w", err)
			}

			for _, candidate := range chunk.Candidates {
				if candidate.Content != nil {
					for _, part := range candidate.Content.Parts {
						// Check for function calls
						if part.FunctionCall != nil {
							functionCalls = append(functionCalls, part.FunctionCall)

							// Emit tool call event
							if callback != nil {
								// Convert arguments to JSON
								argsJSON, _ := json.Marshal(part.FunctionCall.Args)
								toolCallEvent := chat.StreamEvent{
									Type: chat.StreamEventTypeToolCall,
									ToolCalls: []chat.ToolCall{
										{
											ID:        part.FunctionCall.ID,
											Name:      part.FunctionCall.Name,
											Arguments: json.RawMessage(argsJSON),
										},
									},
								}
								if err := callback(toolCallEvent); err != nil {
									return chat.Message{}, err
								}
							}
						}

						// Check for regular content
						if part.Text != "" {
							content := part.Text
							respContent.WriteString(content)

							// Call the callback with the content event
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
				}
				// Extract token usage if available
				if chunk.UsageMetadata != nil {
					usage := chat.TokenUsage{
						InputTokens:  int(chunk.UsageMetadata.PromptTokenCount),
						OutputTokens: int(chunk.UsageMetadata.CandidatesTokenCount),
						TotalTokens:  int(chunk.UsageMetadata.TotalTokenCount),
						CachedTokens: int(chunk.UsageMetadata.CachedContentTokenCount),
					}
					c.cumulativeUsage.InputTokens += usage.InputTokens
					c.cumulativeUsage.OutputTokens += usage.OutputTokens
					c.cumulativeUsage.TotalTokens += usage.TotalTokens
					c.cumulativeUsage.CachedTokens += usage.CachedTokens
				}
			}
		}

		// If we got more function calls, continue the loop
		if len(functionCalls) > 0 {
			continue
		}

		// No more function calls, we have the final response
		finalMsg := chat.Message{
			Role:    chat.AssistantRole,
			Content: respContent.String(),
		}

		// Update history with the final response
		c.msgs = append(c.msgs, finalMsg)

		return finalMsg, nil
	}

	// If we get here, we exceeded maxRounds
	return chat.Message{}, fmt.Errorf("exceeded maximum rounds of function calls (%d)", maxRounds)
}

// handleFunctionCalls processes function calls from the model and returns function results
func (c *chatClient) handleFunctionCalls(ctx context.Context, functionCalls []*genai.FunctionCall) ([]*genai.FunctionResponse, error) {
	if len(functionCalls) == 0 {
		return nil, nil
	}

	var functionResults []*genai.FunctionResponse

	c.toolsLock.RLock()
	defer c.toolsLock.RUnlock()

	for _, fc := range functionCalls {
		tool, exists := c.tools[fc.Name]
		if !exists {
			// Tool not found, return error message
			errorResponse := map[string]interface{}{
				"error": fmt.Sprintf("Tool %s not found", fc.Name),
			}
			functionResults = append(functionResults, &genai.FunctionResponse{
				ID:       fc.ID,
				Name:     fc.Name,
				Response: errorResponse,
			})
			continue
		}

		// Convert function arguments to JSON string for the handler
		argsJSON, err := json.Marshal(fc.Args)
		if err != nil {
			errorResponse := map[string]interface{}{
				"error": fmt.Sprintf("Failed to marshal function arguments: %v", err),
			}
			functionResults = append(functionResults, &genai.FunctionResponse{
				ID:       fc.ID,
				Name:     fc.Name,
				Response: errorResponse,
			})
			continue
		}

		// Execute the tool
		resultStr := tool.Handler(ctx, string(argsJSON))

		// Parse the result back into a map for the response
		var resultMap map[string]interface{}
		if err := json.Unmarshal([]byte(resultStr), &resultMap); err != nil {
			// If parsing fails, treat the result as a simple string response
			resultMap = map[string]interface{}{
				"result": resultStr,
			}
		}

		// Add function result
		functionResults = append(functionResults, &genai.FunctionResponse{
			ID:       fc.ID,
			Name:     fc.Name,
			Response: resultMap,
		})
	}

	return functionResults, nil
}
