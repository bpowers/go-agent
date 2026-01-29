package mcp

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/bpowers/go-agent/chat"
)

// Registry holds a collection of tools that can be exposed via an MCP server.
// It is safe for concurrent use; tools can be registered while the server is running.
type Registry struct {
	mu          sync.Mutex
	tools       map[string]chat.Tool
	definitions map[string]ToolDefinition
	order       []string
}

// NewRegistry creates an empty tool registry.
func NewRegistry() *Registry {
	return &Registry{
		tools:       make(map[string]chat.Tool),
		definitions: make(map[string]ToolDefinition),
		order:       make([]string, 0),
	}
}

// Register adds a tool to the registry. The tool's MCPJsonSchema method is called
// to extract its name and schema. If a tool with the same name already exists,
// it is replaced. Returns an error if the tool is nil or has an invalid schema.
func (r *Registry) Register(tool chat.Tool) error {
	if tool == nil {
		return fmt.Errorf("register tool: nil tool")
	}

	definition, err := toolDefinition(tool)
	if err != nil {
		return fmt.Errorf("register tool: %w", err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[definition.Name]; !exists {
		r.order = append(r.order, definition.Name)
	}

	r.tools[definition.Name] = tool
	r.definitions[definition.Name] = definition
	return nil
}

// Get retrieves a tool by name. Returns the tool and true if found,
// or nil and false if no tool with that name is registered.
func (r *Registry) Get(name string) (chat.Tool, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	tool, ok := r.tools[name]
	return tool, ok
}

// Definitions returns the tool definitions for all registered tools
// in the order they were first registered. This is used by tools/list.
func (r *Registry) Definitions() []ToolDefinition {
	r.mu.Lock()
	defer r.mu.Unlock()

	defs := make([]ToolDefinition, 0, len(r.order))
	for _, name := range r.order {
		if def, ok := r.definitions[name]; ok {
			defs = append(defs, def)
		}
	}
	return defs
}

func toolDefinition(tool chat.Tool) (ToolDefinition, error) {
	var schema struct {
		Name         string          `json:"name"`
		Description  string          `json:"description"`
		InputSchema  json.RawMessage `json:"inputSchema"`
		OutputSchema json.RawMessage `json:"outputSchema"`
	}

	if err := json.Unmarshal([]byte(tool.MCPJsonSchema()), &schema); err != nil {
		return ToolDefinition{}, fmt.Errorf("parse MCPJsonSchema: %w", err)
	}
	if schema.Name == "" {
		return ToolDefinition{}, fmt.Errorf("missing tool name")
	}
	if len(schema.InputSchema) == 0 {
		return ToolDefinition{}, fmt.Errorf("missing input schema for %q", schema.Name)
	}

	return ToolDefinition{
		Name:         schema.Name,
		Description:  schema.Description,
		InputSchema:  schema.InputSchema,
		OutputSchema: schema.OutputSchema,
	}, nil
}
