package mcp

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/bpowers/go-agent/chat"
)

type Registry struct {
	mu          sync.Mutex
	tools       map[string]chat.Tool
	definitions map[string]ToolDefinition
	order       []string
}

func NewRegistry() *Registry {
	return &Registry{
		tools:       make(map[string]chat.Tool),
		definitions: make(map[string]ToolDefinition),
		order:       make([]string, 0),
	}
}

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

func (r *Registry) Get(name string) (chat.Tool, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	tool, ok := r.tools[name]
	return tool, ok
}

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
