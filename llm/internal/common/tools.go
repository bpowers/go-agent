package common

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"github.com/bpowers/go-agent/chat"
)

// Tools manages tool registrations with thread-safe operations.
// This simple struct handles the common tool management pattern.
type Tools struct {
	mu    sync.RWMutex
	tools map[string]RegisteredTool // For fast lookups by name
	order []string                  // Preserves registration order
}

// NewTools creates a new tool manager.
func NewTools() *Tools {
	return &Tools{
		tools: make(map[string]RegisteredTool),
		order: make([]string, 0),
	}
}

// Register adds a tool to the registry.
func (t *Tools) Register(tool chat.Tool) error {
	toolName := tool.Name()
	if toolName == "" {
		return fmt.Errorf("tool definition missing name")
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	// Check if tool already exists to maintain order on re-registration
	if _, exists := t.tools[toolName]; !exists {
		// New tool, add to order
		t.order = append(t.order, toolName)
	}

	t.tools[toolName] = RegisteredTool{
		Tool: tool,
	}

	return nil
}

// Deregister removes a tool from the registry.
func (t *Tools) Deregister(name string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	delete(t.tools, name)

	// Remove from order slice
	for i, toolName := range t.order {
		if toolName == name {
			t.order = append(t.order[:i], t.order[i+1:]...)
			break
		}
	}
}

// Get retrieves a tool by name.
func (t *Tools) Get(name string) (RegisteredTool, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	tool, exists := t.tools[name]
	return tool, exists
}

// GetAll returns all registered tools in registration order.
func (t *Tools) GetAll() []RegisteredTool {
	t.mu.RLock()
	defer t.mu.RUnlock()

	result := make([]RegisteredTool, 0, len(t.order))
	for _, name := range t.order {
		if tool, exists := t.tools[name]; exists {
			result = append(result, tool)
		}
	}
	return result
}

// List returns tool names in registration order.
func (t *Tools) List() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	// Return a copy to prevent external modification
	return slices.Clone(t.order)
}

// Count returns the number of registered tools.
func (t *Tools) Count() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.tools)
}

// Execute runs a tool by name with the given context and input.
func (t *Tools) Execute(ctx context.Context, name string, input string) (string, error) {
	tool, exists := t.Get(name)
	if !exists {
		return "", fmt.Errorf("tool %q not found", name)
	}
	return tool.Tool.Call(ctx, input), nil
}
