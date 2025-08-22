package common

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolsPreservesRegistrationOrder(t *testing.T) {
	tools := NewTools()

	// Register tools in a specific order
	toolNames := []string{"zebra", "apple", "middle", "banana", "xylophone"}
	for _, name := range toolNames {
		def := mockToolDef{
			name:        name,
			description: "test tool " + name,
			schema:      `{"type": "object"}`,
		}
		handler := func(ctx context.Context, input string) string {
			return "result for " + name
		}
		err := tools.Register(def, handler)
		require.NoError(t, err)
	}

	// GetAll should return tools in registration order
	allTools := tools.GetAll()
	assert.Len(t, allTools, 5)
	for i, name := range toolNames {
		assert.Equal(t, name, allTools[i].Definition.Name(), "tool at index %d should be %s", i, name)
	}

	// List should return tool names in registration order, not alphabetical
	orderedNames := tools.List()
	assert.Equal(t, toolNames, orderedNames, "tools.List() should preserve registration order")
}

func TestToolsOrderAfterDeregistration(t *testing.T) {
	tools := NewTools()

	// Register tools
	toolNames := []string{"first", "second", "third", "fourth", "fifth"}
	for _, name := range toolNames {
		def := mockToolDef{
			name:        name,
			description: "test tool " + name,
			schema:      `{"type": "object"}`,
		}
		handler := func(ctx context.Context, input string) string {
			return "result for " + name
		}
		err := tools.Register(def, handler)
		require.NoError(t, err)
	}

	// Deregister a tool in the middle
	tools.Deregister("third")

	// Remaining tools should maintain their relative order
	expected := []string{"first", "second", "fourth", "fifth"}
	orderedNames := tools.List()
	assert.Equal(t, expected, orderedNames, "order should be preserved after deregistration")
}

func TestToolsReregistrationUpdatesInPlace(t *testing.T) {
	tools := NewTools()

	// Register initial tools
	toolNames := []string{"alpha", "beta", "gamma"}
	for _, name := range toolNames {
		def := mockToolDef{
			name:        name,
			description: "test tool " + name,
			schema:      `{"type": "object"}`,
		}
		nameCapture := name // Capture the name for the closure
		handler := func(ctx context.Context, input string) string {
			return "result v1 for " + nameCapture
		}
		err := tools.Register(def, handler)
		require.NoError(t, err)
	}

	// Re-register "beta" with a new handler
	newDef := mockToolDef{
		name:        "beta",
		description: "updated test tool beta",
		schema:      `{"type": "object"}`,
	}
	newHandler := func(ctx context.Context, input string) string {
		return "result v2 for beta"
	}
	err := tools.Register(newDef, newHandler)
	require.NoError(t, err)

	// Order should remain the same
	orderedNames := tools.List()
	assert.Equal(t, toolNames, orderedNames, "re-registration should not change order")

	// But the handler should be updated
	result, err := tools.Execute(context.Background(), "beta", "test input")
	require.NoError(t, err)
	assert.Equal(t, "result v2 for beta", result)
}
