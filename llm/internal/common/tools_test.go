package common

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
)

// mockToolDef implements chat.ToolDef for testing
type mockToolDef struct {
	name        string
	description string
	schema      string
}

func (m mockToolDef) Name() string {
	return m.name
}

func (m mockToolDef) Description() string {
	return m.description
}

func (m mockToolDef) MCPJsonSchema() string {
	return m.schema
}

// ensureToolDef is a helper function to ensure mockToolDef implements chat.ToolDef
func ensureToolDef(t chat.ToolDef) chat.ToolDef {
	return t
}

func TestTools_NewTools(t *testing.T) {
	t.Parallel()

	tools := NewTools()
	require.NotNil(t, tools)
	assert.Empty(t, tools.List())
	assert.Equal(t, 0, tools.Count())
}

func TestTools_Register(t *testing.T) {
	t.Parallel()

	t.Run("register valid tool", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		def := mockToolDef{
			name:        "test_tool",
			description: "A test tool",
			schema:      `{"type": "object"}`,
		}
		// Ensure mockToolDef implements chat.ToolDef
		_ = ensureToolDef(def)

		handler := func(ctx context.Context, input string) string {
			return "result: " + input
		}

		err := tools.Register(def, handler)
		require.NoError(t, err)

		assert.Equal(t, 1, tools.Count())
		assert.Equal(t, []string{"test_tool"}, tools.List())
	})

	t.Run("register tool with empty name", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		def := mockToolDef{
			name:        "",
			description: "A test tool",
			schema:      `{"type": "object"}`,
		}

		handler := func(ctx context.Context, input string) string {
			return "result"
		}

		err := tools.Register(def, handler)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "missing name")
	})

	t.Run("register multiple tools", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		for i := 0; i < 5; i++ {
			def := mockToolDef{
				name:        fmt.Sprintf("tool_%d", i),
				description: fmt.Sprintf("Tool %d", i),
				schema:      `{"type": "object"}`,
			}

			handler := func(ctx context.Context, input string) string {
				return "result"
			}

			err := tools.Register(def, handler)
			require.NoError(t, err)
		}

		assert.Equal(t, 5, tools.Count())
		list := tools.List()
		assert.Len(t, list, 5)
		// List should be sorted
		assert.Equal(t, []string{"tool_0", "tool_1", "tool_2", "tool_3", "tool_4"}, list)
	})

	t.Run("override existing tool", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		def1 := mockToolDef{
			name:        "tool",
			description: "First version",
			schema:      `{"v": 1}`,
		}

		handler1 := func(ctx context.Context, input string) string {
			return "v1"
		}

		err := tools.Register(def1, handler1)
		require.NoError(t, err)

		// Register with same name
		def2 := mockToolDef{
			name:        "tool",
			description: "Second version",
			schema:      `{"v": 2}`,
		}

		handler2 := func(ctx context.Context, input string) string {
			return "v2"
		}

		err = tools.Register(def2, handler2)
		require.NoError(t, err)

		assert.Equal(t, 1, tools.Count())

		// Verify it was overridden
		tool, exists := tools.Get("tool")
		require.True(t, exists)
		assert.Equal(t, "Second version", tool.Definition.Description())
		assert.Equal(t, "v2", tool.Handler(context.Background(), ""))
	})
}

func TestTools_Deregister(t *testing.T) {
	t.Parallel()

	tools := NewTools()

	def := mockToolDef{
		name:        "test_tool",
		description: "A test tool",
		schema:      `{"type": "object"}`,
	}

	handler := func(ctx context.Context, input string) string {
		return "result"
	}

	err := tools.Register(def, handler)
	require.NoError(t, err)
	assert.Equal(t, 1, tools.Count())

	// Deregister
	tools.Deregister("test_tool")
	assert.Equal(t, 0, tools.Count())

	// Deregister non-existent tool (should not panic)
	tools.Deregister("non_existent")
	assert.Equal(t, 0, tools.Count())
}

func TestTools_Get(t *testing.T) {
	t.Parallel()

	tools := NewTools()

	def := mockToolDef{
		name:        "test_tool",
		description: "A test tool",
		schema:      `{"type": "object"}`,
	}

	handler := func(ctx context.Context, input string) string {
		return "result: " + input
	}

	err := tools.Register(def, handler)
	require.NoError(t, err)

	// Get existing tool
	tool, exists := tools.Get("test_tool")
	assert.True(t, exists)
	assert.Equal(t, "test_tool", tool.Definition.Name())
	assert.Equal(t, "A test tool", tool.Definition.Description())
	assert.Equal(t, "result: test", tool.Handler(context.Background(), "test"))

	// Get non-existent tool
	_, exists = tools.Get("non_existent")
	assert.False(t, exists)
}

func TestTools_GetAll(t *testing.T) {
	t.Parallel()

	tools := NewTools()

	// Register multiple tools
	for i := 0; i < 3; i++ {
		def := mockToolDef{
			name:        fmt.Sprintf("tool_%d", i),
			description: fmt.Sprintf("Tool %d", i),
			schema:      `{"type": "object"}`,
		}

		handler := func(ctx context.Context, input string) string {
			return "result"
		}

		err := tools.Register(def, handler)
		require.NoError(t, err)
	}

	all := tools.GetAll()
	assert.Len(t, all, 3)

	// Verify all tools are present and in registration order
	for i := 0; i < 3; i++ {
		name := fmt.Sprintf("tool_%d", i)
		assert.Equal(t, name, all[i].Definition.Name())
	}

	// Modifying returned slice shouldn't affect internal state
	if len(all) > 0 {
		all[0] = RegisteredTool{} // Try to modify
	}
	newAll := tools.GetAll()
	assert.Equal(t, "tool_0", newAll[0].Definition.Name()) // Should still be original
	assert.Equal(t, 3, tools.Count())
}

func TestTools_List(t *testing.T) {
	t.Parallel()

	tools := NewTools()

	// Empty list
	assert.Empty(t, tools.List())

	// Register tools with unsorted names
	names := []string{"zebra", "alpha", "beta", "gamma"}
	for _, name := range names {
		def := mockToolDef{
			name:        name,
			description: "desc",
			schema:      `{}`,
		}
		handler := func(ctx context.Context, input string) string {
			return ""
		}
		err := tools.Register(def, handler)
		require.NoError(t, err)
	}

	// List should preserve registration order
	list := tools.List()
	assert.Equal(t, []string{"zebra", "alpha", "beta", "gamma"}, list)
}

func TestTools_Count(t *testing.T) {
	t.Parallel()

	tools := NewTools()
	assert.Equal(t, 0, tools.Count())

	// Register tools
	for i := 0; i < 5; i++ {
		def := mockToolDef{
			name:        fmt.Sprintf("tool_%d", i),
			description: "desc",
			schema:      `{}`,
		}
		handler := func(ctx context.Context, input string) string {
			return ""
		}
		err := tools.Register(def, handler)
		require.NoError(t, err)
		assert.Equal(t, i+1, tools.Count())
	}

	// Deregister one
	tools.Deregister("tool_2")
	assert.Equal(t, 4, tools.Count())
}

func TestTools_Execute(t *testing.T) {
	t.Parallel()

	t.Run("execute existing tool", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		def := mockToolDef{
			name:        "echo",
			description: "Echoes input",
			schema:      `{}`,
		}

		handler := func(ctx context.Context, input string) string {
			return "echo: " + input
		}

		err := tools.Register(def, handler)
		require.NoError(t, err)

		result, err := tools.Execute(context.Background(), "echo", "hello")
		require.NoError(t, err)
		assert.Equal(t, "echo: hello", result)
	})

	t.Run("execute non-existent tool", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		result, err := tools.Execute(context.Background(), "non_existent", "input")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
		assert.Empty(t, result)
	})

	t.Run("execute with context", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		def := mockToolDef{
			name:        "context_aware",
			description: "Uses context",
			schema:      `{}`,
		}

		handler := func(ctx context.Context, input string) string {
			select {
			case <-ctx.Done():
				return "cancelled"
			default:
				return "ok: " + input
			}
		}

		err := tools.Register(def, handler)
		require.NoError(t, err)

		// Normal execution
		result, err := tools.Execute(context.Background(), "context_aware", "test")
		require.NoError(t, err)
		assert.Equal(t, "ok: test", result)

		// With cancelled context
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		result, err = tools.Execute(ctx, "context_aware", "test")
		require.NoError(t, err)              // Execute itself doesn't fail
		assert.Equal(t, "cancelled", result) // But handler detects cancellation
	})
}

func TestTools_Concurrency(t *testing.T) {
	t.Parallel()

	t.Run("concurrent registrations", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		const numGoroutines = 100

		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(id int) {
				defer wg.Done()

				def := mockToolDef{
					name:        fmt.Sprintf("tool_%d", id),
					description: fmt.Sprintf("Tool %d", id),
					schema:      `{}`,
				}

				handler := func(ctx context.Context, input string) string {
					return fmt.Sprintf("result_%d", id)
				}

				err := tools.Register(def, handler)
				if err != nil {
					t.Errorf("Failed to register tool %d: %v", id, err)
				}
			}(i)
		}

		wg.Wait()

		assert.Equal(t, numGoroutines, tools.Count())
	})

	t.Run("concurrent reads and writes", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		// Pre-register some tools
		for i := 0; i < 10; i++ {
			def := mockToolDef{
				name:        fmt.Sprintf("tool_%d", i),
				description: "desc",
				schema:      `{}`,
			}
			handler := func(ctx context.Context, input string) string {
				return "result"
			}
			err := tools.Register(def, handler)
			require.NoError(t, err)
		}

		const numReaders = 50
		const numWriters = 20
		const iterations = 100

		var wg sync.WaitGroup

		// Start readers
		wg.Add(numReaders)
		for i := 0; i < numReaders; i++ {
			go func() {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					// These should not panic or race
					tools.List()
					tools.Count()
					tools.GetAll()
					tools.Get(fmt.Sprintf("tool_%d", j%10))
					tools.Execute(context.Background(), fmt.Sprintf("tool_%d", j%10), "input")
				}
			}()
		}

		// Start writers
		wg.Add(numWriters)
		for i := 0; i < numWriters; i++ {
			go func(id int) {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					if j%3 == 0 {
						// Register new tool
						def := mockToolDef{
							name:        fmt.Sprintf("dynamic_%d_%d", id, j),
							description: "dynamic",
							schema:      `{}`,
						}
						handler := func(ctx context.Context, input string) string {
							return "dynamic"
						}
						tools.Register(def, handler)
					} else if j%3 == 1 {
						// Deregister tool
						tools.Deregister(fmt.Sprintf("dynamic_%d_%d", id, j-1))
					} else {
						// Re-register existing tool
						def := mockToolDef{
							name:        fmt.Sprintf("tool_%d", j%10),
							description: "updated",
							schema:      `{}`,
						}
						handler := func(ctx context.Context, input string) string {
							return "updated"
						}
						tools.Register(def, handler)
					}
				}
			}(i)
		}

		wg.Wait()

		// Verify state is consistent
		count := tools.Count()
		list := tools.List()
		all := tools.GetAll()

		assert.Equal(t, len(list), count)
		assert.Equal(t, len(all), count)
	})

	t.Run("concurrent executions", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		var counter int64

		def := mockToolDef{
			name:        "counter",
			description: "Counts executions",
			schema:      `{}`,
		}

		handler := func(ctx context.Context, input string) string {
			atomic.AddInt64(&counter, 1)
			return "counted"
		}

		err := tools.Register(def, handler)
		require.NoError(t, err)

		const numGoroutines = 100
		const executionsPerGoroutine = 100

		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func() {
				defer wg.Done()
				for j := 0; j < executionsPerGoroutine; j++ {
					result, err := tools.Execute(context.Background(), "counter", "")
					if err != nil {
						t.Errorf("Execution failed: %v", err)
					}
					if result != "counted" {
						t.Errorf("Unexpected result: %s", result)
					}
				}
			}()
		}

		wg.Wait()

		expectedCount := int64(numGoroutines * executionsPerGoroutine)
		assert.Equal(t, expectedCount, atomic.LoadInt64(&counter))
	})

	t.Run("register and deregister race", func(t *testing.T) {
		t.Parallel()
		tools := NewTools()

		const iterations = 1000

		var wg sync.WaitGroup
		wg.Add(2)

		// Registerer
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				def := mockToolDef{
					name:        "race_tool",
					description: "racing",
					schema:      `{}`,
				}
				handler := func(ctx context.Context, input string) string {
					return "result"
				}
				tools.Register(def, handler)
			}
		}()

		// Deregisterer
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				tools.Deregister("race_tool")
			}
		}()

		wg.Wait()

		// Final state should be consistent
		_, exists := tools.Get("race_tool")
		count := tools.Count()
		if exists {
			assert.Equal(t, 1, count)
		} else {
			assert.Equal(t, 0, count)
		}
	})
}
