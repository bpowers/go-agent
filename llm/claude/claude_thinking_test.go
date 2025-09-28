package claude

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSupportsThinking(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		// Models that support thinking
		{"opus-4-1 exact", "claude-opus-4-1", true},
		{"opus-4 exact", "claude-opus-4", true},
		{"sonnet-4 exact", "claude-sonnet-4", true},
		{"3-7-sonnet exact", "claude-3-7-sonnet", true},
		{"3-5-sonnet exact", "claude-3-5-sonnet", true},

		// Versioned models (with dates)
		{"opus-4-1 versioned", "claude-opus-4-1-20250805", true},
		{"opus-4 versioned", "claude-opus-4-20241029", true},
		{"sonnet-4 versioned", "claude-sonnet-4-20241022", true},
		{"3-7-sonnet versioned", "claude-3-7-sonnet-20241029", true},
		{"3-5-sonnet versioned", "claude-3-5-sonnet-20241022", true},

		// Models that don't support thinking
		{"haiku exact", "claude-3-haiku", false},
		{"3-5-haiku exact", "claude-3-5-haiku", false},
		{"haiku versioned", "claude-3-haiku-20241022", false},
		{"3-5-haiku versioned", "claude-3-5-haiku-latest", false},

		// Non-existent models
		{"non-existent", "claude-nonexistent", false},
		{"gpt model", "gpt-4", false},
		{"empty", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := supportsThinking(tt.model)
			assert.Equal(t, tt.expected, result, "Model %s: expected %v but got %v", tt.model, tt.expected, result)
		})
	}
}

func TestGenericSet(t *testing.T) {
	t.Run("string set operations", func(t *testing.T) {
		s := newSet("a", "b", "c")

		assert.True(t, s.contains("a"), "set should contain 'a'")
		assert.True(t, s.contains("b"), "set should contain 'b'")
		assert.True(t, s.contains("c"), "set should contain 'c'")
		assert.False(t, s.contains("d"), "set should not contain 'd'")
	})

	t.Run("containsWithPredicate", func(t *testing.T) {
		s := newSet("apple", "banana", "cherry")

		// Predicate that checks if any item starts with 'b'
		hasB := s.containsWithPredicate(func(item string) bool {
			return item[0] == 'b'
		})
		assert.True(t, hasB, "set should have item starting with 'b'")

		// Predicate that checks if any item starts with 'z'
		hasZ := s.containsWithPredicate(func(item string) bool {
			return item[0] == 'z'
		})
		assert.False(t, hasZ, "set should not have item starting with 'z'")
	})

	t.Run("integer set operations", func(t *testing.T) {
		s := newSet(1, 2, 3)

		assert.True(t, s.contains(1))
		assert.True(t, s.contains(2))
		assert.True(t, s.contains(3))
		assert.False(t, s.contains(4))
	})
}
