package chat

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWithSystemReminder(t *testing.T) {
	t.Run("adds reminder function to context", func(t *testing.T) {
		t.Parallel()
		ctx := context.Background()
		called := false
		reminderFunc := func() string {
			called = true
			return "test reminder"
		}

		ctx = WithSystemReminder(ctx, reminderFunc)

		retrievedFunc := GetSystemReminder(ctx)
		assert.NotNil(t, retrievedFunc)
		result := retrievedFunc()
		assert.True(t, called)
		assert.Equal(t, "test reminder", result)
	})

	t.Run("handles nil reminder function", func(t *testing.T) {
		t.Parallel()
		ctx := context.Background()
		ctx = WithSystemReminder(ctx, nil)

		retrievedFunc := GetSystemReminder(ctx)
		assert.Nil(t, retrievedFunc)
	})

	t.Run("returns nil when no reminder set", func(t *testing.T) {
		t.Parallel()
		ctx := context.Background()

		retrievedFunc := GetSystemReminder(ctx)
		assert.Nil(t, retrievedFunc)
	})

	t.Run("preserves context chain", func(t *testing.T) {
		t.Parallel()
		type testKey struct{}
		ctx := context.Background()
		ctx = context.WithValue(ctx, testKey{}, "test-value")

		reminderFunc := func() string { return "reminder" }
		ctx = WithSystemReminder(ctx, reminderFunc)

		// Original context value should still be accessible
		assert.Equal(t, "test-value", ctx.Value(testKey{}))

		// Reminder function should also be accessible
		retrievedFunc := GetSystemReminder(ctx)
		assert.NotNil(t, retrievedFunc)
		assert.Equal(t, "reminder", retrievedFunc())
	})

	t.Run("closure captures state", func(t *testing.T) {
		t.Parallel()
		counter := 0
		ctx := context.Background()

		ctx = WithSystemReminder(ctx, func() string {
			counter++
			return "called"
		})

		reminderFunc := GetSystemReminder(ctx)
		assert.NotNil(t, reminderFunc)

		// Call multiple times to verify state capture
		reminderFunc()
		reminderFunc()
		reminderFunc()

		assert.Equal(t, 3, counter)
	})
}
