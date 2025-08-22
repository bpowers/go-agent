package common

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
)

func TestState_NewState(t *testing.T) {
	t.Parallel()

	t.Run("empty initial state", func(t *testing.T) {
		t.Parallel()
		s := NewState("system prompt", nil)
		require.NotNil(t, s)

		systemPrompt, msgs := s.History()
		assert.Equal(t, "system prompt", systemPrompt)
		assert.Empty(t, msgs)
	})

	t.Run("with initial messages", func(t *testing.T) {
		t.Parallel()
		initialMsgs := []chat.Message{
			{Role: chat.UserRole, Content: "Hello"},
			{Role: chat.AssistantRole, Content: "Hi there"},
		}

		s := NewState("system", initialMsgs)
		require.NotNil(t, s)

		systemPrompt, msgs := s.History()
		assert.Equal(t, "system", systemPrompt)
		assert.Equal(t, initialMsgs, msgs)
	})

	t.Run("copies initial messages", func(t *testing.T) {
		t.Parallel()
		initialMsgs := []chat.Message{
			{Role: chat.UserRole, Content: "Hello"},
		}

		s := NewState("system", initialMsgs)

		// Modify original slice
		initialMsgs[0].Content = "Modified"

		// State should have the original value
		_, msgs := s.History()
		assert.Equal(t, "Hello", msgs[0].Content)
	})
}

func TestState_Snapshot(t *testing.T) {
	t.Parallel()

	s := NewState("system prompt", []chat.Message{
		{Role: chat.UserRole, Content: "Hello"},
	})

	systemPrompt, msgs := s.Snapshot()
	assert.Equal(t, "system prompt", systemPrompt)
	assert.Len(t, msgs, 1)

	// Modifying snapshot shouldn't affect state
	msgs[0].Content = "Modified"

	_, originalMsgs := s.History()
	assert.Equal(t, "Hello", originalMsgs[0].Content)
}

func TestState_AppendMessages(t *testing.T) {
	t.Parallel()

	t.Run("append without usage", func(t *testing.T) {
		t.Parallel()
		s := NewState("system", nil)

		newMsgs := []chat.Message{
			{Role: chat.UserRole, Content: "Question"},
			{Role: chat.AssistantRole, Content: "Answer"},
		}

		s.AppendMessages(newMsgs, nil)

		_, msgs := s.History()
		assert.Equal(t, newMsgs, msgs)

		// Token usage should be zero
		usage, err := s.TokenUsage()
		require.NoError(t, err)
		assert.Equal(t, 0, usage.LastMessage.TotalTokens)
		assert.Equal(t, 0, usage.Cumulative.TotalTokens)
	})

	t.Run("append with usage", func(t *testing.T) {
		t.Parallel()
		s := NewState("system", nil)

		usage1 := &chat.TokenUsageDetails{
			InputTokens:  10,
			OutputTokens: 20,
			TotalTokens:  30,
			CachedTokens: 5,
		}

		s.AppendMessages([]chat.Message{
			{Role: chat.UserRole, Content: "Q1"},
		}, usage1)

		tokenUsage, err := s.TokenUsage()
		require.NoError(t, err)
		assert.Equal(t, 30, tokenUsage.LastMessage.TotalTokens)
		assert.Equal(t, 30, tokenUsage.Cumulative.TotalTokens)
		assert.Equal(t, 5, tokenUsage.Cumulative.CachedTokens)

		// Append more with usage
		usage2 := &chat.TokenUsageDetails{
			InputTokens:  15,
			OutputTokens: 25,
			TotalTokens:  40,
			CachedTokens: 10,
		}

		s.AppendMessages([]chat.Message{
			{Role: chat.AssistantRole, Content: "A1"},
		}, usage2)

		tokenUsage, err = s.TokenUsage()
		require.NoError(t, err)
		assert.Equal(t, 40, tokenUsage.LastMessage.TotalTokens)
		assert.Equal(t, 70, tokenUsage.Cumulative.TotalTokens)  // 30 + 40
		assert.Equal(t, 15, tokenUsage.Cumulative.CachedTokens) // 5 + 10
	})

	t.Run("ignore zero token usage", func(t *testing.T) {
		t.Parallel()
		s := NewState("system", nil)

		usage1 := &chat.TokenUsageDetails{
			InputTokens:  10,
			OutputTokens: 20,
			TotalTokens:  30,
		}

		s.AppendMessages([]chat.Message{{Role: chat.UserRole, Content: "Q"}}, usage1)

		// Append with zero usage
		zeroUsage := &chat.TokenUsageDetails{
			InputTokens:  0,
			OutputTokens: 0,
			TotalTokens:  0,
		}

		s.AppendMessages([]chat.Message{{Role: chat.AssistantRole, Content: "A"}}, zeroUsage)

		tokenUsage, err := s.TokenUsage()
		require.NoError(t, err)
		assert.Equal(t, 30, tokenUsage.LastMessage.TotalTokens) // Should still be 30
		assert.Equal(t, 30, tokenUsage.Cumulative.TotalTokens)
	})
}

func TestState_UpdateUsage(t *testing.T) {
	t.Parallel()

	s := NewState("system", nil)

	usage1 := chat.TokenUsageDetails{
		InputTokens:  10,
		OutputTokens: 20,
		TotalTokens:  30,
	}

	s.UpdateUsage(usage1)

	tokenUsage, err := s.TokenUsage()
	require.NoError(t, err)
	assert.Equal(t, 30, tokenUsage.LastMessage.TotalTokens)
	assert.Equal(t, 30, tokenUsage.Cumulative.TotalTokens)

	// Update again
	usage2 := chat.TokenUsageDetails{
		InputTokens:  5,
		OutputTokens: 10,
		TotalTokens:  15,
	}

	s.UpdateUsage(usage2)

	tokenUsage, err = s.TokenUsage()
	require.NoError(t, err)
	assert.Equal(t, 15, tokenUsage.LastMessage.TotalTokens) // Last message updated
	assert.Equal(t, 45, tokenUsage.Cumulative.TotalTokens)  // Cumulative: 30 + 15

	// Zero usage should be ignored
	s.UpdateUsage(chat.TokenUsageDetails{})

	tokenUsage, err = s.TokenUsage()
	require.NoError(t, err)
	assert.Equal(t, 15, tokenUsage.LastMessage.TotalTokens) // Unchanged
	assert.Equal(t, 45, tokenUsage.Cumulative.TotalTokens)  // Unchanged
}

func TestState_Concurrency(t *testing.T) {
	t.Parallel()

	t.Run("concurrent appends", func(t *testing.T) {
		t.Parallel()
		s := NewState("system", nil)

		const numGoroutines = 100
		const msgsPerGoroutine = 10

		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(id int) {
				defer wg.Done()
				for j := 0; j < msgsPerGoroutine; j++ {
					msg := chat.Message{
						Role:    chat.UserRole,
						Content: "msg",
					}
					s.AppendMessages([]chat.Message{msg}, nil)
				}
			}(i)
		}

		wg.Wait()

		_, msgs := s.History()
		assert.Len(t, msgs, numGoroutines*msgsPerGoroutine)
	})

	t.Run("concurrent reads and writes", func(t *testing.T) {
		t.Parallel()
		s := NewState("system", []chat.Message{
			{Role: chat.UserRole, Content: "initial"},
		})

		const numReaders = 50
		const numWriters = 50
		const iterations = 100

		var wg sync.WaitGroup

		// Start readers
		wg.Add(numReaders)
		for i := 0; i < numReaders; i++ {
			go func() {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					// These should not panic or race
					s.Snapshot()
					s.History()
					s.TokenUsage()
				}
			}()
		}

		// Start writers
		wg.Add(numWriters)
		for i := 0; i < numWriters; i++ {
			go func(id int) {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					msg := chat.Message{
						Role:    chat.AssistantRole,
						Content: "response",
					}
					usage := &chat.TokenUsageDetails{
						InputTokens:  1,
						OutputTokens: 1,
						TotalTokens:  2,
					}
					if j%2 == 0 {
						s.AppendMessages([]chat.Message{msg}, usage)
					} else {
						s.UpdateUsage(*usage)
					}
				}
			}(i)
		}

		wg.Wait()

		// Verify state is consistent
		_, msgs := s.History()
		assert.NotEmpty(t, msgs)

		usage, err := s.TokenUsage()
		require.NoError(t, err)
		assert.Greater(t, usage.Cumulative.TotalTokens, 0)
	})

	t.Run("concurrent usage updates", func(t *testing.T) {
		t.Parallel()
		s := NewState("system", nil)

		const numGoroutines = 100
		const updatesPerGoroutine = 100

		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func() {
				defer wg.Done()
				for j := 0; j < updatesPerGoroutine; j++ {
					s.UpdateUsage(chat.TokenUsageDetails{
						InputTokens:  1,
						OutputTokens: 1,
						TotalTokens:  2,
					})
				}
			}()
		}

		wg.Wait()

		usage, err := s.TokenUsage()
		require.NoError(t, err)
		// Each goroutine adds 2 tokens per update
		expectedTotal := numGoroutines * updatesPerGoroutine * 2
		assert.Equal(t, expectedTotal, usage.Cumulative.TotalTokens)
	})
}

func TestState_History(t *testing.T) {
	t.Parallel()

	initialMsgs := []chat.Message{
		{Role: chat.UserRole, Content: "Hello"},
		{Role: chat.AssistantRole, Content: "Hi"},
	}

	s := NewState("system", initialMsgs)

	systemPrompt, msgs := s.History()
	assert.Equal(t, "system", systemPrompt)
	assert.Equal(t, initialMsgs, msgs)

	// Modifying returned messages shouldn't affect internal state
	msgs[0].Content = "Modified"

	_, originalMsgs := s.History()
	assert.Equal(t, "Hello", originalMsgs[0].Content)
}

func TestState_TokenUsage(t *testing.T) {
	t.Parallel()

	s := NewState("system", nil)

	// Initial usage should be zero
	usage, err := s.TokenUsage()
	require.NoError(t, err)
	assert.Equal(t, 0, usage.LastMessage.TotalTokens)
	assert.Equal(t, 0, usage.Cumulative.TotalTokens)

	// Update usage
	s.UpdateUsage(chat.TokenUsageDetails{
		InputTokens:  100,
		OutputTokens: 200,
		TotalTokens:  300,
		CachedTokens: 50,
	})

	usage, err = s.TokenUsage()
	require.NoError(t, err)
	assert.Equal(t, 100, usage.LastMessage.InputTokens)
	assert.Equal(t, 200, usage.LastMessage.OutputTokens)
	assert.Equal(t, 300, usage.LastMessage.TotalTokens)
	assert.Equal(t, 50, usage.LastMessage.CachedTokens)

	assert.Equal(t, 100, usage.Cumulative.InputTokens)
	assert.Equal(t, 200, usage.Cumulative.OutputTokens)
	assert.Equal(t, 300, usage.Cumulative.TotalTokens)
	assert.Equal(t, 50, usage.Cumulative.CachedTokens)
}
