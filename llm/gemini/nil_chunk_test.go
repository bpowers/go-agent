package gemini

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
)

// TestNilChunkHandling tests that we handle nil chunks from the genai library
// gracefully without panicking. This can happen when the iterator yields an error.
func TestNilChunkHandling(t *testing.T) {
	// This test attempts to reproduce the segfault where chunk is nil
	// when the genai library encounters an error.

	// Skip if no API key - we need a real client to test this
	if getAPIKey() == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	// Create a client with an invalid configuration that might cause errors
	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	chatSession := client.NewChat("You are a helpful assistant.")

	ctx := context.Background()

	// Try various operations that might cause the genai library to yield errors
	// with nil chunks

	// Test 1: Empty message (might cause an error)
	_, err = chatSession.Message(ctx, chat.Message{})
	// We expect this might error, but it should not panic
	assert.Error(t, err)

	// Test 2: Very long message that might exceed limits
	longContent := make([]byte, 10*1024*1024) // 10MB
	for i := range longContent {
		longContent[i] = 'a'
	}
	_, _ = chatSession.Message(ctx, chat.UserMessage(string(longContent)))
	// Should either succeed or fail gracefully, but not panic
	// We don't assert on error because it depends on the model's limits
}

// TestCancelledContext tests that a cancelled context doesn't cause a panic
func TestCancelledContext(t *testing.T) {
	if getAPIKey() == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	client, err := NewClient(getAPIKey(), WithModel(getTestModel()))
	require.NoError(t, err)
	require.NotNil(t, client)

	chatSession := client.NewChat("You are a helpful assistant.")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Try to send a message with cancelled context
	_, err = chatSession.Message(ctx, chat.UserMessage("Hello"))
	// Should error due to cancelled context, but not panic
	assert.Error(t, err)
}
