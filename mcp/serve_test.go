package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/psanford/memfs"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/examples/fstools"
)

func TestServe(t *testing.T) {
	registry := NewRegistry()
	tool := &stubTool{
		name:        "Echo",
		description: "echoes input",
		schema:      `{"name":"Echo","description":"echoes input","inputSchema":{"type":"object","properties":{"msg":{"type":"string"}},"additionalProperties":false},"outputSchema":{"type":"object","properties":{"msg":{"type":"string"}},"additionalProperties":false}}`,
		result:      `{"msg":"hello"}`,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	// Build input with multiple JSON-RPC messages
	input := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"name":"test","version":"1.0"},"capabilities":{}}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":2,"method":"ping"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"Echo","arguments":{"msg":"hello"}}}`,
	}, "\n")

	in := strings.NewReader(input)
	out := &bytes.Buffer{}

	err = server.Serve(context.Background(), in, out)
	require.NoError(t, err)

	// Parse the output - should have 4 responses (no response for notification)
	decoder := json.NewDecoder(out)
	var responses []Response
	for decoder.More() {
		var resp Response
		require.NoError(t, decoder.Decode(&resp))
		responses = append(responses, resp)
	}

	require.Len(t, responses, 4)

	// Check initialize response
	assert.Equal(t, json.RawMessage("1"), responses[0].ID)
	assert.Nil(t, responses[0].Error)

	// Check ping response
	assert.Equal(t, json.RawMessage("2"), responses[1].ID)
	assert.Nil(t, responses[1].Error)

	// Check tools/list response
	assert.Equal(t, json.RawMessage("3"), responses[2].ID)
	assert.Nil(t, responses[2].Error)

	// Check tools/call response
	assert.Equal(t, json.RawMessage("4"), responses[3].ID)
	assert.Nil(t, responses[3].Error)
}

func TestServeParseError(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	in := strings.NewReader("not-json")
	out := &bytes.Buffer{}

	err = server.Serve(context.Background(), in, out)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode failed")

	// Should still have written an error response
	var resp Response
	require.NoError(t, json.Unmarshal(out.Bytes(), &resp))
	assert.Equal(t, errParse, resp.Error.Code)
}

func TestServeContextCancellation(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Use a reader that blocks forever (would hang without context cancellation)
	in := &blockingReader{}
	out := &bytes.Buffer{}

	err = server.Serve(ctx, in, out)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context canceled")
}

// blockingReader is a reader that never returns data, used to test context cancellation
type blockingReader struct{}

func (b *blockingReader) Read(p []byte) (n int, err error) {
	// Block forever - this simulates waiting for input that never comes
	select {}
}

func TestServeWithFSTools(t *testing.T) {
	// Create an in-memory filesystem
	mfs := memfs.New()
	require.NoError(t, mfs.WriteFile("test.txt", []byte("hello world"), 0o644))

	// Create context with filesystem
	ctx := fstools.WithFS(context.Background(), mfs)

	// Register fs tools
	registry := NewRegistry()
	require.NoError(t, registry.Register(fstools.ReadFileTool))
	require.NoError(t, registry.Register(fstools.WriteFileTool))
	require.NoError(t, registry.Register(fstools.ReadDirTool))

	server, err := NewServer(registry, Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	// Test reading the file we created
	input := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"name":"test","version":"1.0"},"capabilities":{}}}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ReadFile","arguments":{"fileName":"test.txt"}}}`,
		`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"WriteFile","arguments":{"fileName":"new.txt","content":"new content"}}}`,
		`{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"ReadFile","arguments":{"fileName":"new.txt"}}}`,
		`{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ReadDir","arguments":{"path":"."}}}`,
	}, "\n")

	in := strings.NewReader(input)
	out := &bytes.Buffer{}

	err = server.Serve(ctx, in, out)
	require.NoError(t, err)

	// Parse responses
	decoder := json.NewDecoder(out)
	var responses []Response
	for decoder.More() {
		var resp Response
		require.NoError(t, decoder.Decode(&resp))
		responses = append(responses, resp)
	}

	require.Len(t, responses, 6)

	// Check tools/list has all 3 tools
	listResult, ok := responses[1].Result.(map[string]any)
	require.True(t, ok)
	tools, ok := listResult["tools"].([]any)
	require.True(t, ok)
	assert.Len(t, tools, 3)

	// Check ReadFile result for test.txt
	readResult, ok := responses[2].Result.(map[string]any)
	require.True(t, ok)
	structured, ok := readResult["structuredContent"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "hello world", structured["content"])

	// Check WriteFile succeeded
	writeResult, ok := responses[3].Result.(map[string]any)
	require.True(t, ok)
	structured, ok = writeResult["structuredContent"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, structured["success"])

	// Check ReadFile result for new.txt (verifies round-trip)
	readNewResult, ok := responses[4].Result.(map[string]any)
	require.True(t, ok)
	structured, ok = readNewResult["structuredContent"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "new content", structured["content"])

	// Check ReadDir shows both files
	dirResult, ok := responses[5].Result.(map[string]any)
	require.True(t, ok)
	structured, ok = dirResult["structuredContent"].(map[string]any)
	require.True(t, ok)
	files, ok := structured["files"].([]any)
	require.True(t, ok)
	assert.Len(t, files, 2)
}

func TestServeWithFSToolsError(t *testing.T) {
	// Create an in-memory filesystem (empty)
	mfs := memfs.New()

	// Create context with filesystem
	ctx := fstools.WithFS(context.Background(), mfs)

	// Register fs tools
	registry := NewRegistry()
	require.NoError(t, registry.Register(fstools.ReadFileTool))

	server, err := NewServer(registry, Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	// Try to read a non-existent file
	input := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ReadFile","arguments":{"fileName":"nonexistent.txt"}}}`

	in := strings.NewReader(input)
	out := &bytes.Buffer{}

	err = server.Serve(ctx, in, out)
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(out.Bytes(), &resp))
	require.Nil(t, resp.Error) // JSON-RPC level should succeed

	// Check that the tool result indicates an error
	result, ok := resp.Result.(map[string]any)
	require.True(t, ok)
	assert.True(t, result["isError"].(bool))
	structured, ok := result["structuredContent"].(map[string]any)
	require.True(t, ok)
	assert.Contains(t, structured["error"], "nonexistent.txt")
}
