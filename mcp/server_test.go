package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewServerNilRegistry(t *testing.T) {
	_, err := NewServer(nil, Implementation{Name: "test", Version: "1.0"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "registry is required")
}

func TestNewServerEmptyName(t *testing.T) {
	_, err := NewServer(NewRegistry(), Implementation{Name: "", Version: "1.0"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "server name is required")
}

func TestNewServerEmptyVersion(t *testing.T) {
	_, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: ""})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "server version is required")
}

func TestNewServerWithInstructions(t *testing.T) {
	server, err := NewServer(
		NewRegistry(),
		Implementation{Name: "test", Version: "1.0"},
		WithInstructions("Use this server to do things"),
	)
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"name":"client","version":"1.0"},"capabilities":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)

	result, ok := resp.Result.(InitializeResult)
	require.True(t, ok)
	assert.Equal(t, "Use this server to do things", result.Instructions)
}

func TestNewServerWithProtocolVersion(t *testing.T) {
	server, err := NewServer(
		NewRegistry(),
		Implementation{Name: "test", Version: "1.0"},
		WithProtocolVersion("custom-2025"),
	)
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"name":"client","version":"1.0"},"capabilities":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)

	result, ok := resp.Result.(InitializeResult)
	require.True(t, ok)
	assert.Equal(t, "custom-2025", result.ProtocolVersion)
}

func TestNewServerWithEmptyProtocolVersion(t *testing.T) {
	_, err := NewServer(
		NewRegistry(),
		Implementation{Name: "test", Version: "1.0"},
		WithProtocolVersion(""),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "protocol version is required")
}

func TestServeNilServer(t *testing.T) {
	var server *Server
	err := server.Serve(context.Background(), strings.NewReader(""), &bytes.Buffer{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "server is nil")
}

func TestServeNilReader(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	err = server.Serve(context.Background(), nil, &bytes.Buffer{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "input reader is nil")
}

func TestServeNilWriter(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	err = server.Serve(context.Background(), strings.NewReader(""), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "output writer is nil")
}

func TestServerInvalidJsonRpcVersion(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"1.0","id":42,"method":"ping"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidRequest, resp.Error.Code)
	assert.Equal(t, json.RawMessage("42"), resp.ID)
}

func TestServerMissingMethod(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":99,"method":""}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidRequest, resp.Error.Code)
	assert.Equal(t, json.RawMessage("99"), resp.ID)
}

func TestServerUnknownMethod(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":100,"method":"unknown/method"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errMethodNotFound, resp.Error.Code)
	assert.Equal(t, "unknown/method", resp.Error.Data)
}

func TestServerInvalidRequestPreservesId(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"invalid","id":"string-id-123","method":"ping"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidRequest, resp.Error.Code)
	assert.Equal(t, json.RawMessage(`"string-id-123"`), resp.ID)
}

func TestServerInitialize(t *testing.T) {
	registry := NewRegistry()
	server, err := NewServer(registry, Implementation{Name: "simlin-mcp", Version: "dev"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"name":"client","version":"1.0"},"capabilities":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)

	result, ok := resp.Result.(InitializeResult)
	require.True(t, ok)
	assert.Equal(t, ProtocolVersion, result.ProtocolVersion)
	assert.Equal(t, "simlin-mcp", result.ServerInfo.Name)
	require.NotNil(t, result.Capabilities.Tools)
}

func TestServerInitializeMissingParams(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Equal(t, "missing params", resp.Error.Message)
}

func TestServerInitializeInvalidParamsJson(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":"not-an-object"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Equal(t, "invalid params", resp.Error.Message)
}

func TestServerInitializeMissingProtocolVersion(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"client","version":"1.0"},"capabilities":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Contains(t, resp.Error.Data, "missing required fields")
}

func TestServerInitializeMissingClientName(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"version":"1.0"},"capabilities":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Contains(t, resp.Error.Data, "missing required fields")
}

func TestServerInitializeMissingClientVersion(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"name":"client"},"capabilities":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Contains(t, resp.Error.Data, "missing required fields")
}

func TestServerInitializeMissingCapabilities(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","clientInfo":{"name":"client","version":"1.0"}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Contains(t, resp.Error.Data, "missing client capabilities")
}

func TestServerListTools(t *testing.T) {
	registry := NewRegistry()
	tool := &stubTool{
		name:        "CreateModel",
		description: "create model",
		schema:      `{"name":"CreateModel","description":"create model","inputSchema":{"type":"object"},"outputSchema":{"type":"object"}}`,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "simlin-mcp", Version: "dev"})
	require.NoError(t, err)
	req := json.RawMessage(`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)

	result, ok := resp.Result.(ListToolsResult)
	require.True(t, ok)
	require.Len(t, result.Tools, 1)
	assert.Equal(t, "CreateModel", result.Tools[0].Name)
}

func TestServerListToolsInvalidParams(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/list","params":"not-an-object"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
}

func TestServerCallTool(t *testing.T) {
	registry := NewRegistry()
	calledWith := ""
	tool := &stubTool{
		name:        "CreateModel",
		description: "create model",
		schema:      `{"name":"CreateModel","description":"create model","inputSchema":{"type":"object"},"outputSchema":{"type":"object"}}`,
		result:      `{"modelName":"main","error":null}`,
		calledWith:  &calledWith,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "simlin-mcp", Version: "dev"})
	require.NoError(t, err)
	req := json.RawMessage(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"CreateModel","arguments":{"projectPath":"demo.json"}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)

	result, ok := resp.Result.(CallToolResult)
	require.True(t, ok)
	require.Len(t, result.Content, 1)
	assert.Equal(t, "text", result.Content[0].Type)
	assert.Equal(t, `{"modelName":"main","error":null}`, result.Content[0].Text)
	assert.Equal(t, "main", result.StructuredContent["modelName"])
	assert.False(t, result.IsError)
	assert.Equal(t, `{"projectPath":"demo.json"}`, calledWith)
}

func TestServerCallToolErrorResult(t *testing.T) {
	registry := NewRegistry()
	tool := &stubTool{
		name:        "CreateModel",
		description: "create model",
		schema:      `{"name":"CreateModel","description":"create model","inputSchema":{"type":"object"},"outputSchema":{"type":"object"}}`,
		result:      `{"error":"boom"}`,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "simlin-mcp", Version: "dev"})
	require.NoError(t, err)
	req := json.RawMessage(`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"CreateModel","arguments":{"projectPath":"demo.json"}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)

	result, ok := resp.Result.(CallToolResult)
	require.True(t, ok)
	assert.True(t, result.IsError)
}

func TestServerCallToolMissing(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "simlin-mcp", Version: "dev"})
	require.NoError(t, err)
	req := json.RawMessage(`{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"Missing","arguments":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errMethodNotFound, resp.Error.Code)
}

func TestServerCallToolTaskUnsupported(t *testing.T) {
	registry := NewRegistry()
	tool := &stubTool{
		name:        "CreateModel",
		description: "create model",
		schema:      `{"name":"CreateModel","description":"create model","inputSchema":{"type":"object"},"outputSchema":{"type":"object"}}`,
		result:      `{"modelName":"main","error":null}`,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "simlin-mcp", Version: "dev"})
	require.NoError(t, err)
	req := json.RawMessage(`{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"CreateModel","arguments":{},"task":{"id":"task-1"}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
}

func TestServerCallToolPanicRecovery(t *testing.T) {
	registry := NewRegistry()
	require.NoError(t, registry.Register(&panicTool{}))

	server, err := NewServer(registry, Implementation{Name: "simlin-mcp", Version: "dev"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"PanicTool","arguments":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInternal, resp.Error.Code)
	assert.Equal(t, "tool panic", resp.Error.Message)
	assert.Equal(t, "intentional panic for testing", resp.Error.Data)
}

func TestServerCallToolMissingParams(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/call"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Equal(t, "missing params", resp.Error.Message)
}

func TestServerCallToolInvalidParamsJson(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":"not-an-object"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Equal(t, "invalid params", resp.Error.Message)
}

func TestServerCallToolMissingName(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"arguments":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidParams, resp.Error.Code)
	assert.Contains(t, resp.Error.Data, "tool name is required")
}

func TestServerCallToolNullArguments(t *testing.T) {
	registry := NewRegistry()
	calledWith := ""
	tool := &stubTool{
		name:        "Echo",
		description: "echoes input",
		schema:      `{"name":"Echo","description":"echoes input","inputSchema":{"type":"object"}}`,
		result:      `{"result":"ok"}`,
		calledWith:  &calledWith,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"Echo","arguments":null}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)
	assert.Equal(t, "{}", calledWith)
}

func TestServerCallToolEmptyArguments(t *testing.T) {
	registry := NewRegistry()
	calledWith := ""
	tool := &stubTool{
		name:        "Echo",
		description: "echoes input",
		schema:      `{"name":"Echo","description":"echoes input","inputSchema":{"type":"object"}}`,
		result:      `{"result":"ok"}`,
		calledWith:  &calledWith,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"Echo"}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Nil(t, resp.Error)
	assert.Equal(t, "{}", calledWith)
}

func TestServerCallToolInvalidJsonOutput(t *testing.T) {
	registry := NewRegistry()
	tool := &stubTool{
		name:        "BadOutput",
		description: "returns invalid JSON",
		schema:      `{"name":"BadOutput","description":"returns invalid JSON","inputSchema":{"type":"object"}}`,
		result:      `not valid json`,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"BadOutput","arguments":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInternal, resp.Error.Code)
	assert.Equal(t, "failed to parse tool result", resp.Error.Message)
}

func TestServerCallToolNullJsonOutput(t *testing.T) {
	registry := NewRegistry()
	tool := &stubTool{
		name:        "NullOutput",
		description: "returns null",
		schema:      `{"name":"NullOutput","description":"returns null","inputSchema":{"type":"object"}}`,
		result:      `null`,
	}
	require.NoError(t, registry.Register(tool))

	server, err := NewServer(registry, Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"NullOutput","arguments":{}}}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInternal, resp.Error.Code)
	assert.Equal(t, "failed to parse tool result", resp.Error.Message)
}

func TestToolResultHasError(t *testing.T) {
	tests := []struct {
		name     string
		input    map[string]any
		expected bool
	}{
		{
			name:     "no error field",
			input:    map[string]any{"result": "ok"},
			expected: false,
		},
		{
			name:     "error field is nil",
			input:    map[string]any{"error": nil},
			expected: false,
		},
		{
			name:     "error field is empty string",
			input:    map[string]any{"error": ""},
			expected: false,
		},
		{
			name:     "error field is non-empty string",
			input:    map[string]any{"error": "something went wrong"},
			expected: true,
		},
		{
			name:     "error field is a number",
			input:    map[string]any{"error": 42},
			expected: true,
		},
		{
			name:     "error field is an object",
			input:    map[string]any{"error": map[string]any{"code": 500}},
			expected: true,
		},
		{
			name:     "error field is a boolean true",
			input:    map[string]any{"error": true},
			expected: true,
		},
		{
			name:     "error field is a boolean false",
			input:    map[string]any{"error": false},
			expected: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := toolResultHasError(tc.input)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestServerInvalidJsonRequest(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{invalid json`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidRequest, resp.Error.Code)
	assert.Equal(t, "invalid request", resp.Error.Message)
	assert.Equal(t, json.RawMessage("null"), resp.ID)
}

func TestServerUnknownNotification(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"2.0","method":"unknown/notification"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	assert.Nil(t, resp)
}

func TestServerInvalidRequestNoId(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	req := json.RawMessage(`{"jsonrpc":"1.0","method":"ping"}`)
	resp, err := server.handleRaw(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Error)
	assert.Equal(t, errInvalidRequest, resp.Error.Code)
	assert.Equal(t, json.RawMessage("null"), resp.ID)
}

type failingWriter struct{}

func (f *failingWriter) Write(p []byte) (n int, err error) {
	return 0, fmt.Errorf("write failed")
}

func TestServeWriteErrorOnParseErrorResponse(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	in := strings.NewReader("not-json")
	out := &failingWriter{}

	err = server.Serve(context.Background(), in, out)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "writing parse error response")
}

func TestServeWriteErrorOnResponse(t *testing.T) {
	server, err := NewServer(NewRegistry(), Implementation{Name: "test", Version: "1.0"})
	require.NoError(t, err)

	in := strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"ping"}`)
	out := &failingWriter{}

	err = server.Serve(context.Background(), in, out)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "writing response")
}
