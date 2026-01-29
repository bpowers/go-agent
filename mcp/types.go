// Package mcp provides a JSON-RPC based Model Context Protocol (MCP) server implementation.
//
// MCP is a protocol for exposing tools to LLM-powered applications, enabling AI assistants
// to interact with external systems through a standardized interface. This package implements
// the server side of the protocol, allowing Go applications to expose tools that can be
// called by MCP clients such as Claude Code or other LLM orchestrators.
//
// # Basic Usage
//
// Create a registry, register tools that implement [chat.Tool], then create and run a server:
//
//	registry := mcp.NewRegistry()
//	registry.Register(myTool)
//
//	server, err := mcp.NewServer(registry, mcp.Implementation{
//	    Name:    "my-server",
//	    Version: "1.0.0",
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Serve over stdio (typical for MCP)
//	if err := server.Serve(ctx, os.Stdin, os.Stdout); err != nil {
//	    log.Fatal(err)
//	}
//
// # Tools
//
// Tools must implement the [chat.Tool] interface from github.com/bpowers/go-agent/chat.
// The tool's MCPJsonSchema method should return a JSON string containing the tool's name,
// description, and input schema. Tool execution receives JSON input and must return JSON output.
//
// # Protocol Details
//
// This implementation supports the following MCP methods:
//   - initialize: Handshake and capability exchange
//   - ping: Connection health check
//   - tools/list: Enumerate available tools
//   - tools/call: Execute a tool
//   - notifications/initialized: Client ready notification (no response)
package mcp

import "encoding/json"

// ProtocolVersion is the MCP protocol version supported by this server.
const ProtocolVersion = "2025-11-25"

// Request represents a JSON-RPC 2.0 request message.
// The ID field is omitted for notification requests that don't expect a response.
type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitzero"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitzero"`
}

// Response represents a JSON-RPC 2.0 response message.
// Either Result or Error will be set, but not both.
type Response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitzero"`
	Result  any             `json:"result,omitzero"`
	Error   *Error          `json:"error,omitzero"`
}

// Error represents a JSON-RPC 2.0 error object.
// Standard error codes follow the JSON-RPC specification (-32700 to -32600)
// with additional MCP-specific codes as needed.
type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitzero"`
}

// Implementation identifies an MCP server or client implementation.
// Name and Version are required; Description is optional.
type Implementation struct {
	Name        string `json:"name"`
	Version     string `json:"version"`
	Description string `json:"description,omitzero"`
}

// ToolDefinition describes a tool's interface as returned by tools/list.
// InputSchema is required and must be a valid JSON Schema object.
type ToolDefinition struct {
	Name         string          `json:"name"`
	Description  string          `json:"description,omitzero"`
	InputSchema  json.RawMessage `json:"inputSchema"`
	OutputSchema json.RawMessage `json:"outputSchema,omitzero"`
}

// ToolCapabilities describes the server's tool-related capabilities.
// ListChanged indicates whether the server supports dynamic tool list updates.
type ToolCapabilities struct {
	ListChanged bool `json:"listChanged,omitzero"`
}

// ServerCapabilities describes what features the server supports.
// Currently only tool capabilities are implemented.
type ServerCapabilities struct {
	Tools *ToolCapabilities `json:"tools,omitzero"`
}

// InitializeResult is returned by the initialize method during handshake.
// It communicates the server's identity, supported protocol version, and capabilities.
type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	ServerInfo      Implementation     `json:"serverInfo"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	Instructions    string             `json:"instructions,omitzero"`
}

// ListToolsResult is returned by the tools/list method.
// NextCursor is used for pagination; an empty value indicates no more results.
type ListToolsResult struct {
	Tools      []ToolDefinition `json:"tools"`
	NextCursor string           `json:"nextCursor,omitzero"`
}

// ContentBlock represents a piece of content in a tool result.
// Currently only "text" type is supported.
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// CallToolResult is returned by the tools/call method.
// Content contains the tool output as content blocks for display.
// StructuredContent contains the parsed JSON output for programmatic access.
// IsError is true if the tool execution encountered an error (distinct from JSON-RPC errors).
type CallToolResult struct {
	Content           []ContentBlock `json:"content"`
	StructuredContent map[string]any `json:"structuredContent,omitzero"`
	IsError           bool           `json:"isError,omitzero"`
}
