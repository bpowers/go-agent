package mcp

import "encoding/json"

// ProtocolVersion is the MCP protocol version supported by this server.
const ProtocolVersion = "2025-11-25"

type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitzero"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitzero"`
}

type Response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitzero"`
	Result  any             `json:"result,omitzero"`
	Error   *Error          `json:"error,omitzero"`
}

type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitzero"`
}

type Implementation struct {
	Name        string `json:"name"`
	Version     string `json:"version"`
	Description string `json:"description,omitzero"`
}

type ToolDefinition struct {
	Name         string          `json:"name"`
	Description  string          `json:"description,omitzero"`
	InputSchema  json.RawMessage `json:"inputSchema"`
	OutputSchema json.RawMessage `json:"outputSchema,omitzero"`
}

type ToolCapabilities struct {
	ListChanged bool `json:"listChanged,omitzero"`
}

type ServerCapabilities struct {
	Tools *ToolCapabilities `json:"tools,omitzero"`
}

type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	ServerInfo      Implementation     `json:"serverInfo"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	Instructions    string             `json:"instructions,omitzero"`
}

type ListToolsResult struct {
	Tools      []ToolDefinition `json:"tools"`
	NextCursor string           `json:"nextCursor,omitzero"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type CallToolResult struct {
	Content           []ContentBlock `json:"content"`
	StructuredContent map[string]any `json:"structuredContent,omitzero"`
	IsError           bool           `json:"isError,omitzero"`
}
