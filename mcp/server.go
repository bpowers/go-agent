package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
)

const (
	errParse          = -32700
	errInvalidRequest = -32600
	errMethodNotFound = -32601
	errInvalidParams  = -32602
	errInternal       = -32603
)

type Option func(*Server)

type Server struct {
	registry        *Registry
	info            Implementation
	protocolVersion string
	instructions    string
}

func NewServer(registry *Registry, info Implementation, opts ...Option) (*Server, error) {
	if registry == nil {
		return nil, fmt.Errorf("new server: registry is required")
	}
	if info.Name == "" {
		return nil, fmt.Errorf("new server: server name is required")
	}
	if info.Version == "" {
		return nil, fmt.Errorf("new server: server version is required")
	}

	server := &Server{
		registry:        registry,
		info:            info,
		protocolVersion: ProtocolVersion,
	}

	for _, opt := range opts {
		if opt != nil {
			opt(server)
		}
	}

	if server.protocolVersion == "" {
		return nil, fmt.Errorf("new server: protocol version is required")
	}

	return server, nil
}

func WithInstructions(instructions string) Option {
	return func(server *Server) {
		server.instructions = instructions
	}
}

func WithProtocolVersion(version string) Option {
	return func(server *Server) {
		server.protocolVersion = version
	}
}

func (s *Server) Serve(ctx context.Context, in io.Reader, out io.Writer) error {
	if s == nil {
		return fmt.Errorf("serve: server is nil")
	}
	if in == nil {
		return fmt.Errorf("serve: input reader is nil")
	}
	if out == nil {
		return fmt.Errorf("serve: output writer is nil")
	}

	decoder := json.NewDecoder(in)
	encoder := json.NewEncoder(out)

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("serve: %w", ctx.Err())
		default:
		}

		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			if err == io.EOF {
				return nil
			}
			resp := errorResponse(json.RawMessage("null"), errParse, "parse error", err.Error())
			if encodeErr := encoder.Encode(resp); encodeErr != nil {
				return fmt.Errorf("serve: writing parse error response: %w", encodeErr)
			}
			return fmt.Errorf("serve: decode failed: %w", err)
		}

		resp, err := s.handleRaw(ctx, raw)
		if err != nil {
			return err
		}
		if resp == nil {
			continue
		}
		if err := encoder.Encode(resp); err != nil {
			return fmt.Errorf("serve: writing response: %w", err)
		}
	}
}

func (s *Server) handleRaw(ctx context.Context, raw json.RawMessage) (*Response, error) {
	var req Request
	if err := json.Unmarshal(raw, &req); err != nil {
		return errorResponse(json.RawMessage("null"), errInvalidRequest, "invalid request", err.Error()), nil
	}

	if req.JSONRPC != "2.0" || req.Method == "" {
		return errorResponse(requestID(req.ID), errInvalidRequest, "invalid request", nil), nil
	}

	if len(req.ID) == 0 {
		return s.handleNotification(ctx, req), nil
	}

	switch req.Method {
	case "initialize":
		return s.handleInitialize(req)
	case "ping":
		return resultResponse(req.ID, struct{}{}), nil
	case "tools/list":
		return s.handleListTools(req)
	case "tools/call":
		return s.handleCallTool(ctx, req)
	default:
		return errorResponse(req.ID, errMethodNotFound, "method not found", req.Method), nil
	}
}

func (s *Server) handleNotification(ctx context.Context, req Request) *Response {
	switch req.Method {
	case "notifications/initialized":
		return nil
	default:
		return nil
	}
}

func (s *Server) handleInitialize(req Request) (*Response, error) {
	if len(req.Params) == 0 {
		return errorResponse(req.ID, errInvalidParams, "missing params", nil), nil
	}

	var params struct {
		ProtocolVersion string          `json:"protocolVersion"`
		ClientInfo      Implementation  `json:"clientInfo"`
		Capabilities    json.RawMessage `json:"capabilities"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		return errorResponse(req.ID, errInvalidParams, "invalid params", err.Error()), nil
	}
	if params.ProtocolVersion == "" || params.ClientInfo.Name == "" || params.ClientInfo.Version == "" {
		return errorResponse(req.ID, errInvalidParams, "invalid params", "missing required fields"), nil
	}
	if len(params.Capabilities) == 0 {
		return errorResponse(req.ID, errInvalidParams, "invalid params", "missing client capabilities"), nil
	}

	result := InitializeResult{
		ProtocolVersion: s.protocolVersion,
		ServerInfo:      s.info,
		Capabilities: ServerCapabilities{
			Tools: &ToolCapabilities{},
		},
	}
	if s.instructions != "" {
		result.Instructions = s.instructions
	}

	return resultResponse(req.ID, result), nil
}

func (s *Server) handleListTools(req Request) (*Response, error) {
	if len(req.Params) > 0 {
		var params struct {
			Cursor json.RawMessage `json:"cursor"`
		}
		if err := json.Unmarshal(req.Params, &params); err != nil {
			return errorResponse(req.ID, errInvalidParams, "invalid params", err.Error()), nil
		}
		// Pagination is not implemented; cursor is parsed but ignored.
	}

	result := ListToolsResult{
		Tools: s.registry.Definitions(),
	}
	return resultResponse(req.ID, result), nil
}

func (s *Server) handleCallTool(ctx context.Context, req Request) (resp *Response, err error) {
	if len(req.Params) == 0 {
		return errorResponse(req.ID, errInvalidParams, "missing params", nil), nil
	}

	var params struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		Task      json.RawMessage `json:"task"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		return errorResponse(req.ID, errInvalidParams, "invalid params", err.Error()), nil
	}
	if params.Name == "" {
		return errorResponse(req.ID, errInvalidParams, "invalid params", "tool name is required"), nil
	}
	if len(params.Task) > 0 && !bytes.Equal(bytes.TrimSpace(params.Task), []byte("null")) {
		return errorResponse(req.ID, errInvalidParams, "task augmentation not supported", nil), nil
	}

	tool, ok := s.registry.Get(params.Name)
	if !ok {
		return errorResponse(req.ID, errMethodNotFound, "tool not found", params.Name), nil
	}

	args := normalizeArguments(params.Arguments)

	// Recover from panics in tool execution to prevent server crash
	defer func() {
		if r := recover(); r != nil {
			resp = errorResponse(req.ID, errInternal, "tool panic", fmt.Sprintf("%v", r))
			err = nil
		}
	}()

	output := tool.Call(ctx, string(args))

	result, err := parseToolResult(output)
	if err != nil {
		return errorResponse(req.ID, errInternal, "failed to parse tool result", err.Error()), nil
	}

	return resultResponse(req.ID, result), nil
}

func parseToolResult(output string) (CallToolResult, error) {
	result := CallToolResult{
		Content: []ContentBlock{
			{
				Type: "text",
				Text: output,
			},
		},
	}

	var structured map[string]any
	if err := json.Unmarshal([]byte(output), &structured); err != nil {
		return CallToolResult{}, fmt.Errorf("parse tool output: %w", err)
	}
	if structured == nil {
		return CallToolResult{}, fmt.Errorf("parse tool output: expected object")
	}

	result.StructuredContent = structured
	result.IsError = toolResultHasError(structured)

	return result, nil
}

func toolResultHasError(structured map[string]any) bool {
	raw, ok := structured["error"]
	if !ok || raw == nil {
		return false
	}

	if errStr, ok := raw.(string); ok {
		return errStr != ""
	}
	return true
}

func normalizeArguments(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return []byte("{}")
	}
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("null")) {
		return []byte("{}")
	}
	return trimmed
}

func resultResponse(id json.RawMessage, result any) *Response {
	return &Response{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
}

func errorResponse(id json.RawMessage, code int, message string, data any) *Response {
	return &Response{
		JSONRPC: "2.0",
		ID:      id,
		Error: &Error{
			Code:    code,
			Message: message,
			Data:    data,
		},
	}
}

func requestID(id json.RawMessage) json.RawMessage {
	if len(id) == 0 {
		return json.RawMessage("null")
	}
	return id
}
