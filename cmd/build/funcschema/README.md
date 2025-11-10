# funcschema

A code generation tool that creates Model Context Protocol (MCP) Tool definitions from Go functions.

## Purpose

`funcschema` bridges the gap between Go functions and MCP-compatible tool definitions by automatically generating:
- JSON Schema definitions for function inputs and outputs
- `chat.Tool` implementations that expose those functions to LLM providers
- Wrapper logic that marshals/unmarshals JSON and maps Go errors into MCP responses

This enables Go functions to be exposed as tools that can be invoked by AI agents and other MCP-compatible systems.

## Usage

```bash
go run . -func <FunctionName> -input <source.go>
```

This generates a `<functionname>_tool.go` file containing a `<FunctionName>Tool` value that implements `chat.Tool`. The generated type includes:
- Tool definition metadata (name, description, JSON schema)
- A `Call(context.Context, string) string` method that bridges between JSON input and your Go function
- An internal result wrapper that includes the tool's JSON schema + error propagation

## Function Requirements

Functions must follow this pattern:

```go
// With arguments
func MyFunction(ctx context.Context, req RequestStruct) (ResultStruct, error) {
    // ...
}

// Without arguments (context-only)
func MyFunction(ctx context.Context) (ResultStruct, error) {
    // ...
}
```

### Requirements:
- First parameter must be `context.Context`
- Optional second parameter must be a struct type (not a pointer)
- Functions must return exactly two values: `(ResultStruct, error)`
- The result struct can contain any fields you need; the generator wraps it with an error pointer automatically
- Function must be standalone (not a method)

## Features

### JSON Schema Generation
- Generates OpenAI-compatible JSON schemas
- Handles complex Go types: structs, arrays, maps, pointers
- Respects JSON struct tags for field naming
- Treats pointer fields as nullable (using `["type", "null"]` format)
- All fields are marked as required for OpenAI compatibility

### Tool Naming
- Automatically converts Go function names from CamelCase to snake_case
- Example: `DatasetGet` becomes `dataset_get` in the tool definition

### Wrapper Functions
The generated `chat.Tool` implementation:
- Accepts `context.Context` and a JSON string via its `Call` method
- Unmarshals JSON into the request struct (if applicable)
- Calls the original function and captures the returned `(result, error)`
- Wraps the Go result with an internal struct that adds an optional `error` field for MCP responses
- Marshals the wrapped result to JSON before returning

## Example

Given this function:
```go
type GetDataRequest struct {
    DatasetId string `json:"datasetId"`
    Limit     int    `json:"limit"`
}

type GetDataResult struct {
    Data []string `json:"data"`
}

func GetData(ctx context.Context, req GetDataRequest) (GetDataResult, error) {
    // Implementation
    return GetDataResult{Data: []string{"foo"}}, nil
}
```

Running `funcschema -func GetData -input data.go` generates:
- MCP tool definition with name `get_data`
- Input schema matching `GetDataRequest` structure
- Output schema matching `GetDataResult` structure plus an `error` field
- `var GetDataTool chat.Tool = getDataTool{}` which you can register directly with any `chat.Chat`

## OpenAI Compatibility

The tool follows OpenAI's JSON Schema subset limitations:
- No `$ref` references
- No `oneOf` for complex types (uses `anyOf` instead)
- All fields in `required` arrays
- Simple nullable types use `["type", "null"]` format
