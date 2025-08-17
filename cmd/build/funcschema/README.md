# funcschema

A code generation tool that creates Model Context Protocol (MCP) Tool definitions from Go functions.

## Purpose

`funcschema` bridges the gap between Go functions and MCP-compatible tool definitions by automatically generating:
- JSON Schema definitions for function inputs and outputs
- MCP Tool definition constants
- Generic wrapper functions that handle JSON marshaling/unmarshaling

This enables Go functions to be exposed as tools that can be invoked by AI agents and other MCP-compatible systems.

## Usage

```bash
go run . -func <FunctionName> -input <source.go>
```

This generates a `<functionname>_tool.go` file containing:
- `<FunctionName>ToolDef`: A constant with the MCP tool definition JSON
- `<FunctionName>Tool`: A wrapper function that accepts and returns JSON strings

## Function Requirements

Functions must follow this pattern:

```go
// With arguments
func MyFunction(ctx context.Context, req RequestStruct) ResultStruct {
    // ...
}

// Without arguments (context-only)
func MyFunction(ctx context.Context) ResultStruct {
    // ...
}
```

### Requirements:
- First parameter must be `context.Context`
- Optional second parameter must be a struct type (not a pointer)
- Must return exactly one value (a struct)
- Return struct must contain an `Error *string` field for error handling
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
The generated wrapper function:
- Accepts `context.Context` and a JSON string as input
- Unmarshals JSON to the request struct (if applicable)
- Calls the original function
- Marshals the result back to JSON
- Handles errors by populating the Error field

## Example

Given this function:
```go
type GetDataRequest struct {
    DatasetId string `json:"datasetId"`
    Limit     int    `json:"limit"`
}

type GetDataResult struct {
    Data  []string
    Error *string
}

func GetData(ctx context.Context, req GetDataRequest) GetDataResult {
    // Implementation
}
```

Running `funcschema -func GetData -input data.go` generates:
- MCP tool definition with name `get_data`
- Input schema matching `GetDataRequest` structure
- Output schema matching `GetDataResult` structure
- `GetDataTool(ctx, input string) string` wrapper function

## OpenAI Compatibility

The tool follows OpenAI's JSON Schema subset limitations:
- No `$ref` references
- No `oneOf` for complex types (uses `anyOf` instead)
- All fields in `required` arrays
- Simple nullable types use `["type", "null"]` format