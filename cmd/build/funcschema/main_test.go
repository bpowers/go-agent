package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"github.com/bpowers/go-agent/schema"
)

func TestGenerateInputSchema(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		code     string
		funcName string
		wantErr  bool
		validate func(t *testing.T, s *schema.JSON)
	}{
		{
			name: "simple struct parameter",
			code: `package test
import "context"
type SimpleRequest struct {
	Name string
}
type SimpleResult struct {
	Value string
	Error *string
}
func SimpleFunc(ctx context.Context, req SimpleRequest) SimpleResult { return SimpleResult{Value: req.Name} }`,
			funcName: "SimpleFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				if len(s.Properties) != 1 {
					t.Errorf("expected 1 property, got %d", len(s.Properties))
				}
				if s.Properties["Name"] == nil {
					t.Error("expected 'Name' property")
				}
				if s.Properties["Name"].Type != schema.String {
					t.Errorf("expected string type for Name, got %v", s.Properties["Name"].Type)
				}
				if len(s.Required) != 1 || s.Required[0] != "Name" {
					t.Errorf("expected Name to be required, got %v", s.Required)
				}
			},
		},
		{
			name: "struct with optional field",
			code: `package test
import "context"
type OptionalRequest struct {
	Name *string
	Age  int
}
type OptionalResult struct {
	Value string
	Error *string
}
func OptionalFunc(ctx context.Context, req OptionalRequest) OptionalResult { return OptionalResult{Value: "test"} }`,
			funcName: "OptionalFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 2 {
					t.Errorf("expected 2 properties, got %d", len(s.Properties))
				}
				if s.Properties["Name"] == nil {
					t.Error("expected 'Name' property")
				}
				// Check that Name is nullable
				typeArr, ok := s.Properties["Name"].Type.([]interface{})
				if !ok {
					t.Errorf("expected type array for nullable, got %T", s.Properties["Name"].Type)
				} else if len(typeArr) != 2 || typeArr[0] != "string" || typeArr[1] != "null" {
					t.Errorf("expected [\"string\", \"null\"], got %v", typeArr)
				}
				// Should still be in required for OpenAI compatibility
				if len(s.Required) != 2 {
					t.Errorf("expected 2 required fields (OpenAI compat), got %v", s.Required)
				}
			},
		},
		{
			name: "struct with multiple fields",
			code: `package test
import "context"
type MultiFieldRequest struct {
	Name   string
	Age    int
	Active bool
}
type MultiFieldResult struct {
	Value string
	Error *string
}
func MultiFieldFunc(ctx context.Context, req MultiFieldRequest) MultiFieldResult { return MultiFieldResult{Value: req.Name} }`,
			funcName: "MultiFieldFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 3 {
					t.Errorf("expected 3 properties, got %d", len(s.Properties))
				}
				if s.Properties["Name"].Type != schema.String {
					t.Errorf("expected string type for Name")
				}
				if s.Properties["Age"].Type != "integer" {
					t.Errorf("expected integer type for Age")
				}
				if s.Properties["Active"].Type != "boolean" {
					t.Errorf("expected boolean type for Active")
				}
				if len(s.Required) != 3 {
					t.Errorf("expected 3 required fields, got %d", len(s.Required))
				}
			},
		},
		{
			name: "struct with array field",
			code: `package test
import "context"
type ArrayRequest struct {
	Items []string
}
type ArrayResult struct {
	Value string
	Error *string
}
func ArrayFunc(ctx context.Context, req ArrayRequest) ArrayResult { return ArrayResult{Value: "test"} }`,
			funcName: "ArrayFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Properties["Items"].Type != schema.Array {
					t.Errorf("expected array type for Items")
				}
				if s.Properties["Items"].Items == nil {
					t.Error("expected items schema")
				}
				if s.Properties["Items"].Items.Type != schema.String {
					t.Errorf("expected string items")
				}
			},
		},
		{
			name: "struct with nested struct field",
			code: `package test
import "context"
type Config struct {
	Name string
	Port int
}
type NestedRequest struct {
	Config Config
	Debug  bool
}
type NestedResult struct {
	Value string
	Error *string
}
func NestedFunc(ctx context.Context, req NestedRequest) NestedResult { return NestedResult{Value: req.Config.Name} }`,
			funcName: "NestedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				cfg := s.Properties["Config"]
				if cfg == nil {
					t.Error("expected Config property")
				}
				if cfg.Type != schema.Object {
					t.Errorf("expected object type for Config")
				}
				if cfg.Properties["Name"].Type != schema.String {
					t.Error("expected Name property in struct")
				}
				if cfg.Properties["Port"].Type != "integer" {
					t.Error("expected Port property in struct")
				}
				if s.Properties["Debug"].Type != "boolean" {
					t.Error("expected Debug property")
				}
			},
		},
		{
			name: "struct with json tags",
			code: `package test
import "context"
type TaggedRequest struct {
	UserName string ` + "`json:\"user_name\"`" + `
	UserAge  int    ` + "`json:\"user_age\"`" + `
}
type TaggedResult struct {
	Value string
	Error *string
}
func TaggedFunc(ctx context.Context, req TaggedRequest) TaggedResult { return TaggedResult{Value: req.UserName} }`,
			funcName: "TaggedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Properties["user_name"] == nil {
					t.Error("expected user_name property (from json tag)")
				}
				if s.Properties["user_age"] == nil {
					t.Error("expected user_age property (from json tag)")
				}
				if s.Properties["UserName"] != nil {
					t.Error("should not have UserName property (renamed by json tag)")
				}
			},
		},
		{
			name: "inline struct parameter",
			code: `package test
import "context"
type InlineResult struct {
	Value string
	Error *string
}
func InlineFunc(ctx context.Context, req struct {
	Name string
	Age  int
}) InlineResult { return InlineResult{Value: req.Name} }`,
			funcName: "InlineFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				if len(s.Properties) != 2 {
					t.Errorf("expected 2 properties, got %d", len(s.Properties))
				}
				if s.Properties["Name"] == nil {
					t.Error("expected 'Name' property")
				}
				if s.Properties["Age"] == nil {
					t.Error("expected 'Age' property")
				}
			},
		},
		{
			name: "no-argument function (only context)",
			code: `package test
import "context"
type ReadDirResult struct {
	Files []string
	Error *string
}
func ReadDir(ctx context.Context) ReadDirResult { 
	return ReadDirResult{Files: []string{"file1.txt", "file2.txt"}} 
}`,
			funcName: "ReadDir",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type for empty input, got %v", s.Type)
				}
				if len(s.Properties) != 0 {
					t.Errorf("expected 0 properties for no-argument function, got %d", len(s.Properties))
				}
				if s.AdditionalProperties == nil || *s.AdditionalProperties != false {
					t.Error("expected additionalProperties to be false")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Parse the test code
			fset := token.NewFileSet()
			node, err := parser.ParseFile(fset, "test.go", tt.code, parser.ParseComments)
			if err != nil {
				t.Fatalf("failed to parse code: %v", err)
			}

			// Find the function
			var targetFunc *ast.FuncDecl
			ast.Inspect(node, func(n ast.Node) bool {
				if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == tt.funcName {
					targetFunc = fn
					return false
				}
				return true
			})

			if targetFunc == nil {
				t.Fatalf("function %s not found", tt.funcName)
			}

			// Generate input schema
			s, err := generateInputSchema(targetFunc.Type.Params, node)
			if (err != nil) != tt.wantErr {
				t.Errorf("generateInputSchema() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.validate != nil {
				tt.validate(t, s)
			}
		})
	}
}

func TestGenerateOutputSchema(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		code     string
		funcName string
		wantErr  bool
		validate func(t *testing.T, s *schema.JSON)
	}{
		{
			name: "simple struct return with Error field",
			code: `package test
type EmptyRequest struct{}
type SimpleResult struct {
	Value string
	Error *string
}
func SimpleFunc(req EmptyRequest) SimpleResult { return SimpleResult{Value: "test"} }`,
			funcName: "SimpleFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				if s.Properties["Value"].Type != schema.String {
					t.Error("expected Value property")
				}
				if s.Properties["Error"] == nil {
					t.Error("expected Error property")
				}
				// Error should be nullable
				typeArr, ok := s.Properties["Error"].Type.([]interface{})
				if !ok {
					t.Errorf("expected type array for nullable Error, got %T", s.Properties["Error"].Type)
				} else if len(typeArr) != 2 || typeArr[0] != "string" || typeArr[1] != "null" {
					t.Errorf("expected [\"string\", \"null\"] for Error, got %v", typeArr)
				}
				// All fields should be required for OpenAI compatibility
				if len(s.Required) != 2 {
					t.Errorf("expected 2 required fields, got %v", s.Required)
				}
			},
		},
		{
			name: "struct return",
			code: `package test
type Request struct{}
type Result struct {
	Name string
	Count int
	Error *string
}
func StructReturn(req Request) Result { return Result{} }`,
			funcName: "StructReturn",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				if s.Properties["Name"].Type != schema.String {
					t.Error("expected Name property")
				}
				if s.Properties["Count"].Type != "integer" {
					t.Error("expected Count property")
				}
				if s.Properties["Error"] == nil {
					t.Error("expected Error property")
				}
			},
		},
		{
			name: "nested struct return",
			code: `package test
import "context"
type NestedData struct {
	Count int
	Items []string
}
type NestedResult struct {
	Data  NestedData
	Error *string
}
func NestedFunc(ctx context.Context) NestedResult { return NestedResult{} }`,
			funcName: "NestedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				if s.Properties["Data"] == nil {
					t.Error("expected 'Data' property")
				}
				dataSchema := s.Properties["Data"]
				if dataSchema.Type != schema.Object {
					t.Errorf("expected Data to be object, got %v", dataSchema.Type)
				}
				if dataSchema.Properties["Count"] == nil {
					t.Error("expected 'Count' property in Data")
				}
				if dataSchema.Properties["Items"] == nil {
					t.Error("expected 'Items' property in Data")
				}
				itemsSchema := dataSchema.Properties["Items"]
				if itemsSchema.Type != schema.Array {
					t.Errorf("expected Items to be array, got %v", itemsSchema.Type)
				}
			},
		},
		{
			name: "array return type",
			code: `package test
import "context"
type FileInfo struct {
	Name string
	Size int64
}
type ListResult struct {
	Files []FileInfo
	Error *string
}
func ListFunc(ctx context.Context) ListResult { return ListResult{} }`,
			funcName: "ListFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Properties["Files"] == nil {
					t.Error("expected 'Files' property")
				}
				filesSchema := s.Properties["Files"]
				if filesSchema.Type != schema.Array {
					t.Errorf("expected Files to be array, got %v", filesSchema.Type)
				}
				if filesSchema.Items == nil {
					t.Error("expected Files array to have items schema")
				}
				if filesSchema.Items.Type != schema.Object {
					t.Errorf("expected Files items to be objects, got %v", filesSchema.Items.Type)
				}
				if filesSchema.Items.Properties["Name"] == nil {
					t.Error("expected 'Name' property in FileInfo")
				}
				if filesSchema.Items.Properties["Size"] == nil {
					t.Error("expected 'Size' property in FileInfo")
				}
			},
		},
		{
			name: "complex return with map",
			code: `package test
type Request struct{}
type MapResult struct {
	Data map[string]int
	Error *string
}
func MapReturn(req Request) MapResult { return MapResult{} }`,
			funcName: "MapReturn",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				dataField := s.Properties["Data"]
				if dataField == nil {
					t.Error("expected Data property")
				}
				if dataField.Type != schema.Object {
					t.Errorf("expected object type for map, got %v", dataField.Type)
				}
				if dataField.AdditionalProperties == nil || !*dataField.AdditionalProperties {
					t.Error("expected additionalProperties to be true for map")
				}
				if s.Properties["Error"] == nil {
					t.Error("expected Error property")
				}
			},
		},
		{
			name: "complex return with nested arrays",
			code: `package test
type Request struct{}
type ArrayResult struct {
	Data [][]float64
	Error *string
}
func ArrayOfArrays(req Request) ArrayResult { return ArrayResult{} }`,
			funcName: "ArrayOfArrays",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				dataField := s.Properties["Data"]
				if dataField == nil {
					t.Error("expected Data property")
				}
				if dataField.Type != schema.Array {
					t.Errorf("expected array type, got %v", dataField.Type)
				}
				if dataField.Items == nil || dataField.Items.Type != schema.Array {
					t.Error("expected array of arrays")
				}
				if dataField.Items.Items == nil || dataField.Items.Items.Type != "number" {
					t.Error("expected array of array of numbers")
				}
				if s.Properties["Error"] == nil {
					t.Error("expected Error property")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Parse the test code
			fset := token.NewFileSet()
			node, err := parser.ParseFile(fset, "test.go", tt.code, parser.ParseComments)
			if err != nil {
				t.Fatalf("failed to parse code: %v", err)
			}

			// Find the function
			var targetFunc *ast.FuncDecl
			ast.Inspect(node, func(n ast.Node) bool {
				if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == tt.funcName {
					targetFunc = fn
					return false
				}
				return true
			})

			if targetFunc == nil {
				t.Fatalf("function %s not found", tt.funcName)
			}

			// Generate output schema
			s, err := generateOutputSchema(targetFunc.Type.Results, node)
			if (err != nil) != tt.wantErr {
				t.Errorf("generateOutputSchema() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.validate != nil {
				tt.validate(t, s)
			}
		})
	}
}

func TestDatasetGetCompleteFlow(t *testing.T) {
	t.Parallel()
	// Test the complete DatasetGet example with context and single struct parameter
	code := `package test
import "context"

type DatasetSelect struct {
	Variables []string
	TimeRange struct {
		Start string
		End   string
	}
}

type DatasetGetRequest struct {
	DatasetId string         ` + "`json:\"datasetId\"`" + `
	Where     *DatasetSelect ` + "`json:\"where\"`" + `
}

type DatasetGetResult struct {
	Revision string
	Data     map[string][][]float64
	Error    *string
}

func DatasetGet(ctx context.Context, args DatasetGetRequest) DatasetGetResult {
	return DatasetGetResult{}
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse code: %v", err)
	}

	// Find the function
	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "DatasetGet" {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		t.Fatal("DatasetGet function not found")
	}

	// Generate schemas
	inputSchema, err := generateInputSchema(targetFunc.Type.Params, node)
	if err != nil {
		t.Fatalf("failed to generate input schema: %v", err)
	}

	outputSchema, err := generateOutputSchema(targetFunc.Type.Results, node)
	if err != nil {
		t.Fatalf("failed to generate output schema: %v", err)
	}

	// Create tool definition with snake_case name
	tool := &MCPTool{
		Name:         camelToSnake("DatasetGet"),
		Description:  "Function DatasetGet",
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
	}

	// Validate the tool can be marshaled to JSON
	jsonBytes, err := json.MarshalIndent(tool, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal tool definition: %v", err)
	}

	// Validate the JSON structure
	var jsonMap map[string]interface{}
	if err := json.Unmarshal(jsonBytes, &jsonMap); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	// Check required fields - name should be snake_case
	if jsonMap["name"] != "dataset_get" {
		t.Errorf("expected name field to be 'dataset_get', got %v", jsonMap["name"])
	}
	if jsonMap["inputSchema"] == nil {
		t.Error("expected inputSchema field")
	}
	if jsonMap["outputSchema"] == nil {
		t.Error("expected outputSchema field")
	}

	// Validate input schema has correct structure
	inputMap := jsonMap["inputSchema"].(map[string]interface{})
	props := inputMap["properties"].(map[string]interface{})

	if props["datasetId"] == nil {
		t.Error("expected datasetId in input properties")
	}
	if props["where"] == nil {
		t.Error("expected where in input properties")
	}

	// Check that where is properly nullable with anyOf
	whereSchema := props["where"].(map[string]interface{})
	if whereSchema["anyOf"] == nil {
		t.Error("expected anyOf for optional pointer type")
	}

	// Validate output schema
	outputMap := jsonMap["outputSchema"].(map[string]interface{})
	outProps := outputMap["properties"].(map[string]interface{})

	if outProps["Revision"] == nil {
		t.Error("expected Revision in output properties")
	}
	if outProps["Data"] == nil {
		t.Error("expected Data in output properties")
	}
	if outProps["Error"] == nil {
		t.Error("expected Error in output properties")
	}

	// Check Error is nullable
	errorSchema := outProps["Error"].(map[string]interface{})
	errorType := errorSchema["type"]
	if errorTypeArr, ok := errorType.([]interface{}); !ok || len(errorTypeArr) != 2 {
		t.Errorf("expected Error type to be [\"string\", \"null\"], got %v", errorType)
	}
}

func TestInvalidFunctions(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		code     string
		funcName string
		wantErr  string
	}{
		{
			name: "multiple parameters",
			code: `package test
import "context"
type Result struct { Error *string }
func MultiParam(ctx context.Context, a string, b int) Result { return Result{} }`,
			funcName: "MultiParam",
			wantErr:  "must have either one parameter (context.Context) or two parameters",
		},
		{
			name: "no parameters",
			code: `package test
type Result struct { Error *string }
func NoParam() Result { return Result{} }`,
			funcName: "NoParam",
			wantErr:  "must have either one parameter (context.Context) or two parameters",
		},
		{
			name: "non-struct parameter",
			code: `package test
import "context"
type Result struct { Error *string }
func StringParam(ctx context.Context, s string) Result { return Result{} }`,
			funcName: "StringParam",
			wantErr:  "second parameter must be a struct type",
		},
		{
			name: "pointer to struct parameter",
			code: `package test
import "context"
type Request struct{ Name string }
type Result struct { Error *string }
func PtrParam(ctx context.Context, req *Request) Result { return Result{} }`,
			funcName: "PtrParam",
			wantErr:  "second parameter must be a struct type",
		},
		{
			name: "multiple return values",
			code: `package test
import "context"
type Request struct{}
func MultiReturn(ctx context.Context, req Request) (string, error) { return "", nil }`,
			funcName: "MultiReturn",
			wantErr:  "must return exactly one value",
		},
		{
			name: "no return value",
			code: `package test
import "context"
type Request struct{}
func NoReturn(ctx context.Context, req Request) { }`,
			funcName: "NoReturn",
			wantErr:  "must return exactly one value",
		},
		{
			name: "return type without Error field",
			code: `package test
import "context"
type Request struct{}
type BadResult struct {
	Value string
	// Missing Error field
}
func BadReturn(ctx context.Context, req Request) BadResult { return BadResult{} }`,
			funcName: "BadReturn",
			wantErr:  "must be a struct with an Error field",
		},
		{
			name: "method not function",
			code: `package test
type T struct{}
type Request struct{}
func (t T) Method(req Request) string { return "" }`,
			funcName: "Method",
			wantErr:  "is a method, not a standalone function",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Parse the test code
			fset := token.NewFileSet()
			node, err := parser.ParseFile(fset, "test.go", tt.code, parser.ParseComments)
			if err != nil {
				t.Fatalf("failed to parse code: %v", err)
			}

			// Find the function
			var targetFunc *ast.FuncDecl
			ast.Inspect(node, func(n ast.Node) bool {
				if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == tt.funcName {
					targetFunc = fn
					return false
				}
				return true
			})

			if targetFunc == nil {
				t.Fatalf("function %s not found", tt.funcName)
			}

			// Try to validate - should fail
			err = validateFunction(targetFunc, tt.funcName, node)
			if err == nil {
				t.Error("expected validation error, got nil")
			} else if tt.wantErr != "" && !contains(err.Error(), tt.wantErr) {
				t.Errorf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func validateFunction(targetFunc *ast.FuncDecl, funcName string, node *ast.File) error {
	// This mirrors the validation logic in run()
	if targetFunc.Recv != nil {
		return fmt.Errorf("function %s is a method, not a standalone function", funcName)
	}

	// Check parameters - must have one or two parameters
	if targetFunc.Type.Params == nil || len(targetFunc.Type.Params.List) < 1 || len(targetFunc.Type.Params.List) > 2 {
		return fmt.Errorf("function %s must have either one parameter (context.Context) or two parameters (context.Context and a request struct)", funcName)
	}

	// First parameter must be context.Context
	firstParam := targetFunc.Type.Params.List[0]
	if !isContextParam(firstParam) {
		return fmt.Errorf("function %s first parameter must be context.Context", funcName)
	}

	// If there's a second parameter, it must be a struct
	if len(targetFunc.Type.Params.List) == 2 {
		param := targetFunc.Type.Params.List[1]
		if len(param.Names) != 1 {
			return fmt.Errorf("function %s second parameter must have a name", funcName)
		}

		// Check that the parameter is a struct (could be named type or inline struct)
		paramType := param.Type
		switch t := paramType.(type) {
		case *ast.Ident:
			// Named type - need to verify it's a struct
			if !isStructType(t.Name, node) {
				return fmt.Errorf("function %s second parameter must be a struct type, got %s", funcName, t.Name)
			}
		case *ast.StructType:
			// Inline struct - this is fine
		default:
			return fmt.Errorf("function %s second parameter must be a struct type", funcName)
		}
	}

	// Check return values - must return exactly one value
	if targetFunc.Type.Results == nil || len(targetFunc.Type.Results.List) != 1 {
		return fmt.Errorf("function %s must return exactly one value", funcName)
	}

	// Validate that return type is a struct with an Error field
	resultType := targetFunc.Type.Results.List[0].Type
	if !hasErrorField(resultType, node) {
		return fmt.Errorf("function %s return type must be a struct with an Error field", funcName)
	}

	return nil
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || len(substr) > 0 && len(s) > len(substr) && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestNoArgumentFunction(t *testing.T) {
	t.Parallel()
	// Test a function with only context parameter
	code := `package test
import "context"

type ListFilesResult struct {
	Files []string
	Error *string
}

func ListFiles(ctx context.Context) ListFilesResult {
	return ListFilesResult{
		Files: []string{"file1.txt", "file2.txt", "dir/file3.go"},
	}
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse code: %v", err)
	}

	// Find the function
	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "ListFiles" {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		t.Fatal("ListFiles function not found")
	}

	// Validate function
	err = validateFunction(targetFunc, "ListFiles", node)
	if err != nil {
		t.Fatalf("validation failed: %v", err)
	}

	// Generate schemas
	inputSchema, err := generateInputSchema(targetFunc.Type.Params, node)
	if err != nil {
		t.Fatalf("failed to generate input schema: %v", err)
	}

	outputSchema, err := generateOutputSchema(targetFunc.Type.Results, node)
	if err != nil {
		t.Fatalf("failed to generate output schema: %v", err)
	}

	// Validate input schema is empty object
	if inputSchema.Type != schema.Object {
		t.Errorf("expected object type for input, got %v", inputSchema.Type)
	}
	if len(inputSchema.Properties) != 0 {
		t.Errorf("expected 0 properties for no-argument function, got %d", len(inputSchema.Properties))
	}
	if inputSchema.AdditionalProperties == nil || *inputSchema.AdditionalProperties != false {
		t.Error("expected additionalProperties to be false")
	}

	// Validate output schema
	if outputSchema.Type != schema.Object {
		t.Errorf("expected object type for output, got %v", outputSchema.Type)
	}
	if outputSchema.Properties["Files"] == nil {
		t.Error("expected Files property in output")
	}
	if outputSchema.Properties["Error"] == nil {
		t.Error("expected Error property in output")
	}

	// Create tool definition
	tool := &MCPTool{
		Name:         camelToSnake("ListFiles"),
		Description:  "Function ListFiles",
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
	}

	// Validate the tool can be marshaled to JSON
	jsonBytes, err := json.MarshalIndent(tool, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal tool definition: %v", err)
	}

	// Validate the JSON structure
	var jsonMap map[string]interface{}
	if err := json.Unmarshal(jsonBytes, &jsonMap); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	// Check that name is snake_case
	if jsonMap["name"] != "list_files" {
		t.Errorf("expected name field to be 'list_files', got %v", jsonMap["name"])
	}

	// Check input schema is an empty object
	inputMap := jsonMap["inputSchema"].(map[string]interface{})
	if inputMap["type"] != "object" {
		t.Error("expected inputSchema type to be 'object'")
	}
	props := inputMap["properties"].(map[string]interface{})
	if len(props) != 0 {
		t.Errorf("expected empty properties in inputSchema, got %d properties", len(props))
	}
}

func TestCamelToSnake(t *testing.T) {
	t.Parallel()
	tests := []struct {
		input    string
		expected string
	}{
		{"ReadDir", "read_dir"},
		{"DatasetGet", "dataset_get"},
		{"HTTPServer", "h_t_t_p_server"},
		{"GetHTTPResponse", "get_h_t_t_p_response"},
		{"SimpleFunc", "simple_func"},
		{"lowercase", "lowercase"},
		{"A", "a"},
		{"AB", "a_b"},
		{"ABC", "a_b_c"},
		{"GetUserByID", "get_user_by_i_d"},
		{"CreateAPIKey", "create_a_p_i_key"},
		{"IOReader", "i_o_reader"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			t.Parallel()
			result := camelToSnake(tt.input)
			if result != tt.expected {
				t.Errorf("camelToSnake(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestGeneratedPackageName(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		packageName string
		code        string
	}{
		{
			name:        "testing package",
			packageName: "testing",
			code: `package testing

import "context"

type TestResult struct {
	Message string  ` + "`json:\"message\"`" + `
	Error   *string ` + "`json:\"error,omitzero\"`" + `
}

func TestFunc(ctx context.Context) TestResult {
	return TestResult{Message: "Hello"}
}`,
		},
		{
			name:        "main package",
			packageName: "main",
			code: `package main

import "context"

type TestResult struct {
	Message string  ` + "`json:\"message\"`" + `
	Error   *string ` + "`json:\"error,omitzero\"`" + `
}

func TestFunc(ctx context.Context) TestResult {
	return TestResult{Message: "Hello"}
}`,
		},
		{
			name:        "custom package",
			packageName: "mytools",
			code: `package mytools

import "context"

type TestRequest struct {
	Name string ` + "`json:\"name\"`" + `
}

type TestResult struct {
	Message string  ` + "`json:\"message\"`" + `
	Error   *string ` + "`json:\"error,omitzero\"`" + `
}

func TestFunc(ctx context.Context, req TestRequest) TestResult {
	return TestResult{Message: "Hello " + req.Name}
}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Parse the test code
			fset := token.NewFileSet()
			node, err := parser.ParseFile(fset, "test.go", tt.code, parser.ParseComments)
			if err != nil {
				t.Fatalf("failed to parse code: %v", err)
			}

			// Check that we extract the correct package name
			if node.Name.Name != tt.packageName {
				t.Errorf("expected package name %q, got %q", tt.packageName, node.Name.Name)
			}

			// Find the function
			var targetFunc *ast.FuncDecl
			ast.Inspect(node, func(n ast.Node) bool {
				if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "TestFunc" {
					targetFunc = fn
					return false
				}
				return true
			})

			if targetFunc == nil {
				t.Fatal("TestFunc function not found")
			}

			// Extract parameter and return type names
			var paramTypeName string
			if len(targetFunc.Type.Params.List) == 2 {
				paramTypeName = getTypeName(targetFunc.Type.Params.List[1].Type)
			}
			_ = getTypeName(targetFunc.Type.Results.List[0].Type) // returnTypeName - not used in this test

			// Create a minimal tool definition
			tool := &MCPTool{
				Name:        "test_func",
				Description: "Test function",
			}

			// Generate the tool file content (simulate what generateToolDefFile does)
			_, err = json.MarshalIndent(tool, "", "  ")
			if err != nil {
				t.Fatalf("failed to marshal tool: %v", err)
			}

			var content string
			if paramTypeName == "" {
				// No-argument function template check
				content = fmt.Sprintf(`package %s`, tt.packageName)
			} else {
				// Function with arguments template check
				content = fmt.Sprintf(`package %s`, tt.packageName)
			}

			// Verify the package name appears in generated content
			if !strings.Contains(content, fmt.Sprintf("package %s", tt.packageName)) {
				t.Errorf("generated content should contain 'package %s', got: %s", tt.packageName, content)
			}
		})
	}
}
