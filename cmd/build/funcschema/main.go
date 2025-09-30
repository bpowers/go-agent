package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/bpowers/go-agent/schema"
	"github.com/iancoleman/strcase"
	"mvdan.cc/gofumpt/format"
)

var (
	funcName  = flag.String("func", "", "Name of the function to generate tool definition for (required)")
	inputFile = flag.String("input", "", "Input Go source file (required)")
)

// MCPTool represents an MCP Tool definition
type MCPTool struct {
	Name         string                 `json:"name"`
	Title        string                 `json:"title,omitzero"`
	Description  string                 `json:"description"`
	InputSchema  *schema.JSON           `json:"inputSchema"`
	OutputSchema *schema.JSON           `json:"outputSchema,omitzero"`
	Annotations  map[string]interface{} `json:"annotations,omitzero"`
}

func main() {
	flag.Parse()

	if *funcName == "" || *inputFile == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// Parse the input file
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, *inputFile, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("parsing file: %w", err)
	}

	// Create doc.Package for extracting documentation using NewFromFiles
	// This avoids the deprecated ast.Package
	// Use doc.PreserveAST to prevent NewFromFiles from modifying the AST nodes
	docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls|doc.PreserveAST)
	if err != nil {
		return fmt.Errorf("creating doc package: %w", err)
	}

	// Find the target function
	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == *funcName {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		return fmt.Errorf("function %s not found in %s", *funcName, *inputFile)
	}

	// Validate the function
	if targetFunc.Recv != nil {
		return fmt.Errorf("function %s is a method, not a standalone function", *funcName)
	}

	// Check parameters - must have one or two parameters
	if targetFunc.Type.Params == nil || len(targetFunc.Type.Params.List) < 1 || len(targetFunc.Type.Params.List) > 2 {
		return fmt.Errorf("function %s must have either one parameter (context.Context) or two parameters (context.Context and a request struct)", *funcName)
	}

	// First parameter must be context.Context
	firstParam := targetFunc.Type.Params.List[0]
	if !isContextParam(firstParam) {
		return fmt.Errorf("function %s first parameter must be context.Context", *funcName)
	}

	// If there's a second parameter, it must be a struct
	if len(targetFunc.Type.Params.List) == 2 {
		param := targetFunc.Type.Params.List[1]
		if len(param.Names) != 1 {
			return fmt.Errorf("function %s second parameter must have a name", *funcName)
		}

		// Check that the parameter is a struct (could be named type or inline struct)
		paramType := param.Type
		switch t := paramType.(type) {
		case *ast.Ident:
			// Named type - need to verify it's a struct
			if !isStructType(t.Name, node) {
				return fmt.Errorf("function %s second parameter must be a struct type, got %s", *funcName, t.Name)
			}
		case *ast.StructType:
			// Inline struct - this is fine
		default:
			return fmt.Errorf("function %s second parameter must be a struct type", *funcName)
		}
	}

	// Check return values - must return exactly one value
	if targetFunc.Type.Results == nil || len(targetFunc.Type.Results.List) != 1 {
		return fmt.Errorf("function %s must return exactly one value", *funcName)
	}

	// Validate that return type is a struct with an Error field
	resultType := targetFunc.Type.Results.List[0].Type
	if !hasErrorField(resultType, node) {
		return fmt.Errorf("function %s return type must be a struct with an Error field", *funcName)
	}

	// Generate input schema from parameters
	inputSchema, err := generateInputSchema(targetFunc.Type.Params, node, docPkg)
	if err != nil {
		return fmt.Errorf("generating input schema: %w", err)
	}

	// Generate output schema from return type
	outputSchema, err := generateOutputSchema(targetFunc.Type.Results, node, docPkg)
	if err != nil {
		return fmt.Errorf("generating output schema: %w", err)
	}

	// Extract description from godoc comments directly from AST
	description := extractDescriptionFromAST(targetFunc)

	// Create the MCP tool definition
	tool := &MCPTool{
		Name:         strcase.ToSnake(*funcName),
		Description:  description,
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
	}

	// Extract parameter and return type names for the wrapper function
	var paramTypeName string
	if len(targetFunc.Type.Params.List) == 2 {
		// Skip the context parameter (index 0) and get the actual request struct (index 1)
		paramTypeName = getTypeName(targetFunc.Type.Params.List[1].Type)
	} else {
		// No request parameter, just context
		paramTypeName = ""
	}
	returnTypeName := getTypeName(targetFunc.Type.Results.List[0].Type)

	// Get the package name from the parsed file
	packageName := node.Name.Name

	// Generate the Go file with the tool definition const and wrapper function
	if err := generateToolDefFile(tool, *funcName, paramTypeName, returnTypeName, *inputFile, packageName); err != nil {
		return fmt.Errorf("generating tool definition file: %w", err)
	}

	fmt.Printf("Generated tool definition for %s\n", *funcName)

	return nil
}

func generateInputSchema(params *ast.FieldList, file *ast.File, docPkg *doc.Package) (*schema.JSON, error) {
	// We now expect one or two parameters: context.Context and optionally a struct
	if params == nil || len(params.List) < 1 || len(params.List) > 2 {
		return nil, fmt.Errorf("expected one or two parameters: context.Context and optionally a struct")
	}

	// If there's only one parameter (context), return empty object schema
	if len(params.List) == 1 {
		return &schema.JSON{
			Schema:               schema.URL,
			Type:                 schema.Object,
			Properties:           make(map[string]*schema.JSON),
			AdditionalProperties: boolPtr(false),
		}, nil
	}

	// Skip the first parameter (context.Context) and use the second
	param := params.List[1]
	if len(param.Names) != 1 {
		return nil, fmt.Errorf("second parameter must have a name")
	}

	// Generate schema for the struct parameter
	// This will return the struct's schema directly
	paramSchema, _, err := generateTypeSchema(param.Type, file, docPkg)
	if err != nil {
		return nil, fmt.Errorf("generating schema for parameter: %w", err)
	}

	// For a struct parameter, we return its schema directly as the input schema
	// The struct's properties become the top-level properties of the input
	return paramSchema, nil
}

func generateOutputSchema(results *ast.FieldList, file *ast.File, docPkg *doc.Package) (*schema.JSON, error) {
	if results == nil || len(results.List) != 1 {
		return nil, fmt.Errorf("function must return exactly one value")
	}

	result := results.List[0]
	s, _, err := generateTypeSchema(result.Type, file, docPkg)
	return s, err
}

func generateTypeSchema(expr ast.Expr, file *ast.File, docPkg *doc.Package) (*schema.JSON, bool, error) {
	switch t := expr.(type) {
	case *ast.SelectorExpr:
		// Handle qualified types like time.Time, url.URL, etc.
		if pkg, ok := t.X.(*ast.Ident); ok {
			qualifiedType := pkg.Name + "." + t.Sel.Name
			if s := generateStdlibTypeSchema(qualifiedType); s != nil {
				return s, false, nil
			}
		}
		// Fallback to object for unknown qualified types
		return &schema.JSON{Type: schema.Object}, false, nil
	case *ast.Ident:
		// Basic types or type references
		s, err := generateBasicTypeSchema(t.Name)
		if err != nil {
			return nil, false, err
		}
		// If it's not a basic type, it might be a struct type defined in the file
		if s.Type == schema.Object && t.Name != "interface{}" {
			// Try to find the type definition
			structSchema := findAndGenerateStructSchema(t.Name, file, docPkg)
			if structSchema != nil {
				return structSchema, false, nil
			}
		}
		return s, false, nil
	case *ast.StarExpr:
		// Pointer type - it's optional (can be the type or null)
		s, _, err := generateTypeSchema(t.X, file, docPkg)
		if err != nil {
			return nil, true, err
		}
		// For OpenAI compatibility, use type array format for nullable
		if simpleType, ok := s.Type.(schema.Type); ok && s.Properties == nil {
			// Simple type - use ["type", "null"] format
			s.Type = []interface{}{string(simpleType), "null"}
			return s, true, nil
		} else if s.Type == nil && s.AnyOf != nil {
			// Already using anyOf, keep it
			return s, true, nil
		} else {
			// Complex type (object) - also use type array if possible
			if s.Type == schema.Object {
				// For objects, we can't use ["object", "null"] in OpenAI
				// We need to use anyOf at a higher level
				return &schema.JSON{
					AnyOf: []*schema.JSON{
						s,
						{Type: "null"},
					},
				}, true, nil
			}
			return s, true, nil
		}
	case *ast.ArrayType:
		// Array type
		itemSchema, _, err := generateTypeSchema(t.Elt, file, docPkg)
		if err != nil {
			return nil, false, err
		}
		return &schema.JSON{
			Type:  schema.Array,
			Items: itemSchema,
		}, false, nil
	case *ast.MapType:
		// Map type - for now, we'll treat maps as generic objects
		// TODO: We could potentially create a custom schema structure that preserves value type info
		return &schema.JSON{
			Type:                 schema.Object,
			AdditionalProperties: boolPtr(true),
		}, false, nil
	case *ast.StructType:
		// Inline struct
		return generateStructTypeSchema(t, file, docPkg, "")
	case *ast.InterfaceType:
		// Interface type - treat as any
		return &schema.JSON{}, false, nil
	default:
		// Unknown type - return a generic object schema
		return &schema.JSON{Type: schema.Object}, false, nil
	}
}

func generateStdlibTypeSchema(qualifiedType string) *schema.JSON {
	switch qualifiedType {
	case "time.Time":
		// RFC3339 format for JSON - add description since we can't use format field
		return &schema.JSON{
			Type:        schema.String,
			Description: "RFC3339 date-time string",
		}
	case "time.Duration":
		// Duration as string (e.g., "5s", "1h30m")
		return &schema.JSON{
			Type:        schema.String,
			Description: "Duration string (e.g., '5s', '1h30m')",
		}
	case "url.URL":
		return &schema.JSON{
			Type:        schema.String,
			Description: "URL string",
		}
	case "uuid.UUID":
		return &schema.JSON{
			Type:        schema.String,
			Description: "UUID string",
		}
	case "net.IP":
		return &schema.JSON{
			Type:        schema.String,
			Description: "IP address (IPv4 or IPv6)",
		}
	case "json.RawMessage":
		// Any valid JSON
		return &schema.JSON{}
	default:
		return nil
	}
}

func generateBasicTypeSchema(typeName string) (*schema.JSON, error) {
	switch typeName {
	case "string":
		return &schema.JSON{Type: schema.String}, nil
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64":
		return &schema.JSON{Type: "integer"}, nil
	case "float32", "float64":
		return &schema.JSON{Type: "number"}, nil
	case "bool":
		return &schema.JSON{Type: "boolean"}, nil
	default:
		// Unknown type - treat as object for now
		return &schema.JSON{Type: schema.Object}, nil
	}
}

func findAndGenerateStructSchema(typeName string, file *ast.File, docPkg *doc.Package) *schema.JSON {
	var targetType *ast.TypeSpec
	ast.Inspect(file, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == typeName {
			targetType = ts
			return false
		}
		return true
	})

	if targetType == nil {
		return nil
	}

	structType, ok := targetType.Type.(*ast.StructType)
	if !ok {
		return nil
	}

	s, _, _ := generateStructTypeSchema(structType, file, docPkg, typeName)
	return s
}

func generateStructTypeSchema(structType *ast.StructType, file *ast.File, docPkg *doc.Package, typeName string) (*schema.JSON, bool, error) {
	s := &schema.JSON{
		Type:                 schema.Object,
		Properties:           make(map[string]*schema.JSON),
		AdditionalProperties: boolPtr(false),
	}

	var required []string
	// Track field names to handle shadowing - outer struct fields override embedded ones
	fieldNames := make(map[string]bool)

	// First pass: process regular (non-embedded) fields
	for _, field := range structType.Fields.List {
		if len(field.Names) == 0 {
			continue // Skip anonymous fields in first pass
		}

		for _, name := range field.Names {
			fieldName := name.Name
			if !ast.IsExported(fieldName) {
				continue // Skip unexported fields
			}

			// Get JSON tag if present
			jsonName := fieldName
			if field.Tag != nil {
				jsonTag, _ := parseJSONTag(field.Tag)
				if jsonTag == "-" {
					continue // Skip fields with json:"-"
				}
				if jsonTag != "" {
					jsonName = jsonTag
				}
			}

			// Generate schema for field type
			fieldSchema, _, err := generateTypeSchema(field.Type, file, docPkg)
			if err != nil {
				return nil, false, err
			}

			// Check for enum tag
			if field.Tag != nil {
				enumValues := parseEnumTag(field.Tag)
				if len(enumValues) > 0 && fieldSchema.Type == schema.String {
					fieldSchema.Enum = enumValues
				}
			}

			// Extract field documentation from doc.Package if available
			if typeName != "" && docPkg != nil {
				for _, dt := range docPkg.Types {
					if dt.Name == typeName {
						// Look for field documentation in the type's fields
						for _, f := range dt.Decl.Specs {
							if ts, ok := f.(*ast.TypeSpec); ok && ts.Name.Name == typeName {
								if st, ok := ts.Type.(*ast.StructType); ok {
									// Find the matching field
									for _, docField := range st.Fields.List {
										for _, docFieldName := range docField.Names {
											if docFieldName.Name == fieldName {
												// Use doc comments if available
												if docField.Doc != nil {
													description := extractFieldDescription(docField.Doc)
													if description != "" {
														fieldSchema.Description = description
													}
												} else if docField.Comment != nil {
													description := extractFieldDescription(docField.Comment)
													if description != "" {
														fieldSchema.Description = description
													}
												}
												break
											}
										}
									}
								}
							}
						}
						break
					}
				}
			}

			s.Properties[jsonName] = fieldSchema
			fieldNames[jsonName] = true

			// For OpenAI compatibility, ALL fields must be required
			required = append(required, jsonName)
		}
	}

	// Second pass: process embedded fields
	for _, field := range structType.Fields.List {
		if len(field.Names) != 0 {
			continue // Skip regular fields in second pass
		}

		// Anonymous field (embedded struct)
		embeddedType := field.Type

		// Handle pointer to struct
		if starExpr, ok := embeddedType.(*ast.StarExpr); ok {
			embeddedType = starExpr.X
		}

		// Get the embedded type name
		var embeddedTypeName string
		if ident, ok := embeddedType.(*ast.Ident); ok {
			embeddedTypeName = ident.Name

			// Skip unexported embedded types
			if !ast.IsExported(embeddedTypeName) {
				continue
			}

			// Find and process the embedded struct
			embeddedStruct := findAndGetStructType(embeddedTypeName, file)
			if embeddedStruct != nil {
				// Recursively get the schema for the embedded struct
				embeddedSchema, _, err := generateStructTypeSchema(embeddedStruct, file, docPkg, embeddedTypeName)
				if err != nil {
					return nil, false, err
				}

				// Merge embedded fields into parent, skipping fields that are already defined (shadowed)
				for propName, propSchema := range embeddedSchema.Properties {
					if !fieldNames[propName] {
						s.Properties[propName] = propSchema
						fieldNames[propName] = true
						required = append(required, propName)
					}
				}
			}
		}
		// TODO: Handle inline struct type embedding if needed
	}

	if len(required) > 0 {
		s.Required = required
	}

	return s, false, nil
}

// findAndGetStructType finds a struct type definition by name in the file
func findAndGetStructType(typeName string, file *ast.File) *ast.StructType {
	var result *ast.StructType
	ast.Inspect(file, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == typeName {
			if st, ok := ts.Type.(*ast.StructType); ok {
				result = st
				return false
			}
		}
		return true
	})
	return result
}

func parseJSONTag(tag *ast.BasicLit) (name string, omitempty bool) {
	if tag == nil {
		return "", false
	}

	// Remove quotes and backticks
	tagValue := tag.Value
	if len(tagValue) >= 2 {
		tagValue = tagValue[1 : len(tagValue)-1]
	}

	// Find json tag
	const jsonPrefix = "json:\""
	idx := strings.Index(tagValue, jsonPrefix)
	if idx == -1 {
		return "", false
	}

	jsonTag := tagValue[idx+len(jsonPrefix):]
	endIdx := strings.Index(jsonTag, "\"")
	if endIdx != -1 {
		jsonTag = jsonTag[:endIdx]
	}

	parts := strings.Split(jsonTag, ",")
	if len(parts) > 0 {
		name = parts[0]
	}

	for _, part := range parts[1:] {
		if part == "omitempty" || part == "omitzero" {
			omitempty = true
		}
	}

	return name, omitempty
}

func parseEnumTag(tag *ast.BasicLit) []string {
	if tag == nil {
		return nil
	}

	// Remove quotes and backticks
	tagValue := tag.Value
	if len(tagValue) >= 2 {
		tagValue = tagValue[1 : len(tagValue)-1]
	}

	// Find enum tag
	const enumPrefix = "enum:\""
	idx := strings.Index(tagValue, enumPrefix)
	if idx == -1 {
		return nil
	}

	enumTag := tagValue[idx+len(enumPrefix):]
	endIdx := strings.Index(enumTag, "\"")
	if endIdx != -1 {
		enumTag = enumTag[:endIdx]
	}

	if enumTag == "" {
		return nil
	}

	values := strings.Split(enumTag, ",")
	for i := range values {
		values[i] = strings.TrimSpace(values[i])
	}

	return values
}

func generateToolDefFile(tool *MCPTool, funcName, paramTypeName, returnTypeName, inputFile, packageName string) error {
	// Marshal the tool definition to JSON (compact, not pretty-printed)
	jsonBytes, err := json.Marshal(tool)
	if err != nil {
		return fmt.Errorf("marshaling tool definition: %w", err)
	}

	// Generate the Go file content
	dir := filepath.Dir(inputFile)
	outputFile := filepath.Join(dir, fmt.Sprintf("%s_tool.go", strings.ToLower(funcName)))

	// Create the private struct type name
	lowerFuncName := strings.ToLower(funcName[:1]) + funcName[1:]
	structTypeName := fmt.Sprintf("%sToolDefType", lowerFuncName)

	// Use backticks for the JSON string for better readability
	// Only fall back to strconv.Quote if the JSON contains backticks
	jsonString := string(jsonBytes)
	if strings.Contains(jsonString, "`") {
		// Rare case: JSON contains backticks, use escaped quotes
		jsonString = strconv.Quote(jsonString)
	} else {
		// Normal case: use backticks for readability
		jsonString = "`" + jsonString + "`"
	}

	var content string
	if paramTypeName == "" {
		// No-argument function (only context)
		content = fmt.Sprintf(`// Code generated by funcschema. DO NOT EDIT.

package %s

import (
	"context"
	"encoding/json"

	"github.com/bpowers/go-agent/chat"
)

// %s implements chat.ToolDef for the %s function
type %s struct{}

func (%s) MCPJsonSchema() string {
	return %s
}

func (%s) Name() string {
	return %q
}

func (%s) Description() string {
	return %q
}

// %sToolDef is the MCP tool definition for the %s function
var %sToolDef chat.ToolDef = %s{}


// %sTool is a generic wrapper that accepts JSON input and returns JSON output
func %sTool(ctx context.Context, input string) string {
	// No input parameters needed, ignore input JSON
	
	// Call the actual function with context only
	result := %s(ctx)

	// Marshal the result
	respBytes, err := json.Marshal(result)
	if err != nil {
		errStr := "failed to marshal response: " + err.Error()
		errResp := %s{
			Error: &errStr,
		}
		respBytes, _ := json.Marshal(errResp)
		return string(respBytes)
	}

	return string(respBytes)
}
`, packageName,
			structTypeName, funcName, // type comment
			structTypeName,             // type declaration
			structTypeName, jsonString, // MCPJsonSchema method
			structTypeName, tool.Name, // Name method
			structTypeName, tool.Description, // Description method
			funcName, funcName, funcName, structTypeName, // var declaration
			funcName, funcName, funcName, returnTypeName) // Tool function
	} else {
		// Function with arguments
		content = fmt.Sprintf(`// Code generated by funcschema. DO NOT EDIT.

package %s

import (
	"context"
	"encoding/json"

	"github.com/bpowers/go-agent/chat"
)

// %s implements chat.ToolDef for the %s function
type %s struct{}

func (%s) MCPJsonSchema() string {
	return %s
}

func (%s) Name() string {
	return %q
}

func (%s) Description() string {
	return %q
}

// %sToolDef is the MCP tool definition for the %s function
var %sToolDef chat.ToolDef = %s{}


// %sTool is a generic wrapper that accepts JSON input and returns JSON output
func %sTool(ctx context.Context, input string) string {
	// Parse the input JSON
	var req %s
	if err := json.Unmarshal([]byte(input), &req); err != nil {
		errStr := "failed to parse input: " + err.Error()
		errResp := %s{
			Error: &errStr,
		}
		respBytes, _ := json.Marshal(errResp)
		return string(respBytes)
	}

	// Call the actual function with context
	result := %s(ctx, req)

	// Marshal the result
	respBytes, err := json.Marshal(result)
	if err != nil {
		errStr := "failed to marshal response: " + err.Error()
		errResp := %s{
			Error: &errStr,
		}
		respBytes, _ := json.Marshal(errResp)
		return string(respBytes)
	}

	return string(respBytes)
}
`, packageName,
			structTypeName, funcName, // type comment
			structTypeName,             // type declaration
			structTypeName, jsonString, // MCPJsonSchema method
			structTypeName, tool.Name, // Name method
			structTypeName, tool.Description, // Description method
			funcName, funcName, funcName, structTypeName, // var declaration
			funcName, funcName, paramTypeName, returnTypeName, // Tool function
			funcName, returnTypeName)
	}

	// Format the generated code with gofumpt
	formatted, err := format.Source([]byte(content), format.Options{})
	if err != nil {
		// If formatting fails, fall back to unformatted code
		log.Printf("Warning: gofumpt formatting failed: %v", err)
		formatted = []byte(content)
	}

	// Write the file
	if err := os.WriteFile(outputFile, formatted, 0o644); err != nil {
		return fmt.Errorf("writing output file: %w", err)
	}

	return nil
}

func isContextParam(param *ast.Field) bool {
	// Check if the parameter type is context.Context
	selectorExpr, ok := param.Type.(*ast.SelectorExpr)
	if !ok {
		return false
	}

	// Check if it's context.Context
	if ident, ok := selectorExpr.X.(*ast.Ident); ok {
		return ident.Name == "context" && selectorExpr.Sel.Name == "Context"
	}
	return false
}

func isStructType(typeName string, file *ast.File) bool {
	var found bool
	ast.Inspect(file, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == typeName {
			_, isStruct := ts.Type.(*ast.StructType)
			found = isStruct
			return false
		}
		return true
	})
	return found
}

func hasErrorField(expr ast.Expr, file *ast.File) bool {
	switch t := expr.(type) {
	case *ast.Ident:
		// Named type - need to find its definition
		var typeSpec *ast.TypeSpec
		ast.Inspect(file, func(n ast.Node) bool {
			if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == t.Name {
				typeSpec = ts
				return false
			}
			return true
		})
		if typeSpec == nil {
			return false
		}
		structType, ok := typeSpec.Type.(*ast.StructType)
		if !ok {
			return false
		}
		return hasErrorFieldInStruct(structType)
	case *ast.StructType:
		// Inline struct
		return hasErrorFieldInStruct(t)
	default:
		return false
	}
}

func hasErrorFieldInStruct(structType *ast.StructType) bool {
	for _, field := range structType.Fields.List {
		for _, name := range field.Names {
			if name.Name == "Error" {
				// Check if it's a pointer to string or *string
				switch ft := field.Type.(type) {
				case *ast.StarExpr:
					if ident, ok := ft.X.(*ast.Ident); ok && ident.Name == "string" {
						return true
					}
				}
			}
		}
	}
	return false
}

func getTypeName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StructType:
		// For inline structs, we'll use a generic name
		// In practice, users should use named types
		return "struct{}"
	case *ast.StarExpr:
		// Pointer type
		return "*" + getTypeName(t.X)
	case *ast.ArrayType:
		// Array type
		return "[]" + getTypeName(t.Elt)
	case *ast.MapType:
		// Map type
		return "map[" + getTypeName(t.Key) + "]" + getTypeName(t.Value)
	default:
		return "interface{}"
	}
}

func boolPtr(b bool) *bool {
	return &b
}

func extractDescriptionFromAST(fn *ast.FuncDecl) string {
	if fn.Doc == nil || len(fn.Doc.List) == 0 {
		return fmt.Sprintf("Function %s", fn.Name.Name)
	}

	// Get the whole doc comment
	var docLines []string
	for _, comment := range fn.Doc.List {
		text := comment.Text
		// Remove the comment prefix (// or /*)
		if strings.HasPrefix(text, "//") {
			text = strings.TrimSpace(strings.TrimPrefix(text, "//"))
		} else if strings.HasPrefix(text, "/*") {
			text = strings.TrimSpace(strings.TrimPrefix(text, "/*"))
			text = strings.TrimSuffix(text, "*/")
		}
		if text != "" {
			docLines = append(docLines, text)
		}
	}

	if len(docLines) == 0 {
		return fmt.Sprintf("Function %s", fn.Name.Name)
	}

	// Join all lines
	fullDoc := strings.Join(docLines, " ")

	// If the comment starts with the function name, remove it and capitalize the next word
	if strings.HasPrefix(fullDoc, fn.Name.Name+" ") {
		remainder := strings.TrimPrefix(fullDoc, fn.Name.Name+" ")
		if len(remainder) > 0 {
			// Capitalize the first letter
			fullDoc = strings.ToUpper(remainder[:1]) + remainder[1:]
		}
	} else if fullDoc == "" || fullDoc == fn.Name.Name {
		// If just the function name or empty, use default
		return fmt.Sprintf("Function %s", fn.Name.Name)
	}

	return fullDoc
}

func extractFieldDescription(commentGroup *ast.CommentGroup) string {
	if commentGroup == nil || len(commentGroup.List) == 0 {
		return ""
	}

	var docLines []string
	for _, comment := range commentGroup.List {
		text := comment.Text
		// Remove the comment prefix (// or /*)
		if strings.HasPrefix(text, "//") {
			text = strings.TrimSpace(strings.TrimPrefix(text, "//"))
		} else if strings.HasPrefix(text, "/*") {
			text = strings.TrimSpace(strings.TrimPrefix(text, "/*"))
			text = strings.TrimSuffix(text, "*/")
		}
		if text != "" {
			docLines = append(docLines, text)
		}
	}

	if len(docLines) == 0 {
		return ""
	}

	// Join all lines with a space
	return strings.Join(docLines, " ")
}
