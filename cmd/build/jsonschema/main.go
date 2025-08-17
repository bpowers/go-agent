package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/bpowers/go-agent/schema"
)

var (
	typeName   = flag.String("type", "", "Name of the type to generate schema for (required)")
	inputFile  = flag.String("input", "", "Input Go source file (required)")
	outputJSON = flag.String("json", "", "Output JSON schema file (required)")
	outputGo   = flag.String("go", "", "Output Go file with embedded schema (required)")
	pkgName    = flag.String("package", "", "Package name for generated Go file (defaults to directory name)")
)

func main() {
	flag.Parse()

	if *typeName == "" || *inputFile == "" || *outputJSON == "" || *outputGo == "" {
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

	// Find the target type
	var targetType *ast.TypeSpec
	ast.Inspect(node, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == *typeName {
			targetType = ts
			return false
		}
		return true
	})

	if targetType == nil {
		return fmt.Errorf("type %s not found in %s", *typeName, *inputFile)
	}

	// Generate the JSON schema
	schemaObj, err := generateSchema(targetType, node)
	if err != nil {
		return fmt.Errorf("generating schema: %w", err)
	}

	// Write JSON schema file
	jsonBytes, err := json.MarshalIndent(schemaObj, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling schema: %w", err)
	}

	if err := os.WriteFile(*outputJSON, jsonBytes, 0o644); err != nil {
		return fmt.Errorf("writing JSON file: %w", err)
	}

	// Generate Go file with embedded schema
	pkg := *pkgName
	if pkg == "" {
		pkg = filepath.Base(filepath.Dir(*outputGo))
	}

	if err := generateGoFile(pkg, *typeName, *outputJSON, schemaObj, *outputGo); err != nil {
		return fmt.Errorf("generating Go file: %w", err)
	}

	return nil
}

func generateSchema(typeSpec *ast.TypeSpec, file *ast.File) (*schema.JSON, error) {
	structType, ok := typeSpec.Type.(*ast.StructType)
	if !ok {
		return nil, fmt.Errorf("type %s is not a struct", typeSpec.Name.Name)
	}

	s := &schema.JSON{
		Schema:               schema.URL,
		Type:                 schema.Object,
		Description:          fmt.Sprintf("JSON schema for %s", typeSpec.Name.Name),
		Properties:           make(map[string]*schema.JSON),
		AdditionalProperties: boolPtr(false),
	}

	var required []string

	for _, field := range structType.Fields.List {
		if len(field.Names) == 0 {
			continue // Skip embedded fields for now
		}

		fieldName := field.Names[0].Name
		if !ast.IsExported(fieldName) {
			continue // Skip unexported fields
		}

		// Get JSON tag
		jsonName, omitempty := parseJSONTag(field.Tag)
		if jsonName == "-" {
			continue // Skip fields with json:"-"
		}
		if jsonName == "" {
			jsonName = fieldName
		}

		// Generate schema for field
		fieldSchema, err := generateFieldSchema(field.Type, file)
		if err != nil {
			return nil, fmt.Errorf("generating schema for field %s: %w", fieldName, err)
		}

		s.Properties[jsonName] = fieldSchema

		// Add to required unless it has omitempty or omitzero
		if !omitempty {
			required = append(required, jsonName)
		}
	}

	if len(required) > 0 {
		s.Required = required
	}

	return s, nil
}

func generateFieldSchema(expr ast.Expr, file *ast.File) (*schema.JSON, error) {
	switch t := expr.(type) {
	case *ast.Ident:
		return generateIdentSchema(t.Name)
	case *ast.ArrayType:
		itemSchema, err := generateFieldSchema(t.Elt, file)
		if err != nil {
			return nil, err
		}
		return &schema.JSON{
			Type:  schema.Array,
			Items: itemSchema,
		}, nil
	case *ast.StarExpr:
		// Pointer type - generate schema for the underlying type
		return generateFieldSchema(t.X, file)
	case *ast.SelectorExpr:
		// Qualified identifier (e.g., pkg.Type)
		if ident, ok := t.X.(*ast.Ident); ok {
			return generateIdentSchema(fmt.Sprintf("%s.%s", ident.Name, t.Sel.Name))
		}
		return nil, fmt.Errorf("unsupported selector expression")
	case *ast.StructType:
		// Inline struct
		s := &schema.JSON{
			Type:                 schema.Object,
			Properties:           make(map[string]*schema.JSON),
			AdditionalProperties: boolPtr(false),
		}

		var required []string
		for _, field := range t.Fields.List {
			if len(field.Names) == 0 {
				continue
			}

			fieldName := field.Names[0].Name
			if !ast.IsExported(fieldName) {
				continue
			}

			jsonName, omitempty := parseJSONTag(field.Tag)
			if jsonName == "-" {
				continue
			}
			if jsonName == "" {
				jsonName = fieldName
			}

			fieldSchema, err := generateFieldSchema(field.Type, file)
			if err != nil {
				return nil, err
			}

			s.Properties[jsonName] = fieldSchema

			if !omitempty {
				required = append(required, jsonName)
			}
		}

		if len(required) > 0 {
			s.Required = required
		}

		return s, nil
	default:
		return nil, fmt.Errorf("unsupported type: %T", expr)
	}
}

func generateIdentSchema(typeName string) (*schema.JSON, error) {
	switch typeName {
	case "string":
		return &schema.JSON{Type: schema.String}, nil
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64":
		return &schema.JSON{Type: schema.Type("integer")}, nil
	case "float32", "float64":
		return &schema.JSON{Type: schema.Type("number")}, nil
	case "bool":
		return &schema.JSON{Type: schema.Type("boolean")}, nil
	case "VariableType":
		// Custom enum type from sdjson
		return &schema.JSON{
			Type: schema.String,
			Enum: []string{"variable", "stock", "flow"},
		}, nil
	case "Polarity":
		// Custom enum type from sdjson
		return &schema.JSON{
			Type: schema.String,
			Enum: []string{"+", "-"},
		}, nil
	case "Variable", "Relationship", "Specs", "Point", "GraphicalFunction":
		// These are other types in the sdjson package
		// For now, we'll treat them as objects
		return &schema.JSON{Type: schema.Object}, nil
	default:
		// Unknown type - treat as object
		return &schema.JSON{Type: schema.Object}, nil
	}
}

func parseJSONTag(tag *ast.BasicLit) (name string, omitempty bool) {
	if tag == nil {
		return "", false
	}

	// Remove quotes and backticks
	tagValue := strings.Trim(tag.Value, "`")

	// Find json tag
	tags := strings.Fields(tagValue)
	for _, t := range tags {
		if strings.HasPrefix(t, "json:") {
			jsonTag := strings.TrimPrefix(t, "json:")
			jsonTag = strings.Trim(jsonTag, `"`)

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
	}

	return "", false
}

func generateGoFile(pkg, typeName, jsonFile string, schemaObj *schema.JSON, outputFile string) error {
	// Template for the generated Go file
	const goTemplate = `// Code generated by jsonschema generator. DO NOT EDIT.

package {{.Package}}

import (
	_ "embed"
	"encoding/json"

	"github.com/bpowers/go-agent/schema"
)

//go:embed {{.JSONFile}}
var {{.VarName}}JSON string

// {{.VarName}}Schema is the JSON schema for {{.TypeName}}
var {{.VarName}}Schema = func() *schema.JSON {
	var s schema.JSON
	if err := json.Unmarshal([]byte({{.VarName}}JSON), &s); err != nil {
		panic("failed to unmarshal embedded schema: " + err.Error())
	}
	return &s
}()
`

	tmpl, err := template.New("go").Parse(goTemplate)
	if err != nil {
		return err
	}

	// Generate variable name (lowercase first letter)
	varName := strings.ToLower(typeName[:1]) + typeName[1:]

	data := struct {
		Package  string
		TypeName string
		VarName  string
		JSONFile string
	}{
		Package:  pkg,
		TypeName: typeName,
		VarName:  varName,
		JSONFile: filepath.Base(jsonFile),
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return err
	}

	return os.WriteFile(outputFile, buf.Bytes(), 0o644)
}

func boolPtr(b bool) *bool {
	return &b
}
