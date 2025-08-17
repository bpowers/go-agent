package main

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/schema"
)

func TestParseJSONTag(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name         string
		tag          string
		expectedName string
		expectedOmit bool
	}{
		{
			name:         "Simple tag",
			tag:          "`json:\"name\"`",
			expectedName: "name",
			expectedOmit: false,
		},
		{
			name:         "Tag with omitempty",
			tag:          "`json:\"name,omitempty\"`",
			expectedName: "name",
			expectedOmit: true,
		},
		{
			name:         "Tag with omitzero",
			tag:          "`json:\"name,omitzero\"`",
			expectedName: "name",
			expectedOmit: true,
		},
		{
			name:         "Ignore tag",
			tag:          "`json:\"-\"`",
			expectedName: "-",
			expectedOmit: false,
		},
		{
			name:         "Empty tag",
			tag:          "`json:\"\"`",
			expectedName: "",
			expectedOmit: false,
		},
		{
			name:         "No JSON tag",
			tag:          "`xml:\"name\"`",
			expectedName: "",
			expectedOmit: false,
		},
		{
			name:         "Multiple tags",
			tag:          "`json:\"name,omitempty\" xml:\"xmlname\"`",
			expectedName: "name",
			expectedOmit: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			lit := &ast.BasicLit{
				Kind:  token.STRING,
				Value: tt.tag,
			}

			name, omit := parseJSONTag(lit)
			assert.Equal(t, tt.expectedName, name)
			assert.Equal(t, tt.expectedOmit, omit)
		})
	}
}

func TestParseJSONTagNil(t *testing.T) {
	t.Parallel()
	name, omit := parseJSONTag(nil)
	assert.Equal(t, "", name)
	assert.Equal(t, false, omit)
}

func TestGenerateIdentSchema(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		typeName string
		expected *schema.JSON
	}{
		{
			name:     "String type",
			typeName: "string",
			expected: &schema.JSON{Type: schema.String},
		},
		{
			name:     "Int type",
			typeName: "int",
			expected: &schema.JSON{Type: schema.Type("integer")},
		},
		{
			name:     "Int64 type",
			typeName: "int64",
			expected: &schema.JSON{Type: schema.Type("integer")},
		},
		{
			name:     "Float64 type",
			typeName: "float64",
			expected: &schema.JSON{Type: schema.Type("number")},
		},
		{
			name:     "Bool type",
			typeName: "bool",
			expected: &schema.JSON{Type: schema.Type("boolean")},
		},
		{
			name:     "VariableType enum",
			typeName: "VariableType",
			expected: &schema.JSON{
				Type: schema.String,
				Enum: []string{"variable", "stock", "flow"},
			},
		},
		{
			name:     "Polarity enum",
			typeName: "Polarity",
			expected: &schema.JSON{
				Type: schema.String,
				Enum: []string{"+", "-"},
			},
		},
		{
			name:     "Unknown type",
			typeName: "CustomType",
			expected: &schema.JSON{Type: schema.Object},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result, err := generateIdentSchema(tt.typeName)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGenerateFieldSchema(t *testing.T) {
	t.Parallel()
	// Test array type
	t.Run("Array of strings", func(t *testing.T) {
		t.Parallel()
		src := `package test
		type Test struct {
			Items []string
		}`

		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "", src, 0)
		require.NoError(t, err)

		// Find the field type
		var fieldType ast.Expr
		ast.Inspect(file, func(n ast.Node) bool {
			if field, ok := n.(*ast.Field); ok && len(field.Names) > 0 && field.Names[0].Name == "Items" {
				fieldType = field.Type
				return false
			}
			return true
		})

		require.NotNil(t, fieldType)

		result, err := generateFieldSchema(fieldType, file)
		require.NoError(t, err)

		expected := &schema.JSON{
			Type:  schema.Array,
			Items: &schema.JSON{Type: schema.String},
		}
		assert.Equal(t, expected, result)
	})

	// Test pointer type
	t.Run("Pointer to string", func(t *testing.T) {
		t.Parallel()
		src := `package test
		type Test struct {
			Name *string
		}`

		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "", src, 0)
		require.NoError(t, err)

		// Find the field type
		var fieldType ast.Expr
		ast.Inspect(file, func(n ast.Node) bool {
			if field, ok := n.(*ast.Field); ok && len(field.Names) > 0 && field.Names[0].Name == "Name" {
				fieldType = field.Type
				return false
			}
			return true
		})

		require.NotNil(t, fieldType)

		result, err := generateFieldSchema(fieldType, file)
		require.NoError(t, err)

		expected := &schema.JSON{Type: schema.String}
		assert.Equal(t, expected, result)
	})
}

func TestGenerateSchema(t *testing.T) {
	t.Parallel()
	src := `package test

type TestStruct struct {
	Name        string    ` + "`json:\"name\"`" + `
	Age         int       ` + "`json:\"age,omitempty\"`" + `
	Active      bool      ` + "`json:\"active\"`" + `
	Tags        []string  ` + "`json:\"tags,omitempty\"`" + `
	ignored     string    // unexported
	IgnoreField string    ` + "`json:\"-\"`" + `
}
`

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	require.NoError(t, err)

	// Find the TestStruct type
	var typeSpec *ast.TypeSpec
	ast.Inspect(file, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == "TestStruct" {
			typeSpec = ts
			return false
		}
		return true
	})

	require.NotNil(t, typeSpec)

	result, err := generateSchema(typeSpec, file)
	require.NoError(t, err)

	// Verify the schema
	assert.Equal(t, schema.URL, result.Schema)
	assert.Equal(t, schema.Object, result.Type)
	assert.Contains(t, result.Description, "TestStruct")
	assert.NotNil(t, result.AdditionalProperties)
	assert.False(t, *result.AdditionalProperties)

	// Check properties
	assert.Len(t, result.Properties, 4)
	assert.Contains(t, result.Properties, "name")
	assert.Contains(t, result.Properties, "age")
	assert.Contains(t, result.Properties, "active")
	assert.Contains(t, result.Properties, "tags")

	// Check required fields (only those without omitempty)
	assert.Len(t, result.Required, 2)
	assert.Contains(t, result.Required, "name")
	assert.Contains(t, result.Required, "active")

	// Verify property types
	assert.Equal(t, schema.String, result.Properties["name"].Type)
	assert.Equal(t, schema.Type("integer"), result.Properties["age"].Type)
	assert.Equal(t, schema.Type("boolean"), result.Properties["active"].Type)
	assert.Equal(t, schema.Array, result.Properties["tags"].Type)
	assert.Equal(t, schema.String, result.Properties["tags"].Items.Type)
}

func TestGenerateSchemaWithInlineStruct(t *testing.T) {
	t.Parallel()
	src := `package test

type TestStruct struct {
	Name string ` + "`json:\"name\"`" + `
	Meta struct {
		Created string ` + "`json:\"created\"`" + `
		Updated string ` + "`json:\"updated,omitempty\"`" + `
	} ` + "`json:\"meta\"`" + `
}
`

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	require.NoError(t, err)

	// Find the TestStruct type
	var typeSpec *ast.TypeSpec
	ast.Inspect(file, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == "TestStruct" {
			typeSpec = ts
			return false
		}
		return true
	})

	require.NotNil(t, typeSpec)

	result, err := generateSchema(typeSpec, file)
	require.NoError(t, err)

	// Check that we have the meta property as an object
	assert.Contains(t, result.Properties, "meta")
	metaSchema := result.Properties["meta"]
	assert.Equal(t, schema.Object, metaSchema.Type)
	assert.NotNil(t, metaSchema.Properties)
	assert.Contains(t, metaSchema.Properties, "created")
	assert.Contains(t, metaSchema.Properties, "updated")

	// Check required fields in nested struct
	assert.Contains(t, metaSchema.Required, "created")
	assert.NotContains(t, metaSchema.Required, "updated")
}

func TestGenerateGoFile(t *testing.T) {
	t.Parallel()
	// Create a temporary directory for test files
	tmpDir := t.TempDir()

	jsonFile := filepath.Join(tmpDir, "test.json")
	goFile := filepath.Join(tmpDir, "test_schema.go")

	// Create a dummy JSON file
	err := os.WriteFile(jsonFile, []byte(`{"type":"object"}`), 0o644)
	require.NoError(t, err)

	// Create a dummy schema
	schemaObj := &schema.JSON{
		Type: schema.Object,
	}

	// Generate the Go file
	err = generateGoFile("testpkg", "TestType", jsonFile, schemaObj, goFile)
	require.NoError(t, err)

	// Read and verify the generated file
	content, err := os.ReadFile(goFile)
	require.NoError(t, err)

	contentStr := string(content)

	// Check that it contains expected elements
	assert.Contains(t, contentStr, "package testpkg")
	assert.Contains(t, contentStr, "//go:embed test.json")
	assert.Contains(t, contentStr, "var testTypeJSON string")
	assert.Contains(t, contentStr, "var testTypeSchema = func() *schema.JSON")
	assert.Contains(t, contentStr, "import (")
	assert.Contains(t, contentStr, `_ "embed"`)
	assert.Contains(t, contentStr, `"github.com/bpowers/go-agent/schema"`)
	assert.Contains(t, contentStr, "// Code generated by jsonschema generator. DO NOT EDIT.")
}

func TestBoolPtr(t *testing.T) {
	t.Parallel()
	truePtr := boolPtr(true)
	require.NotNil(t, truePtr)
	assert.True(t, *truePtr)

	falsePtr := boolPtr(false)
	require.NotNil(t, falsePtr)
	assert.False(t, *falsePtr)
}

func TestGenerateSchemaErrors(t *testing.T) {
	t.Parallel()
	t.Run("Non-struct type", func(t *testing.T) {
		t.Parallel()
		src := `package test
		type TestType int`

		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "", src, 0)
		require.NoError(t, err)

		var typeSpec *ast.TypeSpec
		ast.Inspect(file, func(n ast.Node) bool {
			if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == "TestType" {
				typeSpec = ts
				return false
			}
			return true
		})

		require.NotNil(t, typeSpec)

		_, err = generateSchema(typeSpec, file)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not a struct")
	})
}

func TestMainFlags(t *testing.T) {
	t.Parallel()
	// This test verifies that the flags are properly defined
	// We can't easily test the main function itself without refactoring
	assert.NotNil(t, typeName)
	assert.NotNil(t, inputFile)
	assert.NotNil(t, outputJSON)
	assert.NotNil(t, outputGo)
	assert.NotNil(t, pkgName)
}

// Integration test with real sdjson Model struct
func TestGenerateSchemaForSDJSONModel(t *testing.T) {
	t.Parallel()
	src := `package sdjson

type Point struct {
	X float64 ` + "`json:\"x\"`" + `
	Y float64 ` + "`json:\"y\"`" + `
}

type GraphicalFunction struct {
	Points []Point ` + "`json:\"points\"`" + `
}

type Variable struct {
	Name              string             ` + "`json:\"name\"`" + `
	Type              VariableType       ` + "`json:\"type\"`" + `
	Equation          string             ` + "`json:\"equation,omitzero\"`" + `
	Documentation     string             ` + "`json:\"documentation,omitzero\"`" + `
	Units             string             ` + "`json:\"units,omitzero\"`" + `
	Inflows           []string           ` + "`json:\"inflows,omitzero\"`" + `
	Outflows          []string           ` + "`json:\"outflows,omitzero\"`" + `
	GraphicalFunction *GraphicalFunction ` + "`json:\"graphicalFunction,omitzero\"`" + `
}

type Relationship struct {
	From              string ` + "`json:\"from\"`" + `
	To                string ` + "`json:\"to\"`" + `
	Polarity          string ` + "`json:\"polarity\"`" + `
	Reasoning         string ` + "`json:\"reasoning,omitzero\"`" + `
	PolarityReasoning string ` + "`json:\"polarityReasoning,omitzero\"`" + `
}

type Specs struct {
	StartTime float64 ` + "`json:\"startTime\"`" + `
	StopTime  float64 ` + "`json:\"stopTime\"`" + `
	DT        float64 ` + "`json:\"dt,omitzero\"`" + `
	SaveStep  float64 ` + "`json:\"saveStep,omitzero\"`" + `
	TimeUnits string  ` + "`json:\"timeUnits,omitzero\"`" + `
}

type Model struct {
	Variables     []Variable     ` + "`json:\"variables,omitzero\"`" + `
	Relationships []Relationship ` + "`json:\"relationships,omitzero\"`" + `
	Specs         Specs          ` + "`json:\"specs,omitzero\"`" + `
}
`

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	require.NoError(t, err)

	// Find the Model type
	var typeSpec *ast.TypeSpec
	ast.Inspect(file, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == "Model" {
			typeSpec = ts
			return false
		}
		return true
	})

	require.NotNil(t, typeSpec)

	result, err := generateSchema(typeSpec, file)
	require.NoError(t, err)

	// Verify the top-level schema
	assert.Equal(t, schema.URL, result.Schema)
	assert.Equal(t, schema.Object, result.Type)
	assert.Contains(t, result.Description, "Model")

	// Check properties
	assert.Contains(t, result.Properties, "variables")
	assert.Contains(t, result.Properties, "relationships")
	assert.Contains(t, result.Properties, "specs")

	// All fields have omitzero/omitempty, so nothing should be required
	assert.Empty(t, result.Required)

	// Check array types
	assert.Equal(t, schema.Array, result.Properties["variables"].Type)
	assert.Equal(t, schema.Array, result.Properties["relationships"].Type)
	assert.Equal(t, schema.Object, result.Properties["specs"].Type)

	// Verify it can be marshaled to JSON
	jsonBytes, err := json.MarshalIndent(result, "", "  ")
	require.NoError(t, err)
	assert.Contains(t, string(jsonBytes), `"$schema"`)
	assert.Contains(t, string(jsonBytes), `"type": "object"`)
}
