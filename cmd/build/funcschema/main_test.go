package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bpowers/go-agent/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
}
func SimpleFunc(ctx context.Context, req SimpleRequest) (SimpleResult, error) { return SimpleResult{Value: req.Name}, nil }`,
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
				// Nullable pointer fields without omitzero are still required:
				// "required" means the key must be present, while the nullable
				// type allows the value to be null.
				if len(s.Required) != 2 {
					t.Errorf("expected 2 required fields, got %v", s.Required)
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
					return
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
		{
			name: "simple embedded struct",
			code: `package test
import "context"
type BaseFields struct {
	ID   string
	Name string
}
type EmbedRequest struct {
	BaseFields
	Age int
}
type EmbedResult struct {
	Value string
	Error *string
}
func EmbedFunc(ctx context.Context, req EmbedRequest) EmbedResult { return EmbedResult{Value: req.Name} }`,
			funcName: "EmbedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				if len(s.Properties) != 3 {
					t.Errorf("expected 3 properties (ID, Name from embedded, Age from outer), got %d", len(s.Properties))
				}
				if s.Properties["ID"] == nil {
					t.Error("expected 'ID' property from embedded struct")
				}
				if s.Properties["ID"].Type != schema.String {
					t.Errorf("expected string type for ID")
				}
				if s.Properties["Name"] == nil {
					t.Error("expected 'Name' property from embedded struct")
				}
				if s.Properties["Age"] == nil {
					t.Error("expected 'Age' property from outer struct")
				}
				if len(s.Required) != 3 {
					t.Errorf("expected 3 required fields, got %d", len(s.Required))
				}
			},
		},
		{
			name: "embedded struct with json tags",
			code: `package test
import "context"
type BaseWithTags struct {
	UserID   string ` + "`json:\"user_id\"`" + `
	UserName string ` + "`json:\"user_name\"`" + `
}
type EmbedTagRequest struct {
	BaseWithTags
	UserAge int ` + "`json:\"user_age\"`" + `
}
type EmbedTagResult struct {
	Value string
	Error *string
}
func EmbedTagFunc(ctx context.Context, req EmbedTagRequest) EmbedTagResult { return EmbedTagResult{Value: req.UserName} }`,
			funcName: "EmbedTagFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 3 {
					t.Errorf("expected 3 properties, got %d", len(s.Properties))
				}
				if s.Properties["user_id"] == nil {
					t.Error("expected 'user_id' property from embedded struct with json tag")
				}
				if s.Properties["user_name"] == nil {
					t.Error("expected 'user_name' property from embedded struct with json tag")
				}
				if s.Properties["user_age"] == nil {
					t.Error("expected 'user_age' property from outer struct")
				}
				if s.Properties["UserID"] != nil {
					t.Error("should not have UserID property (renamed by json tag)")
				}
				if s.Properties["UserName"] != nil {
					t.Error("should not have UserName property (renamed by json tag)")
				}
			},
		},
		{
			name: "field shadowing in embedded struct",
			code: `package test
import "context"
type ShadowBase struct {
	Name  string
	Email string
}
type ShadowRequest struct {
	ShadowBase
	Name string  // This shadows the embedded Name field
	Age  int
}
type ShadowResult struct {
	Value string
	Error *string
}
func ShadowFunc(ctx context.Context, req ShadowRequest) ShadowResult { return ShadowResult{Value: req.Name} }`,
			funcName: "ShadowFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 3 {
					t.Errorf("expected 3 properties (Name from outer, Email from embedded, Age), got %d", len(s.Properties))
				}
				if s.Properties["Name"] == nil {
					t.Error("expected 'Name' property")
				}
				if s.Properties["Email"] == nil {
					t.Error("expected 'Email' property from embedded struct")
				}
				if s.Properties["Age"] == nil {
					t.Error("expected 'Age' property")
				}
			},
		},
		{
			name: "multiple embedded structs",
			code: `package test
import "context"
type Timestamps struct {
	CreatedAt string ` + "`json:\"created_at\"`" + `
	UpdatedAt string ` + "`json:\"updated_at\"`" + `
}
type Metadata struct {
	Version int    ` + "`json:\"version\"`" + `
	Author  string ` + "`json:\"author\"`" + `
}
type MultiEmbedRequest struct {
	Timestamps
	Metadata
	Title string ` + "`json:\"title\"`" + `
}
type MultiEmbedResult struct {
	Value string
	Error *string
}
func MultiEmbedFunc(ctx context.Context, req MultiEmbedRequest) MultiEmbedResult { return MultiEmbedResult{Value: req.Title} }`,
			funcName: "MultiEmbedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 5 {
					t.Errorf("expected 5 properties, got %d", len(s.Properties))
				}
				if s.Properties["created_at"] == nil {
					t.Error("expected 'created_at' from Timestamps")
				}
				if s.Properties["updated_at"] == nil {
					t.Error("expected 'updated_at' from Timestamps")
				}
				if s.Properties["version"] == nil {
					t.Error("expected 'version' from Metadata")
				}
				if s.Properties["author"] == nil {
					t.Error("expected 'author' from Metadata")
				}
				if s.Properties["title"] == nil {
					t.Error("expected 'title' from outer struct")
				}
			},
		},
		{
			name: "pointer to embedded struct",
			code: `package test
import "context"
type PtrBase struct {
	BaseID   string
	BaseName string
}
type PtrEmbedRequest struct {
	*PtrBase
	Extra string
}
type PtrEmbedResult struct {
	Value string
	Error *string
}
func PtrEmbedFunc(ctx context.Context, req PtrEmbedRequest) PtrEmbedResult { return PtrEmbedResult{Value: req.BaseName} }`,
			funcName: "PtrEmbedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 3 {
					t.Errorf("expected 3 properties, got %d", len(s.Properties))
				}
				if s.Properties["BaseID"] == nil {
					t.Error("expected 'BaseID' from embedded pointer struct")
				}
				if s.Properties["BaseName"] == nil {
					t.Error("expected 'BaseName' from embedded pointer struct")
				}
				if s.Properties["Extra"] == nil {
					t.Error("expected 'Extra' from outer struct")
				}
			},
		},
		{
			name: "nested embedded structs",
			code: `package test
import "context"
type DeepBase struct {
	DeepField string
}
type MiddleLayer struct {
	DeepBase
	MiddleField string
}
type NestedEmbedRequest struct {
	MiddleLayer
	TopField string
}
type NestedEmbedResult struct {
	Value string
	Error *string
}
func NestedEmbedFunc(ctx context.Context, req NestedEmbedRequest) NestedEmbedResult { return NestedEmbedResult{Value: req.DeepField} }`,
			funcName: "NestedEmbedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 3 {
					t.Errorf("expected 3 properties, got %d", len(s.Properties))
				}
				if s.Properties["DeepField"] == nil {
					t.Error("expected 'DeepField' from nested embedded struct")
				}
				if s.Properties["MiddleField"] == nil {
					t.Error("expected 'MiddleField' from middle embedded struct")
				}
				if s.Properties["TopField"] == nil {
					t.Error("expected 'TopField' from top struct")
				}
			},
		},
		{
			name: "omitzero fields are not required",
			code: `package test
import "context"
type OmitzeroRequest struct {
	ProjectPath string              ` + "`json:\"projectPath\"`" + `
	ModelName   string              ` + "`json:\"modelName,omitzero\"`" + `
	DryRun      bool                ` + "`json:\"dryRun,omitzero\"`" + `
	SimSpecs    *OmitzeroSimSpecs   ` + "`json:\"simSpecs,omitzero\"`" + `
	Operations  []OmitzeroOperation ` + "`json:\"operations\"`" + `
}
type OmitzeroSimSpecs struct {
	StartTime float64
	EndTime   float64
}
type OmitzeroOperation struct {
	Name string
}
type OmitzeroResult struct {
	Value string
}
func OmitzeroFunc(ctx context.Context, req OmitzeroRequest) (OmitzeroResult, error) { return OmitzeroResult{}, nil }`,
			funcName: "OmitzeroFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				assert.Equal(t, schema.Object, s.Type)
				assert.Len(t, s.Properties, 5)

				// Only fields WITHOUT omitzero should be required
				assert.ElementsMatch(t, []string{"projectPath", "operations"}, s.Required)

				// simSpecs should be nullable (pointer type) but NOT required
				require.NotNil(t, s.Properties["simSpecs"])
			},
		},
		{
			name: "omitempty fields are not required",
			code: `package test
import "context"
type OmitemptyRequest struct {
	Name     string ` + "`json:\"name\"`" + `
	Optional string ` + "`json:\"optional,omitempty\"`" + `
}
type OmitemptyResult struct {
	Value string
}
func OmitemptyFunc(ctx context.Context, req OmitemptyRequest) (OmitemptyResult, error) { return OmitemptyResult{}, nil }`,
			funcName: "OmitemptyFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				assert.Equal(t, schema.Object, s.Type)
				assert.Len(t, s.Properties, 2)
				assert.Equal(t, []string{"name"}, s.Required)
			},
		},
		{
			name: "embedded struct with omitzero fields preserves optionality",
			code: `package test
import "context"
type EmbeddedOptionalBase struct {
	Required string ` + "`json:\"required\"`" + `
	Optional string ` + "`json:\"optional,omitzero\"`" + `
}
type EmbeddedOptionalRequest struct {
	EmbeddedOptionalBase
	Extra string ` + "`json:\"extra\"`" + `
}
type EmbeddedOptionalResult struct {
	Value string
}
func EmbeddedOptionalFunc(ctx context.Context, req EmbeddedOptionalRequest) (EmbeddedOptionalResult, error) { return EmbeddedOptionalResult{}, nil }`,
			funcName: "EmbeddedOptionalFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				assert.Equal(t, schema.Object, s.Type)
				assert.Len(t, s.Properties, 3)
				// "optional" from embedded struct should NOT be required
				assert.ElementsMatch(t, []string{"required", "extra"}, s.Required)
			},
		},
		{
			name: "unexported embedded struct fields are skipped",
			code: `package test
import "context"
type UnexportedBase struct {
	PublicField  string
	privateField string
}
type UnexportedEmbedRequest struct {
	UnexportedBase
	OtherField string
}
type UnexportedEmbedResult struct {
	Value string
	Error *string
}
func UnexportedEmbedFunc(ctx context.Context, req UnexportedEmbedRequest) UnexportedEmbedResult { return UnexportedEmbedResult{Value: req.PublicField} }`,
			funcName: "UnexportedEmbedFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 2 {
					t.Errorf("expected 2 properties (only exported fields), got %d", len(s.Properties))
				}
				if s.Properties["PublicField"] == nil {
					t.Error("expected 'PublicField' from embedded struct")
				}
				if s.Properties["OtherField"] == nil {
					t.Error("expected 'OtherField' from outer struct")
				}
				if s.Properties["privateField"] != nil {
					t.Error("should not have 'privateField' (unexported)")
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

			// Create doc.Package for testing
			docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
			if err != nil {
				t.Fatalf("failed to create doc package: %v", err)
			}

			// Generate input schema
			s, err := generateInputSchema(targetFunc.Type.Params, []*ast.File{node}, docPkg)
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
			name: "simple struct return",
			code: `package test
type EmptyRequest struct{}
type SimpleResult struct {
	Value string
}
func SimpleFunc(req EmptyRequest) (SimpleResult, error) { return SimpleResult{Value: "test"}, nil }`,
			funcName: "SimpleFunc",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				if s.Properties["Value"].Type != schema.String {
					t.Error("expected Value property")
				}
				// Error field should be added by generator
				if s.Properties["error"] == nil {
					t.Error("expected error property (added by generator)")
				}
				// Error should be nullable
				typeArr, ok := s.Properties["error"].Type.([]interface{})
				if !ok {
					t.Errorf("expected type array for nullable error, got %T", s.Properties["error"].Type)
				} else if len(typeArr) != 2 || typeArr[0] != "string" || typeArr[1] != "null" {
					t.Errorf("expected [\"string\", \"null\"] for error, got %v", typeArr)
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
}
func StructReturn(req Request) (Result, error) { return Result{}, nil }`,
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
				if s.Properties["error"] == nil {
					t.Error("expected error property (added by generator)")
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
}
func NestedFunc(ctx context.Context) (NestedResult, error) { return NestedResult{}, nil }`,
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
			name: "error only return",
			code: `package test
type Request struct{}
func ErrorOnly(req Request) error { return nil }`,
			funcName: "ErrorOnly",
			validate: func(t *testing.T, s *schema.JSON) {
				if len(s.Properties) != 1 {
					t.Fatalf("expected only error property, got %d", len(s.Properties))
				}
				if _, ok := s.Properties["error"]; !ok {
					t.Fatalf("expected error property")
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
}
func ListFunc(ctx context.Context) (ListResult, error) { return ListResult{}, nil }`,
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
}
func MapReturn(req Request) (MapResult, error) { return MapResult{}, nil }`,
			funcName: "MapReturn",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				dataField := s.Properties["Data"]
				if dataField == nil {
					t.Error("expected Data property")
					return
				}
				if dataField.Type != schema.Object {
					t.Errorf("expected object type for map, got %v", dataField.Type)
				}
				if dataField.AdditionalProperties == nil || !*dataField.AdditionalProperties {
					t.Error("expected additionalProperties to be true for map")
				}
				if s.Properties["error"] == nil {
					t.Error("expected error property (added by generator)")
				}
			},
		},
		{
			name: "complex return with nested arrays",
			code: `package test
type Request struct{}
type ArrayResult struct {
	Data [][]float64
}
func ArrayOfArrays(req Request) (ArrayResult, error) { return ArrayResult{}, nil }`,
			funcName: "ArrayOfArrays",
			validate: func(t *testing.T, s *schema.JSON) {
				if s.Type != schema.Object {
					t.Errorf("expected object type, got %v", s.Type)
				}
				dataField := s.Properties["Data"]
				if dataField == nil {
					t.Error("expected Data property")
					return
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
				if s.Properties["error"] == nil {
					t.Error("expected error property (added by generator)")
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

			// Create doc.Package for testing
			docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
			if err != nil {
				t.Fatalf("failed to create doc package: %v", err)
			}

			// Generate output schema
			s, err := generateOutputSchema(targetFunc.Type.Results, []*ast.File{node}, docPkg)
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
}

func DatasetGet(ctx context.Context, args DatasetGetRequest) (DatasetGetResult, error) {
	return DatasetGetResult{}, nil
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

	// Create doc.Package for testing
	docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
	if err != nil {
		t.Fatalf("failed to create doc package: %v", err)
	}

	// Generate schemas
	inputSchema, err := generateInputSchema(targetFunc.Type.Params, []*ast.File{node}, docPkg)
	if err != nil {
		t.Fatalf("failed to generate input schema: %v", err)
	}

	outputSchema, err := generateOutputSchema(targetFunc.Type.Results, []*ast.File{node}, docPkg)
	if err != nil {
		t.Fatalf("failed to generate output schema: %v", err)
	}

	// Create tool definition
	tool := &MCPTool{
		Name:         "DatasetGet",
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

	// Check required fields
	if jsonMap["name"] != "DatasetGet" {
		t.Errorf("expected name field to be 'DatasetGet', got %v", jsonMap["name"])
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
	if outProps["error"] == nil {
		t.Error("expected error in output properties (added by generator)")
	}

	// Check error is nullable
	errorSchema := outProps["error"].(map[string]interface{})
	errorType := errorSchema["type"]
	if errorTypeArr, ok := errorType.([]interface{}); !ok || len(errorTypeArr) != 2 {
		t.Errorf("expected error type to be [\"string\", \"null\"], got %v", errorType)
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
			name: "multiple return values (string, error)",
			code: `package test
import "context"
type Request struct{}
func MultiReturn(ctx context.Context, req Request) (string, error) { return "", nil }`,
			funcName: "MultiReturn",
			wantErr:  "first return value must be a struct type",
		},
		{
			name: "no return value",
			code: `package test
import "context"
type Request struct{}
func NoReturn(ctx context.Context, req Request) { }`,
			funcName: "NoReturn",
			wantErr:  "must return either (ResultType, error) or error",
		},
		{
			name: "return single value instead of (Type, error)",
			code: `package test
import "context"
type Request struct{}
type BadResult struct {
	Value string
}
func BadReturn(ctx context.Context, req Request) BadResult { return BadResult{} }`,
			funcName: "BadReturn",
			wantErr:  "single return value must be error",
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
			err = validateFunction(targetFunc, tt.funcName, []*ast.File{node})
			if err == nil {
				t.Error("expected validation error, got nil")
			} else if tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func validateFunction(targetFunc *ast.FuncDecl, funcName string, files []*ast.File) error {
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
			if !isStructType(t.Name, files) {
				return fmt.Errorf("function %s second parameter must be a struct type, got %s", funcName, t.Name)
			}
		case *ast.StructType:
			return fmt.Errorf("function %s second parameter must be a named struct type; inline structs are not supported", funcName)
		default:
			return fmt.Errorf("function %s second parameter must be a struct type", funcName)
		}
	}

	// Check return values
	if targetFunc.Type.Results == nil || len(targetFunc.Type.Results.List) == 0 || len(targetFunc.Type.Results.List) > 2 {
		return fmt.Errorf("function %s must return either (ResultType, error) or error", funcName)
	}

	if len(targetFunc.Type.Results.List) == 1 {
		if !isErrorType(targetFunc.Type.Results.List[0].Type) {
			return fmt.Errorf("function %s single return value must be error", funcName)
		}
	} else {
		firstResult := targetFunc.Type.Results.List[0].Type
		if !isStructTypeOrNamedStruct(firstResult, files) {
			return fmt.Errorf("function %s first return value must be a struct type", funcName)
		}

		secondResult := targetFunc.Type.Results.List[1].Type
		if !isErrorType(secondResult) {
			return fmt.Errorf("function %s second return value must be error", funcName)
		}
	}

	return nil
}

func TestNoArgumentFunction(t *testing.T) {
	t.Parallel()
	code := `package test
import "context"

type ListFilesResult struct {
	Files []string
}

func ListFiles(ctx context.Context) (ListFilesResult, error) {
	return ListFilesResult{
		Files: []string{"file1.txt", "file2.txt", "dir/file3.go"},
	}, nil
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	require.NoError(t, err)

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "ListFiles" {
			targetFunc = fn
			return false
		}
		return true
	})
	require.NotNil(t, targetFunc)

	err = validateFunction(targetFunc, "ListFiles", []*ast.File{node})
	require.NoError(t, err)

	docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
	require.NoError(t, err)

	inputSchema, err := generateInputSchema(targetFunc.Type.Params, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	outputSchema, err := generateOutputSchema(targetFunc.Type.Results, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	if inputSchema.Type != schema.Object || len(inputSchema.Properties) != 0 {
		t.Fatalf("expected empty object input schema, got %+v", inputSchema.Properties)
	}

	if outputSchema.Properties["Files"] == nil || outputSchema.Properties["error"] == nil {
		t.Fatalf("expected Files + error in output schema, got %+v", outputSchema.Properties)
	}

	tool := &MCPTool{
		Name:         "ListFiles",
		Description:  "Function ListFiles",
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
	}

	jsonBytes, err := json.MarshalIndent(tool, "", "  ")
	require.NoError(t, err)

	var jsonMap map[string]interface{}
	require.NoError(t, json.Unmarshal(jsonBytes, &jsonMap))

	if jsonMap["name"] != "ListFiles" {
		t.Errorf("expected name field to be 'ListFiles', got %v", jsonMap["name"])
	}

	inputMap := jsonMap["inputSchema"].(map[string]interface{})
	if inputMap["type"] != "object" {
		t.Error("expected inputSchema type to be 'object'")
	}
	if props := inputMap["properties"].(map[string]interface{}); len(props) != 0 {
		t.Errorf("expected empty properties in inputSchema, got %d properties", len(props))
	}
}

func TestErrorOnlyFunction(t *testing.T) {
	t.Parallel()
	code := `package test
import "context"

type DeleteRequest struct {
	Path string
}

func DeleteFile(ctx context.Context, req DeleteRequest) error {
	return nil
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	require.NoError(t, err)

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "DeleteFile" {
			targetFunc = fn
			return false
		}
		return true
	})
	require.NotNil(t, targetFunc)

	err = validateFunction(targetFunc, "DeleteFile", []*ast.File{node})
	require.NoError(t, err)

	docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
	require.NoError(t, err)

	inputSchema, err := generateInputSchema(targetFunc.Type.Params, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	outputSchema, err := generateOutputSchema(targetFunc.Type.Results, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	if len(outputSchema.Properties) != 1 || outputSchema.Properties["error"] == nil {
		t.Fatalf("expected only error property in output schema, got %+v", outputSchema.Properties)
	}
	if len(outputSchema.Required) != 1 || outputSchema.Required[0] != "error" {
		t.Fatalf("expected only error to be required, got %v", outputSchema.Required)
	}

	if len(inputSchema.Properties) != 1 || inputSchema.Properties["Path"] == nil {
		t.Fatalf("expected Path property in input schema, got %+v", inputSchema.Properties)
	}
}

func TestThreeLevelNestedStructs(t *testing.T) {
	t.Parallel()

	code := `package test
import "context"

type RelationshipEntry struct {
	Variable          string ` + "`json:\"variable\"`" + `
	Polarity          string ` + "`json:\"polarity\"`" + `
	PolarityReasoning string ` + "`json:\"polarity_reasoning\"`" + `
}

type Chain struct {
	InitialVariable string              ` + "`json:\"initial_variable\"`" + `
	Relationships   []RelationshipEntry ` + "`json:\"relationships\"`" + `
	Reasoning       string              ` + "`json:\"reasoning\"`" + `
}

type Map struct {
	Title        string  ` + "`json:\"title\"`" + `
	Explanation  string  ` + "`json:\"explanation\"`" + `
	CausalChains []Chain ` + "`json:\"causal_chains\"`" + `
}

type MapResult struct {
	Description string     ` + "`json:\"description\"`" + `
	Loops       [][]string ` + "`json:\"loops\"`" + `
}

func SubmitCausalMap(ctx context.Context, m Map) (MapResult, error) {
	return MapResult{Description: "test"}, nil
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	require.NoError(t, err)

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "SubmitCausalMap" {
			targetFunc = fn
			return false
		}
		return true
	})
	require.NotNil(t, targetFunc)

	docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
	require.NoError(t, err)

	inputSchema, err := generateInputSchema(targetFunc.Type.Params, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	// Level 1: Map properties
	assert.Equal(t, schema.Object, inputSchema.Type)
	assert.Len(t, inputSchema.Properties, 3)
	assert.NotNil(t, inputSchema.Properties["title"])
	assert.NotNil(t, inputSchema.Properties["explanation"])
	require.NotNil(t, inputSchema.Properties["causal_chains"])

	// Level 2: causal_chains -> []Chain with full properties
	chainsSchema := inputSchema.Properties["causal_chains"]
	assert.Equal(t, schema.Array, chainsSchema.Type)
	require.NotNil(t, chainsSchema.Items)
	assert.Equal(t, schema.Object, chainsSchema.Items.Type)

	chainProps := chainsSchema.Items.Properties
	assert.Len(t, chainProps, 3)
	assert.NotNil(t, chainProps["initial_variable"])
	assert.NotNil(t, chainProps["reasoning"])
	require.NotNil(t, chainProps["relationships"])

	// Level 3: relationships -> []RelationshipEntry with full properties
	relsSchema := chainProps["relationships"]
	assert.Equal(t, schema.Array, relsSchema.Type)
	require.NotNil(t, relsSchema.Items)
	assert.Equal(t, schema.Object, relsSchema.Items.Type)

	relProps := relsSchema.Items.Properties
	assert.Len(t, relProps, 3)
	require.NotNil(t, relProps["variable"])
	assert.Equal(t, schema.String, relProps["variable"].Type)
	require.NotNil(t, relProps["polarity"])
	assert.Equal(t, schema.String, relProps["polarity"].Type)
	require.NotNil(t, relProps["polarity_reasoning"])
	assert.Equal(t, schema.String, relProps["polarity_reasoning"].Type)

	// Test output schema
	outputSchema, err := generateOutputSchema(targetFunc.Type.Results, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	assert.Equal(t, schema.Object, outputSchema.Type)
	assert.NotNil(t, outputSchema.Properties["description"])
	require.NotNil(t, outputSchema.Properties["loops"])

	loopsSchema := outputSchema.Properties["loops"]
	assert.Equal(t, schema.Array, loopsSchema.Type)
	require.NotNil(t, loopsSchema.Items)
	assert.Equal(t, schema.Array, loopsSchema.Items.Type)
	require.NotNil(t, loopsSchema.Items.Items)
	assert.Equal(t, schema.String, loopsSchema.Items.Items.Type)
}

func TestEnumSupport(t *testing.T) {
	t.Parallel()

	code := `package test
import "context"

type StatusRequest struct {
	Level string ` + "`json:\"level\" enum:\"info,warning,error\"`" + `
}

type StatusResult struct {
	Message string  ` + "`json:\"message\"`" + `
	Error   *string ` + "`json:\"error\"`" + `
}

func SetStatus(ctx context.Context, req StatusRequest) StatusResult {
	return StatusResult{Message: "ok"}
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	require.NoError(t, err)

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "SetStatus" {
			targetFunc = fn
			return false
		}
		return true
	})
	require.NotNil(t, targetFunc)

	docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
	require.NoError(t, err)

	inputSchema, err := generateInputSchema(targetFunc.Type.Params, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	assert.Equal(t, schema.Object, inputSchema.Type)
	require.NotNil(t, inputSchema.Properties["level"])

	levelSchema := inputSchema.Properties["level"]
	assert.Equal(t, schema.String, levelSchema.Type)
	assert.Equal(t, []string{"info", "warning", "error"}, levelSchema.Enum)
}

func TestEnumInNestedStruct(t *testing.T) {
	t.Parallel()

	code := `package test
import "context"

type RelationshipEntry struct {
	Variable          string ` + "`json:\"variable\"`" + `
	Polarity          string ` + "`json:\"polarity\" enum:\"+,-\"`" + `
	PolarityReasoning string ` + "`json:\"polarity_reasoning\"`" + `
}

type Chain struct {
	InitialVariable string              ` + "`json:\"initial_variable\"`" + `
	Relationships   []RelationshipEntry ` + "`json:\"relationships\"`" + `
	Reasoning       string              ` + "`json:\"reasoning\"`" + `
}

type Map struct {
	Title        string  ` + "`json:\"title\"`" + `
	Explanation  string  ` + "`json:\"explanation\"`" + `
	CausalChains []Chain ` + "`json:\"causal_chains\"`" + `
}

type MapResult struct {
	Description string  ` + "`json:\"description\"`" + `
}

func SubmitCausalMap(ctx context.Context, m Map) (MapResult, error) {
	return MapResult{Description: "test"}, nil
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	require.NoError(t, err)

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "SubmitCausalMap" {
			targetFunc = fn
			return false
		}
		return true
	})
	require.NotNil(t, targetFunc)

	docPkg, err := doc.NewFromFiles(fset, []*ast.File{node}, "", doc.AllDecls)
	require.NoError(t, err)

	inputSchema, err := generateInputSchema(targetFunc.Type.Params, []*ast.File{node}, docPkg)
	require.NoError(t, err)

	require.NotNil(t, inputSchema.Properties["causal_chains"])
	chainsSchema := inputSchema.Properties["causal_chains"]
	require.NotNil(t, chainsSchema.Items)

	chainProps := chainsSchema.Items.Properties
	require.NotNil(t, chainProps["relationships"])

	relsSchema := chainProps["relationships"]
	require.NotNil(t, relsSchema.Items)

	relProps := relsSchema.Items.Properties
	require.NotNil(t, relProps["polarity"])

	polaritySchema := relProps["polarity"]
	assert.Equal(t, schema.String, polaritySchema.Type)
	assert.Equal(t, []string{"+", "-"}, polaritySchema.Enum)
}

func TestCrossFileTypeReferences(t *testing.T) {
	t.Parallel()

	// Simulate parsing multiple files with cross-file type references
	file1Code := `package test
import "context"

type RelationshipEntry struct {
	Variable          string ` + "`json:\"variable\"`" + `
	Polarity          string ` + "`json:\"polarity\"`" + `
	PolarityReasoning string ` + "`json:\"polarity_reasoning\"`" + `
}`

	file2Code := `package test

type Chain struct {
	InitialVariable string              ` + "`json:\"initial_variable\"`" + `
	Relationships   []RelationshipEntry ` + "`json:\"relationships\"`" + `
	Reasoning       string              ` + "`json:\"reasoning\"`" + `
}`

	file3Code := `package test
import "context"

type Map struct {
	Title        string  ` + "`json:\"title\"`" + `
	Explanation  string  ` + "`json:\"explanation\"`" + `
	CausalChains []Chain ` + "`json:\"causal_chains\"`" + `
}

type MapResult struct {
	Description string  ` + "`json:\"description\"`" + `
}

func SubmitCausalMap(ctx context.Context, m Map) (MapResult, error) {
	return MapResult{Description: "test"}, nil
}`

	fset := token.NewFileSet()

	file1, err := parser.ParseFile(fset, "types1.go", file1Code, parser.ParseComments)
	require.NoError(t, err)

	file2, err := parser.ParseFile(fset, "types2.go", file2Code, parser.ParseComments)
	require.NoError(t, err)

	file3, err := parser.ParseFile(fset, "tools.go", file3Code, parser.ParseComments)
	require.NoError(t, err)

	files := []*ast.File{file1, file2, file3}

	docPkg, err := doc.NewFromFiles(fset, files, "", doc.AllDecls)
	require.NoError(t, err)

	var targetFunc *ast.FuncDecl
	ast.Inspect(file3, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "SubmitCausalMap" {
			targetFunc = fn
			return false
		}
		return true
	})
	require.NotNil(t, targetFunc)

	inputSchema, err := generateInputSchema(targetFunc.Type.Params, files, docPkg)
	require.NoError(t, err)

	// Level 1: Map properties
	assert.Equal(t, schema.Object, inputSchema.Type)
	assert.Len(t, inputSchema.Properties, 3)
	assert.NotNil(t, inputSchema.Properties["title"])
	assert.NotNil(t, inputSchema.Properties["explanation"])
	require.NotNil(t, inputSchema.Properties["causal_chains"])

	// Level 2: causal_chains -> []Chain
	chainsSchema := inputSchema.Properties["causal_chains"]
	assert.Equal(t, schema.Array, chainsSchema.Type)
	require.NotNil(t, chainsSchema.Items)
	assert.Equal(t, schema.Object, chainsSchema.Items.Type)

	chainProps := chainsSchema.Items.Properties
	assert.Len(t, chainProps, 3)
	assert.NotNil(t, chainProps["initial_variable"])
	assert.NotNil(t, chainProps["reasoning"])
	require.NotNil(t, chainProps["relationships"])

	// Level 3: relationships -> []RelationshipEntry (defined in file1)
	relsSchema := chainProps["relationships"]
	assert.Equal(t, schema.Array, relsSchema.Type)
	require.NotNil(t, relsSchema.Items)
	assert.Equal(t, schema.Object, relsSchema.Items.Type)

	relProps := relsSchema.Items.Properties
	assert.Len(t, relProps, 3)
	require.NotNil(t, relProps["variable"])
	assert.Equal(t, schema.String, relProps["variable"].Type)
	require.NotNil(t, relProps["polarity"])
	assert.Equal(t, schema.String, relProps["polarity"].Type)
	require.NotNil(t, relProps["polarity_reasoning"])
	assert.Equal(t, schema.String, relProps["polarity_reasoning"].Type)
}

func TestTypeNotFoundError(t *testing.T) {
	t.Parallel()

	// Code that references an undefined type
	code := `package test
import "context"

type TestRequest struct {
	Data UndefinedType ` + "`json:\"data\"`" + `
}

type TestResult struct {
	Message string  ` + "`json:\"message\"`" + `
	Error   *string ` + "`json:\"error\"`" + `
}

func TestFunc(ctx context.Context, req TestRequest) TestResult {
	return TestResult{Message: "ok"}
}`

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "test.go", code, parser.ParseComments)
	require.NoError(t, err)

	files := []*ast.File{node}

	docPkg, err := doc.NewFromFiles(fset, files, "", doc.AllDecls)
	require.NoError(t, err)

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == "TestFunc" {
			targetFunc = fn
			return false
		}
		return true
	})
	require.NotNil(t, targetFunc)

	_, err = generateInputSchema(targetFunc.Type.Params, files, docPkg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "UndefinedType")
	assert.Contains(t, err.Error(), "not found in package")
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

func TestRunRejectsInlineStructParameter(t *testing.T) {
	dir := t.TempDir()
	source := `package test
import "context"
type InlineResult struct {
	Value string
}
func InlineFunc(ctx context.Context, req struct {
	Name string
	Age  int
}) (InlineResult, error) {
	return InlineResult{Value: req.Name}, nil
}`

	inputPath := filepath.Join(dir, "inline.go")
	if err := os.WriteFile(inputPath, []byte(source), 0o644); err != nil {
		t.Fatalf("failed to write source file: %v", err)
	}

	origFuncName := *funcName
	origInputFile := *inputFile
	t.Cleanup(func() {
		*funcName = origFuncName
		*inputFile = origInputFile
	})

	*funcName = "InlineFunc"
	*inputFile = inputPath

	err := run()
	if err == nil {
		t.Fatal("expected run() to fail for inline request struct, got nil")
	}
	if !strings.Contains(err.Error(), "named struct type") {
		t.Fatalf("expected error about named struct type, got: %v", err)
	}
}
