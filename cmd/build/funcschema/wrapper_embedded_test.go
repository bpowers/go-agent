package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestGeneratedWrapperWithEmbeddedStruct(t *testing.T) {
	t.Parallel()
	// Test wrapper generation for embedded structs
	tmpDir := t.TempDir()

	// Create test file with embedded struct
	testdataContent := `package main

import "context"
import "strconv"

type BaseInfo struct {
	ID        string ` + "`json:\"id\"`" + `
	CreatedAt string ` + "`json:\"created_at\"`" + `
}

type Metadata struct {
	Version int      ` + "`json:\"version\"`" + `
	Tags    []string ` + "`json:\"tags\"`" + `
}

type DocumentRequest struct {
	BaseInfo
	*Metadata
	Title    string ` + "`json:\"title\"`" + `
	Content  string ` + "`json:\"content\"`" + `
}

type DocumentResult struct {
	Success bool    ` + "`json:\"success\"`" + `
	DocID   string  ` + "`json:\"doc_id\"`" + `
}

func CreateDocument(ctx context.Context, req DocumentRequest) (DocumentResult, error) {
	// Simulate using embedded fields
	docID := req.ID + "-v" + strconv.Itoa(req.Version)
	return DocumentResult{
		Success: true,
		DocID:   docID,
	}, nil
}`

	testFile := filepath.Join(tmpDir, "testdata.go")
	if err := os.WriteFile(testFile, []byte(testdataContent), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	// Create go.mod for the test
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("failed to get repo root: %v", err)
	}
	goModContent := `module testpkg

go 1.24

require github.com/bpowers/go-agent v0.0.0

replace github.com/bpowers/go-agent => ` + repoRoot + `
`
	goModFile := filepath.Join(tmpDir, "go.mod")
	if err := os.WriteFile(goModFile, []byte(goModContent), 0o644); err != nil {
		t.Fatalf("failed to write go.mod: %v", err)
	}

	// Run funcschema to generate the wrapper
	cmd := exec.Command("go", "run", ".", "-func", "CreateDocument", "-input", testFile)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run funcschema: %v\nOutput: %s", err, output)
	}

	// Check that the generated file exists
	generatedFile := filepath.Join(tmpDir, "createdocument_tool.go")
	if _, err := os.Stat(generatedFile); os.IsNotExist(err) {
		t.Fatal("generated file does not exist")
	}

	// Read the generated file
	content, err := os.ReadFile(generatedFile)
	if err != nil {
		t.Fatalf("failed to read generated file: %v", err)
	}

	// Check that the generated JSON schema includes embedded fields
	expectedFields := []string{
		`"id"`,         // From BaseInfo
		`"created_at"`, // From BaseInfo
		`"version"`,    // From Metadata pointer
		`"tags"`,       // From Metadata pointer
		`"title"`,      // From DocumentRequest
		`"content"`,    // From DocumentRequest
	}

	for _, field := range expectedFields {
		if !strings.Contains(string(content), field) {
			t.Errorf("generated schema missing expected field: %s", field)
		}
	}

	// Run go mod tidy
	tidyCmd := exec.Command("go", "mod", "tidy")
	tidyCmd.Dir = tmpDir
	if output, err := tidyCmd.CombinedOutput(); err != nil {
		t.Logf("go mod tidy output: %s", output)
	}

	// Test that it compiles and the wrapper works correctly with embedded fields
	testMain := filepath.Join(tmpDir, "main_test.go")
	testContent := `package main

import (
	"context"
	"encoding/json"
	"testing"
)

func TestEmbeddedWrapper(t *testing.T) {
	ctx := context.Background()
	input := ` + "`" + `{
		"id": "doc123",
		"created_at": "2024-01-01T00:00:00Z",
		"version": 2,
		"tags": ["important", "reviewed"],
		"title": "Test Document",
		"content": "This is test content"
	}` + "`" + `

	output := CreateDocumentTool.Call(ctx, input)

	// The result is wrapped with an Error field by the generator
	var result struct {
		DocumentResult
		Error *string ` + "`json:\"error,omitzero\"`" + `
	}
	if err := json.Unmarshal([]byte(output), &result); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", *result.Error)
	}
	if !result.Success {
		t.Errorf("expected Success=true")
	}
	if result.DocID != "doc123-v2" {
		t.Errorf("expected DocID='doc123-v2', got %s", result.DocID)
	}
}`

	if err := os.WriteFile(testMain, []byte(testContent), 0o644); err != nil {
		t.Fatalf("failed to write test main: %v", err)
	}

	// Run the test to verify it compiles and works
	cmd = exec.Command("go", "test", "-v")
	cmd.Dir = tmpDir
	output, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("generated code does not compile or test fails: %v\nOutput: %s", err, output)
	}
}
