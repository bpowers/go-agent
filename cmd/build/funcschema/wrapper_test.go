package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Test that the generated wrapper function compiles and has correct structure
func TestGeneratedWrapperCompiles(t *testing.T) {
	t.Parallel()
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Copy testdata.go to temp dir
	testdataContent := `package main

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
	return DatasetGetResult{
		Revision: "1",
		Data: map[string][][]float64{
			"a": {{0, 1}, {1, 6.3}},
			"b": {{0, 1}, {1, 2}},
		},
	}
}`

	testFile := filepath.Join(tmpDir, "testdata.go")
	if err := os.WriteFile(testFile, []byte(testdataContent), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	// Create go.mod for the test
	// Get the absolute path to the repo root (3 levels up from this test file)
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
	cmd := exec.Command("go", "run", ".", "-func", "DatasetGet", "-input", testFile)
	// Run from current directory since test is in the same package
	// cmd.Dir is not needed since exec.Command will inherit the test's working directory
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run funcschema: %v\nOutput: %s", err, output)
	}

	// Check that the generated file exists
	generatedFile := filepath.Join(tmpDir, "datasetget_tool.go")
	if _, err := os.Stat(generatedFile); os.IsNotExist(err) {
		t.Fatal("generated file does not exist")
	}

	// Read the generated file
	content, err := os.ReadFile(generatedFile)
	if err != nil {
		t.Fatalf("failed to read generated file: %v", err)
	}

	// Check that it contains the expected elements
	expectedElements := []string{
		"DatasetGetToolDef",
		"func DatasetGetTool(ctx context.Context, input string) string",
		"var req DatasetGetRequest",
		"result := DatasetGet(ctx, req)",
		"json.Unmarshal",
		"json.Marshal",
	}

	for _, elem := range expectedElements {
		if !strings.Contains(string(content), elem) {
			t.Errorf("generated file missing expected element: %s", elem)
		}
	}

	// Check that outputSchema is included in the generated JSON
	// The JSON might use backticks or escaped quotes, so check for both
	hasOutputSchema := strings.Contains(string(content), `"outputSchema"`) ||
		strings.Contains(string(content), `\"outputSchema\"`)
	if !hasOutputSchema {
		t.Error("generated JSON missing outputSchema field")
	}

	// Verify the outputSchema contains expected properties
	hasRevision := strings.Contains(string(content), `"Revision"`) ||
		strings.Contains(string(content), `"revision"`) ||
		strings.Contains(string(content), `\"Revision\"`) ||
		strings.Contains(string(content), `\"revision\"`)
	if !hasRevision {
		t.Error("outputSchema missing Revision field")
	}

	hasData := strings.Contains(string(content), `"Data"`) ||
		strings.Contains(string(content), `"data"`) ||
		strings.Contains(string(content), `\"Data\"`) ||
		strings.Contains(string(content), `\"data\"`)
	if !hasData {
		t.Error("outputSchema missing Data field")
	}

	// Run go mod tidy to fetch dependencies after files are generated
	tidyCmd := exec.Command("go", "mod", "tidy")
	tidyCmd.Dir = tmpDir
	if output, err := tidyCmd.CombinedOutput(); err != nil {
		t.Logf("go mod tidy output: %s", output)
		// Don't fail here, the test compilation might still work
	}

	// Test that it compiles
	testMain := filepath.Join(tmpDir, "main_test.go")
	testContent := `package main

import (
	"context"
	"encoding/json"
	"testing"
)

func TestWrapper(t *testing.T) {
	ctx := context.Background()
	input := ` + "`" + `{"datasetId": "test123", "where": null}` + "`" + `
	output := DatasetGetTool(ctx, input)
	
	var result DatasetGetResult
	if err := json.Unmarshal([]byte(output), &result); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	
	if result.Revision != "1" {
		t.Errorf("expected Revision='1', got %s", result.Revision)
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

func TestGeneratedWrapperNoArgumentFunction(t *testing.T) {
	t.Parallel()
	// Test wrapper generation for a function with no arguments (only context)
	tmpDir := t.TempDir()

	// Create test file with no-argument function
	testdataContent := `package main

import "context"

type GetSystemInfoResult struct {
	Hostname string
	OS       string
	Version  string
	Error    *string
}

func GetSystemInfo(ctx context.Context) GetSystemInfoResult {
	return GetSystemInfoResult{
		Hostname: "test-host",
		OS:       "linux",
		Version:  "1.0.0",
	}
}`

	testFile := filepath.Join(tmpDir, "testdata.go")
	if err := os.WriteFile(testFile, []byte(testdataContent), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	// Create go.mod for the test
	// Get the absolute path to the repo root (3 levels up from this test file)
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

	// Run go mod tidy to fetch dependencies
	tidyCmd := exec.Command("go", "mod", "tidy")
	tidyCmd.Dir = tmpDir
	if output, err := tidyCmd.CombinedOutput(); err != nil {
		t.Logf("go mod tidy output: %s", output)
		// Don't fail here, the test compilation might still work
	}

	// Run funcschema to generate the wrapper
	cmd := exec.Command("go", "run", ".", "-func", "GetSystemInfo", "-input", testFile)
	// Run from current directory since test is in the same package
	// cmd.Dir is not needed since exec.Command will inherit the test's working directory
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run funcschema: %v\nOutput: %s", err, output)
	}

	// Check that the generated file exists
	generatedFile := filepath.Join(tmpDir, "getsysteminfo_tool.go")
	if _, err := os.Stat(generatedFile); os.IsNotExist(err) {
		t.Fatal("generated file does not exist")
	}

	// Read the generated file
	content, err := os.ReadFile(generatedFile)
	if err != nil {
		t.Fatalf("failed to read generated file: %v", err)
	}

	// Check that it contains the expected elements for no-argument function
	expectedElements := []string{
		"GetSystemInfoToolDef",
		"func GetSystemInfoTool(ctx context.Context, input string) string",
		"// No input parameters needed, ignore input JSON",
		"result := GetSystemInfo(ctx)", // Call with only context
		"json.Marshal(result)",
	}

	for _, elem := range expectedElements {
		if !strings.Contains(string(content), elem) {
			t.Errorf("generated file missing expected element: %s", elem)
		}
	}

	// Make sure it does NOT contain request unmarshaling
	unexpectedElements := []string{
		"var req",
		"json.Unmarshal([]byte(input), &req)",
	}

	for _, elem := range unexpectedElements {
		if strings.Contains(string(content), elem) {
			t.Errorf("generated file should not contain: %s", elem)
		}
	}

	// Run go mod tidy to fetch dependencies after files are generated
	tidyCmd2 := exec.Command("go", "mod", "tidy")
	tidyCmd2.Dir = tmpDir
	if output, err := tidyCmd2.CombinedOutput(); err != nil {
		t.Logf("go mod tidy output: %s", output)
		// Don't fail here, the test compilation might still work
	}

	// Test that it compiles
	testMain := filepath.Join(tmpDir, "main_test.go")
	testContent := `package main

import (
	"context"
	"encoding/json"
	"testing"
)

func TestNoArgWrapper(t *testing.T) {
	ctx := context.Background()
	// Input is ignored for no-argument functions
	output := GetSystemInfoTool(ctx, "{}")
	
	var result GetSystemInfoResult
	if err := json.Unmarshal([]byte(output), &result); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	
	if result.Hostname != "test-host" {
		t.Errorf("expected Hostname='test-host', got %s", result.Hostname)
	}
	if result.OS != "linux" {
		t.Errorf("expected OS='linux', got %s", result.OS)
	}
	if result.Version != "1.0.0" {
		t.Errorf("expected Version='1.0.0', got %s", result.Version)
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
