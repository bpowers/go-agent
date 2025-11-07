package fstools

import (
	"context"
	"encoding/json"
	"io/fs"
	"testing"

	"github.com/psanford/memfs"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadDirTool(t *testing.T) {
	t.Parallel()
	// Create in-memory filesystem
	testFS := memfs.New()

	// Add some test files
	err := testFS.WriteFile("file1.txt", []byte("content1"), 0o644)
	require.NoError(t, err)
	err = testFS.WriteFile("file2.go", []byte("package main"), 0o644)
	require.NoError(t, err)
	err = testFS.MkdirAll("subdir", 0o755)
	require.NoError(t, err)
	err = testFS.WriteFile("subdir/nested.txt", []byte("nested content"), 0o644)
	require.NoError(t, err)

	// Create context with filesystem
	ctx := WithTestFS(context.Background(), testFS)

	// Call ReadDir directly (empty path defaults to root)
	result, err := ReadDir(ctx, ReadDirRequest{})

	// Check for errors
	require.NoError(t, err)

	// Verify we got the expected files
	assert.Len(t, result.Files, 3)

	// Check that files are present
	foundFile1 := false
	foundFile2 := false
	foundSubdir := false

	for _, f := range result.Files {
		switch f.Name {
		case "file1.txt":
			foundFile1 = true
			assert.False(t, f.IsDir)
			assert.Equal(t, int64(8), f.Size) // "content1"
		case "file2.go":
			foundFile2 = true
			assert.False(t, f.IsDir)
		case "subdir":
			foundSubdir = true
			assert.True(t, f.IsDir)
		}
	}

	assert.True(t, foundFile1)
	assert.True(t, foundFile2)
	assert.True(t, foundSubdir)
}

func TestReadDirToolWrapper(t *testing.T) {
	t.Parallel()
	// Create in-memory filesystem
	testFS := memfs.New()
	err := testFS.WriteFile("test.txt", []byte("test"), 0o644)
	require.NoError(t, err)

	ctx := WithTestFS(context.Background(), testFS)

	// Call the generated wrapper function
	output := ReadDirTool.Call(ctx, "{}")

	// Parse the JSON output (includes error field from wrapper)
	var result struct {
		ReadDirResult
		Error *string `json:"error,omitzero"`
	}
	err = json.Unmarshal([]byte(output), &result)
	require.NoError(t, err)

	require.Nil(t, result.Error)
	assert.Len(t, result.Files, 1)
}

func TestReadFileTool(t *testing.T) {
	t.Parallel()
	// Create in-memory filesystem
	testFS := memfs.New()
	testContent := "This is test content\nWith multiple lines"
	err := testFS.WriteFile("test.txt", []byte(testContent), 0o644)
	require.NoError(t, err)

	ctx := WithTestFS(context.Background(), testFS)

	// Test reading existing file
	result, err := ReadFile(ctx, ReadFileRequest{FileName: "test.txt"})
	require.NoError(t, err)
	assert.Equal(t, testContent, result.Content)

	// Test reading non-existent file
	_, err = ReadFile(ctx, ReadFileRequest{FileName: "nonexistent.txt"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to open file")
}

func TestReadFileToolWrapper(t *testing.T) {
	t.Parallel()
	// Create in-memory filesystem
	testFS := memfs.New()
	testContent := "Test content"
	err := testFS.WriteFile("test.txt", []byte(testContent), 0o644)
	require.NoError(t, err)

	ctx := WithTestFS(context.Background(), testFS)

	// Call the generated wrapper function
	input := `{"fileName": "test.txt"}`
	output := ReadFileTool.Call(ctx, input)

	// Parse the JSON output (includes error field from wrapper)
	var result struct {
		ReadFileResult
		Error *string `json:"error,omitzero"`
	}
	err = json.Unmarshal([]byte(output), &result)
	require.NoError(t, err)

	require.Nil(t, result.Error)
	assert.Equal(t, testContent, result.Content)
}

func TestWriteFileTool(t *testing.T) {
	t.Parallel()
	// Create in-memory filesystem
	testFS := memfs.New()
	ctx := WithTestFS(context.Background(), testFS)

	// Write a new file
	content := "New file content"
	result, err := WriteFile(ctx, WriteFileRequest{
		FileName: "new.txt",
		Content:  content,
	})

	require.NoError(t, err)
	assert.True(t, result.Success)

	// Verify the file was written
	data, err := fs.ReadFile(testFS, "new.txt")
	require.NoError(t, err)
	assert.Equal(t, content, string(data))

	// Test writing to subdirectory (should create it)
	result, err = WriteFile(ctx, WriteFileRequest{
		FileName: "subdir/nested.txt",
		Content:  "nested content",
	})

	require.NoError(t, err)

	// Verify nested file
	data, err = fs.ReadFile(testFS, "subdir/nested.txt")
	require.NoError(t, err)
	assert.Equal(t, "nested content", string(data))
}

func TestWriteFileToolWrapper(t *testing.T) {
	t.Parallel()
	// Create in-memory filesystem
	testFS := memfs.New()
	ctx := WithTestFS(context.Background(), testFS)

	// Call the generated wrapper function
	input := `{"fileName": "test.txt", "content": "Test content"}`
	output := WriteFileTool.Call(ctx, input)

	// Parse the JSON output (includes error field from wrapper)
	var result struct {
		WriteFileResult
		Error *string `json:"error,omitzero"`
	}
	err := json.Unmarshal([]byte(output), &result)
	require.NoError(t, err)

	require.Nil(t, result.Error)
	assert.True(t, result.Success)

	// Verify the file was written
	data, err := fs.ReadFile(testFS, "test.txt")
	require.NoError(t, err)
	assert.Equal(t, "Test content", string(data))
}

func TestPathCleaning(t *testing.T) {
	t.Parallel()
	// Test that path cleaning works
	testFS := memfs.New()
	ctx := WithTestFS(context.Background(), testFS)

	// Write a file with a path that needs cleaning
	_, err := WriteFile(ctx, WriteFileRequest{
		FileName: "/absolute/path.txt",
		Content:  "absolute path",
	})

	require.NoError(t, err)

	// The file should be written to "absolute/path.txt" (cleaned path without leading /)
	data, err := fs.ReadFile(testFS, "absolute/path.txt")
	require.NoError(t, err)
	assert.Equal(t, "absolute path", string(data))

	// Test that reading with absolute path also works
	readResult, err := ReadFile(ctx, ReadFileRequest{FileName: "/absolute/path.txt"})
	require.NoError(t, err)
	assert.Equal(t, "absolute path", readResult.Content)
}

func TestNoFilesystem(t *testing.T) {
	t.Parallel()
	// Test error handling when no filesystem in context
	ctx := context.Background()

	_, err := ReadDir(ctx, ReadDirRequest{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no filesystem found in context")
}
