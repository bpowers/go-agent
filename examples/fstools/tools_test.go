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

	// Call ReadDir directly
	result := ReadDir(ctx)

	// Check for errors
	require.Nil(t, result.Error)

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
	output := ReadDirTool(ctx, "{}")

	// Parse the JSON output
	var result ReadDirResult
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
	result := ReadFile(ctx, ReadFileRequest{FileName: "test.txt"})
	require.Nil(t, result.Error)
	assert.Equal(t, testContent, result.Content)

	// Test reading non-existent file
	result = ReadFile(ctx, ReadFileRequest{FileName: "nonexistent.txt"})
	assert.NotNil(t, result.Error)
	assert.Contains(t, *result.Error, "failed to open file")
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
	output := ReadFileTool(ctx, input)

	// Parse the JSON output
	var result ReadFileResult
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
	result := WriteFile(ctx, WriteFileRequest{
		FileName: "new.txt",
		Content:  content,
	})

	require.Nil(t, result.Error)
	assert.True(t, result.Success)

	// Verify the file was written
	data, err := fs.ReadFile(testFS, "new.txt")
	require.NoError(t, err)
	assert.Equal(t, content, string(data))

	// Test writing to subdirectory (should create it)
	result = WriteFile(ctx, WriteFileRequest{
		FileName: "subdir/nested.txt",
		Content:  "nested content",
	})

	require.Nil(t, result.Error)

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
	output := WriteFileTool(ctx, input)

	// Parse the JSON output
	var result WriteFileResult
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
	result := WriteFile(ctx, WriteFileRequest{
		FileName: "/absolute/path.txt",
		Content:  "absolute path",
	})

	require.Nil(t, result.Error)

	// The file should be written to "absolute/path.txt" (cleaned path without leading /)
	data, err := fs.ReadFile(testFS, "absolute/path.txt")
	require.NoError(t, err)
	assert.Equal(t, "absolute path", string(data))

	// Test that reading with absolute path also works
	readResult := ReadFile(ctx, ReadFileRequest{FileName: "/absolute/path.txt"})
	require.Nil(t, readResult.Error)
	assert.Equal(t, "absolute path", readResult.Content)
}

func TestNoFilesystem(t *testing.T) {
	t.Parallel()
	// Test error handling when no filesystem in context
	ctx := context.Background()

	result := ReadDir(ctx)
	assert.NotNil(t, result.Error)
	assert.Contains(t, *result.Error, "no filesystem found in context")
}
