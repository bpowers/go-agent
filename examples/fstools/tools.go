package fstools

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"strings"
)

// stringPtr is a helper function to get a pointer to a string
func stringPtr(s string) *string {
	return &s
}

// contextKey is a private type for context keys
type contextKey struct{}

// WithTestFS adds a test filesystem to the context
func WithTestFS(ctx context.Context, f fs.FS) context.Context {
	return context.WithValue(ctx, contextKey{}, f)
}

// GetTestFS retrieves the test filesystem from the context
func GetTestFS(ctx context.Context) (fs.FS, error) {
	testFS, ok := ctx.Value(contextKey{}).(fs.FS)
	if !ok {
		return nil, fmt.Errorf("no filesystem found in context")
	}
	return testFS, nil
}

// ReadDirRequest is the input for ReadDir (empty for no-argument function)
type ReadDirRequest struct{}

// ReadDirResult is the output of ReadDir
type ReadDirResult struct {
	Files []FileInfo `json:"files"`
	Error *string    `json:"error,omitzero"`
}

// FileInfo contains information about a file
type FileInfo struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir"`
	Size  int64  `json:"size"`
}

//go:generate go run ../../cmd/build/funcschema/main.go -func ReadDir -input tools.go

// ReadDir reads the root directory of the test filesystem
func ReadDir(ctx context.Context) ReadDirResult {
	fileSystem, err := GetTestFS(ctx)
	if err != nil {
		errStr := err.Error()
		return ReadDirResult{Error: &errStr}
	}

	entries, err := fs.ReadDir(fileSystem, ".")
	if err != nil {
		errStr := err.Error()
		return ReadDirResult{Error: &errStr}
	}

	files := make([]FileInfo, 0, len(entries))
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}
		files = append(files, FileInfo{
			Name:  entry.Name(),
			IsDir: entry.IsDir(),
			Size:  info.Size(),
		})
	}

	return ReadDirResult{Files: files}
}

// ReadFileRequest is the input for ReadFile
type ReadFileRequest struct {
	FileName string `json:"fileName"`
}

// ReadFileResult is the output of ReadFile
type ReadFileResult struct {
	Content string  `json:"content"`
	Error   *string `json:"error,omitzero"`
}

//go:generate go run ../../cmd/build/funcschema/main.go -func ReadFile -input tools.go

// ReadFile reads a file from the test filesystem
func ReadFile(ctx context.Context, req ReadFileRequest) ReadFileResult {
	fileSystem, err := GetTestFS(ctx)
	if err != nil {
		errStr := err.Error()
		return ReadFileResult{Error: &errStr}
	}

	// directory traversal should be prevented with os.Root or the use of an in-memory FS,
	// but still do our best to clean up the path.
	fileName := path.Clean(req.FileName)
	fileName = strings.TrimPrefix(fileName, "/")

	file, err := fileSystem.Open(fileName)
	if err != nil {
		errStr := fmt.Sprintf("failed to open file %s: %v", fileName, err)
		return ReadFileResult{Error: &errStr}
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		errStr := fmt.Sprintf("failed to read file %s: %v", fileName, err)
		return ReadFileResult{Error: &errStr}
	}

	return ReadFileResult{Content: string(content)}
}

// WriteFileRequest is the input for WriteFile
type WriteFileRequest struct {
	FileName string `json:"fileName"`
	Content  string `json:"content"`
}

// WriteFileResult is the output of WriteFile
type WriteFileResult struct {
	Success bool    `json:"success"`
	Error   *string `json:"error,omitzero"`
}

//go:generate go run ../../cmd/build/funcschema/main.go -func WriteFile -input tools.go

// WriteFile writes a file to the test filesystem
func WriteFile(ctx context.Context, req WriteFileRequest) WriteFileResult {
	fileSystem, err := GetTestFS(ctx)
	if err != nil {
		errStr := err.Error()
		return WriteFileResult{Error: &errStr}
	}

	// Clean the path to prevent directory traversal
	fileName := path.Clean(req.FileName)
	fileName = strings.TrimPrefix(fileName, "/")

	// Create directory if needed
	dir := path.Dir(fileName)
	if dir != "." && dir != "/" {
		type mkdirAller interface {
			MkdirAll(path string, perm os.FileMode) error
		}
		if f, ok := fileSystem.(mkdirAller); ok {
			err = f.MkdirAll(dir, 0o755)
			if err != nil {
				errStr := fmt.Sprintf("failed to create directory %s: %v", dir, err)
				return WriteFileResult{Error: &errStr}
			}
		}
	}

	// github.com/psanford/memfs.FS implements this
	type writer interface {
		WriteFile(path string, data []byte, perm os.FileMode) error
	}
	if f, ok := fileSystem.(writer); ok {
		err = f.WriteFile(fileName, []byte(req.Content), 0o644)
		if err != nil {
			errStr := fmt.Sprintf("failed to write file %s: %v", fileName, err)
			return WriteFileResult{Error: &errStr}
		}
	} else {
		return WriteFileResult{Error: stringPtr("read-only filesystem")}
	}

	return WriteFileResult{Success: true}
}
