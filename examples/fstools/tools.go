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

// contextKey is a private type for context keys
type contextKey struct{}

// WithFS adds an fs.FS to the context for downstream tool calls.
func WithFS(ctx context.Context, f fs.FS) context.Context {
	return context.WithValue(ctx, contextKey{}, f)
}

// GetFS retrieves the filesystem from the context.
func GetFS(ctx context.Context) (fs.FS, error) {
	fs, ok := ctx.Value(contextKey{}).(fs.FS)
	if !ok {
		return nil, fmt.Errorf("no filesystem found in context")
	}
	return fs, nil
}

// ReadDirRequest is the input for ReadDir
type ReadDirRequest struct {
	Path string `json:"path,omitzero"` // Directory path to read (defaults to "." for root)
}

// ReadDirResult is the output of ReadDir
type ReadDirResult struct {
	Files []FileInfo `json:"files"`
}

// FileInfo contains information about a file
type FileInfo struct {
	Name  string `json:"name"`
	IsDir bool   `json:"isDir"`
	Size  int64  `json:"size"`
}

//go:generate go run ../../cmd/build/funcschema/main.go -func ReadDir -input tools.go

// ReadDir reads a directory from the test filesystem
func ReadDir(ctx context.Context, req ReadDirRequest) (ReadDirResult, error) {
	fileSystem, err := GetFS(ctx)
	if err != nil {
		return ReadDirResult{}, err
	}

	dirPath := path.Clean(req.Path)
	switch dirPath {
	case "", ".", "/":
		dirPath = "."
	default:
		dirPath = strings.TrimPrefix(dirPath, "/")
		if dirPath == "" {
			dirPath = "."
		}
	}

	entries, err := fs.ReadDir(fileSystem, dirPath)
	if err != nil {
		return ReadDirResult{}, fmt.Errorf("failed to read directory %s: %w", dirPath, err)
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

	return ReadDirResult{Files: files}, nil
}

// ReadFileRequest is the input for ReadFile
type ReadFileRequest struct {
	FileName string `json:"fileName"`
}

// ReadFileResult is the output of ReadFile
type ReadFileResult struct {
	Content string `json:"content"`
}

//go:generate go run ../../cmd/build/funcschema/main.go -func ReadFile -input tools.go

// ReadFile reads a file from the test filesystem
func ReadFile(ctx context.Context, req ReadFileRequest) (ReadFileResult, error) {
	fileSystem, err := GetFS(ctx)
	if err != nil {
		return ReadFileResult{}, err
	}

	// directory traversal should be prevented with os.Root or the use of an in-memory FS,
	// but still do our best to clean up the path.
	fileName := path.Clean(req.FileName)
	fileName = strings.TrimPrefix(fileName, "/")

	file, err := fileSystem.Open(fileName)
	if err != nil {
		return ReadFileResult{}, fmt.Errorf("failed to open file %s: %w", fileName, err)
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return ReadFileResult{}, fmt.Errorf("failed to read file %s: %w", fileName, err)
	}

	return ReadFileResult{Content: string(content)}, nil
}

// WriteFileRequest is the input for WriteFile
type WriteFileRequest struct {
	FileName string `json:"fileName"`
	Content  string `json:"content"`
}

// WriteFileResult is the output of WriteFile
type WriteFileResult struct {
	Success bool `json:"success"`
}

//go:generate go run ../../cmd/build/funcschema/main.go -func WriteFile -input tools.go

// WriteFile writes a file to the test filesystem
func WriteFile(ctx context.Context, req WriteFileRequest) (WriteFileResult, error) {
	fileSystem, err := GetFS(ctx)
	if err != nil {
		return WriteFileResult{}, err
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
				return WriteFileResult{}, fmt.Errorf("failed to create directory %s: %w", dir, err)
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
			return WriteFileResult{}, fmt.Errorf("failed to write file %s: %w", fileName, err)
		}
	} else {
		return WriteFileResult{}, fmt.Errorf("read-only filesystem")
	}

	return WriteFileResult{Success: true}, nil
}
