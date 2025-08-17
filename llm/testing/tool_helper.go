package testing

import (
	"context"
	"io/fs"
	"strings"
	"testing"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/examples/fstools"
	"github.com/psanford/memfs"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestToolsSummarizesFile tests tool calling functionality with filesystem tools
func TestToolsSummarizesFile(t testing.TB, client chat.Client) {
	// Create in-memory filesystem
	testFS := memfs.New()

	// Seed the filesystem with a system dynamics document
	systemDynamicsContent := `System dynamics is a methodology and mathematical modeling technique for understanding complex systems and their behavior over time. It was originally developed in the 1950s by Jay Forrester at MIT to help corporate managers improve their understanding of industrial processes.

The core principle of system dynamics is that the structure of a system - the network of relationships between its components - is just as important as the individual components themselves. This approach recognizes that systems often exhibit counterintuitive behavior, where obvious solutions may actually make problems worse due to feedback loops and delays in the system.

In strategic business applications, system dynamics provides powerful tools for scenario planning and policy design. By creating models that capture the essential feedback structures of a business or market, strategists can test different interventions and understand their long-term consequences before implementation. This is particularly valuable when dealing with complex challenges like market dynamics, supply chain management, or organizational change.

The methodology has proven especially useful in strategy consulting, where it helps executives understand the unintended consequences of their decisions. By mapping out reinforcing and balancing feedback loops, identifying key leverage points, and simulating various scenarios, system dynamics enables more robust strategic planning that accounts for the dynamic complexity inherent in modern business environments.`

	if err := testFS.WriteFile("system_dynamics.md", []byte(systemDynamicsContent), 0o644); err != nil {
		t.Fatalf("Failed to seed filesystem: %v", err)
	}

	// Create context with filesystem
	ctx := fstools.WithTestFS(context.Background(), testFS)

	// Create a new chat session with tool support
	systemPrompt := "You are a helpful assistant with several tools at your disposal."
	chatSession := client.NewChat(systemPrompt)

	// Register the filesystem tools
	err := chatSession.RegisterTool(fstools.ReadDirToolDef, fstools.ReadDirTool)
	require.NoError(t, err)

	err = chatSession.RegisterTool(fstools.ReadFileToolDef, fstools.ReadFileTool)
	require.NoError(t, err)

	err = chatSession.RegisterTool(fstools.WriteFileToolDef, fstools.WriteFileTool)
	require.NoError(t, err)

	// Verify tools are registered
	tools := chatSession.ListTools()
	if len(tools) != 3 {
		t.Errorf("Expected 3 tools registered, got %d", len(tools))
	}

	// Ask the LLM to find and summarize the file
	prompt := "Using your tools, find the file we have and summarize it in 1 paragraph into a new file called summary.md"

	response, err := chatSession.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: prompt,
	})
	if err != nil {
		t.Fatalf("Failed to get response: %v", err)
	}

	// Check that the response indicates success
	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	// Verify that summary.md was created
	summaryData, err := fs.ReadFile(testFS, "summary.md")
	if err != nil {
		t.Errorf("Failed to read summary.md: %v", err)
	}

	t.Logf("Summary:\n%s\n", string(summaryData))

	// Verify the summary contains at least some relevant content
	// We check for basic keywords but don't fail if some are missing
	// since different LLMs may express concepts differently
	summary := strings.ToLower(string(summaryData))
	keywordMatches := 0
	expectedKeywords := []string{"system", "dynamics", "feedback", "strategy", "model", "business"}
	for _, keyword := range expectedKeywords {
		if strings.Contains(summary, keyword) {
			keywordMatches++
		}
	}
	// Require at least 2 keywords to ensure some relevance
	if keywordMatches < 2 {
		t.Errorf("Summary appears unrelated to content: only %d/%d keywords found",
			keywordMatches, len(expectedKeywords))
	}

	// Verify the summary exists and is not empty, but don't enforce strict length limits
	// since different models may produce different length summaries
	if len(summaryData) == 0 {
		t.Error("Summary is empty")
	} else if len(summaryData) > len(systemDynamicsContent) {
		// Only fail if summary is longer than original, which would be clearly wrong
		t.Errorf("Summary is longer than original: %d bytes (original was %d bytes)",
			len(summaryData), len(systemDynamicsContent))
	}

	t.Logf("Tool calling test passed: LLM successfully used tools to read and summarize content")
}

// TestWritesFile tests that context is properly passed through tool calls
func TestWritesFile(t testing.TB, client chat.Client) {
	testFS := memfs.New()
	ctx := fstools.WithTestFS(context.Background(), testFS)

	chatSession := client.NewChat("You are a helpful assistant that can work with files.")

	err := chatSession.RegisterTool(fstools.WriteFileToolDef, fstools.WriteFileTool)
	require.NoError(t, err)

	// Ask the LLM to create a file
	response, err := chatSession.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Please create a file called test.txt with the content 'Hello from tools!'",
	})
	require.NoError(t, err)

	assert.NotEmpty(t, response.Content)

	data, err := fs.ReadFile(testFS, "test.txt")
	require.NoError(t, err)

	content := string(data)
	assert.Contains(t, content, "Hello from tools")
}
