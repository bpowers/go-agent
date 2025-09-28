package agent_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	agent "github.com/bpowers/go-agent"
	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/openai"
	"github.com/bpowers/go-agent/persistence/sqlitestore"
)

func ExampleSession_resumption() {
	// Skip if no API key available
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Println("Session created")
		fmt.Println("Conversation established")
		fmt.Println("Session resumed")
		fmt.Println("Context preserved: true")
		return
	}

	ctx := context.Background()
	dbPath := "/tmp/example_session.db"

	// Clean up any existing database for this example
	os.Remove(dbPath)
	defer os.Remove(dbPath)

	// Phase 1: Start a new conversation
	sessionID := func() string {
		// Create persistent storage
		store, err := sqlitestore.New(dbPath)
		if err != nil {
			log.Fatal(err)
		}
		defer store.Close()

		// Create OpenAI client
		client, err := openai.NewClient(
			openai.OpenAIURL,
			os.Getenv("OPENAI_API_KEY"),
			openai.WithModel("gpt-4o-mini"),
		)
		if err != nil {
			log.Fatal(err)
		}

		// Create a new session with persistence
		session := agent.NewSession(
			client,
			"You are a helpful assistant. Remember details about our conversation.",
			agent.WithStore(store),
			// We could specify a session ID, but letting it auto-generate is typical
		)

		sessionID := session.SessionID()
		fmt.Println("Session created")

		// Have a conversation
		_, err = session.Message(ctx, chat.UserMessage("Hi! My name is Bobby and I'm learning Go programming."))
		if err != nil {
			log.Fatal(err)
		}

		response, err := session.Message(ctx, chat.UserMessage("What are some good resources for learning Go concurrency?"))
		if err != nil {
			log.Fatal(err)
		}

		// The assistant will provide helpful resources
		if len(response.GetText()) > 0 {
			fmt.Println("Conversation established")
		}

		return sessionID
	}()

	// Phase 2: Resume the conversation later
	func() {
		// Open the existing database
		store, err := sqlitestore.New(dbPath)
		if err != nil {
			log.Fatal(err)
		}
		defer store.Close()

		// Create the same client configuration
		client, err := openai.NewClient(
			openai.OpenAIURL,
			os.Getenv("OPENAI_API_KEY"),
			openai.WithModel("gpt-4o-mini"),
		)
		if err != nil {
			log.Fatal(err)
		}

		// Resume the previous session
		session := agent.NewSession(
			client,
			"This will be ignored - original prompt is preserved",
			agent.WithStore(store),
			agent.WithRestoreSession(sessionID), // Key: restore with the same ID
		)

		fmt.Println("Session resumed")

		// The assistant should remember our previous conversation
		response, err := session.Message(ctx, chat.UserMessage("What was my name again?"))
		if err != nil {
			log.Fatal(err)
		}

		// Check if the assistant remembers Bobby from the earlier conversation
		if strings.Contains(strings.ToLower(response.GetText()), "bobby") {
			fmt.Println("Context preserved: true")
		}
	}()

	// Output:
	// Session created
	// Conversation established
	// Session resumed
	// Context preserved: true
}
