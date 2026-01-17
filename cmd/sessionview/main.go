// Command sessionview is a CLI tool for viewing session data stored in SQLite.
//
// Usage:
//
//	sessionview list --db path/to/sessions.db
//	sessionview show --db path/to/sessions.db --session SESSION_ID [--format json|jsonl]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/bpowers/go-agent/persistence/sqlitestore"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	switch cmd {
	case "list":
		if err := runList(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "show":
		if err := runShow(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "-h", "--help", "help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `sessionview - view session data from SQLite

Usage:
  sessionview list --db <path>
      List all session IDs in the database

  sessionview show --db <path> --session <id> [--format json|jsonl]
      Show records for a session (default format: json)

Formats:
  json   - Output as a JSON array (default)
  jsonl  - Output as JSON Lines (one record per line)

Examples:
  sessionview list --db ./sessions.db
  sessionview show --db ./sessions.db --session abc123
  sessionview show --db ./sessions.db --session abc123 --format jsonl | jq .
`)
}

func runList(args []string) error {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	dbPath := fs.String("db", "", "path to SQLite database")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *dbPath == "" {
		return fmt.Errorf("--db is required")
	}

	store, err := sqlitestore.New(*dbPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer store.Close()

	sessions, err := store.ListSessions()
	if err != nil {
		return fmt.Errorf("list sessions: %w", err)
	}

	for _, s := range sessions {
		fmt.Println(s)
	}

	return nil
}

func runShow(args []string) error {
	fs := flag.NewFlagSet("show", flag.ExitOnError)
	dbPath := fs.String("db", "", "path to SQLite database")
	sessionID := fs.String("session", "", "session ID to display")
	format := fs.String("format", "json", "output format: json or jsonl")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *dbPath == "" {
		return fmt.Errorf("--db is required")
	}
	if *sessionID == "" {
		return fmt.Errorf("--session is required")
	}
	if *format != "json" && *format != "jsonl" {
		return fmt.Errorf("--format must be 'json' or 'jsonl'")
	}

	store, err := sqlitestore.New(*dbPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer store.Close()

	records, err := store.GetAllRecords(*sessionID)
	if err != nil {
		return fmt.Errorf("get records: %w", err)
	}

	if len(records) == 0 {
		fmt.Fprintf(os.Stderr, "no records found for session: %s\n", *sessionID)
		return nil
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")

	switch *format {
	case "json":
		if err := enc.Encode(records); err != nil {
			return fmt.Errorf("encode json: %w", err)
		}
	case "jsonl":
		enc.SetIndent("", "") // No indentation for JSONL
		for _, r := range records {
			if err := enc.Encode(r); err != nil {
				return fmt.Errorf("encode jsonl: %w", err)
			}
		}
	}

	return nil
}
