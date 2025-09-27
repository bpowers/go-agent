package common

import (
	"encoding/json"
	"fmt"
)

// FormatToolErrorJSON formats a tool error as a JSON string.
// It attempts to create a properly formatted JSON object with an "error" field.
// If JSON marshaling fails, it falls back to a simple sprintf format.
func FormatToolErrorJSON(errorMsg string) string {
	if errorMsg == "" {
		return "{}"
	}
	payload, err := json.Marshal(map[string]string{"error": errorMsg})
	if err == nil {
		return string(payload)
	}
	// Fallback if JSON marshaling fails (shouldn't happen with string input)
	return fmt.Sprintf(`{"error": "%s"}`, errorMsg)
}
