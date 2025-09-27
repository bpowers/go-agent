package chat

import "context"

// SystemReminderFunc generates a system reminder string.
// It's called after tool execution completes, allowing it to
// incorporate tool results and system state changes.
type SystemReminderFunc func() string

// systemReminderKey is the context key for system reminders
type systemReminderKey struct{}

// WithSystemReminder attaches a system reminder generator to the context.
// The function will be called after tool execution to generate the reminder.
// Example:
//
//	filesModified := 0
//	ctx = chat.WithSystemReminder(ctx, func() string {
//	    if filesModified > 0 {
//	        return fmt.Sprintf("<system-reminder>Modified %d files</system-reminder>", filesModified)
//	    }
//	    return ""
//	})
func WithSystemReminder(ctx context.Context, reminderFunc SystemReminderFunc) context.Context {
	if reminderFunc == nil {
		return ctx
	}
	return context.WithValue(ctx, systemReminderKey{}, reminderFunc)
}

// GetSystemReminder retrieves the system reminder function from context.
// Returns nil if no reminder is set.
func GetSystemReminder(ctx context.Context) SystemReminderFunc {
	if f, ok := ctx.Value(systemReminderKey{}).(SystemReminderFunc); ok {
		return f
	}
	return nil
}
