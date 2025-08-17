package schema

const URL = "http://json-schema.org/draft-07/schema#"

type Type string

const (
	String Type = "string"
	Array  Type = "array"
	Object Type = "object"
)

// JSON is a way to describe a JSON Schema
type JSON struct {
	Type                 interface{}      `json:"type,omitzero"` // Can be Type or []interface{} for union types like ["string", "null"]
	Description          string           `json:"description,omitzero"`
	Properties           map[string]*JSON `json:"properties,omitzero"`
	Items                *JSON            `json:"items,omitzero"`
	Enum                 []string         `json:"enum,omitzero"`
	Required             []string         `json:"required,omitzero"`
	AdditionalProperties *bool            `json:"additionalProperties,omitzero"`
	Schema               string           `json:"$schema,omitzero"`
	OneOf                []*JSON          `json:"oneOf,omitzero"`
	AnyOf                []*JSON          `json:"anyOf,omitzero"`
	AllOf                []*JSON          `json:"allOf,omitzero"`
}
