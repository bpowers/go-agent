package main

import "context"

type DatasetSelect struct {
	Variables []string
	TimeRange struct {
		Start string
		End   string
	}
}

type DatasetGetRequest struct {
	DatasetId string         `json:"datasetId"`
	Where     *DatasetSelect `json:"where"`
}

type DatasetGetResult struct {
	Revision string
	Data     map[string][][]float64
}

func DatasetGet(ctx context.Context, args DatasetGetRequest) (DatasetGetResult, error) {
	// In a real implementation, ctx would be used for cancellation, deadlines, etc.
	return DatasetGetResult{
		Revision: "1",
		Data: map[string][][]float64{
			"a": {
				{0, 1},
				{1, 6.3},
			},
			"b": {
				{0, 1},
				{1, 2},
			},
		},
	}, nil
}
