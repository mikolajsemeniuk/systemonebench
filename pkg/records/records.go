// Package records is the JSONL format shared by all cmd programs (output/ files).
package records

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
)

// Record is one model decision for one example.
type Record struct {
	ID            int                `json:"id"`
	Label         string             `json:"label"`
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	LatencyMS     float64            `json:"latency_ms"`
	Calls         int                `json:"calls"` // model calls used for this decision
	Error         string             `json:"error,omitempty"`
}

// Write saves records as JSONL, creating parent directories.
func Write(path string, recs []Record) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, r := range recs {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return w.Flush()
}

// Read loads a JSONL file of records.
func Read(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Record
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, sc.Err()
}

// Split returns probabilities and labels of records without error, in order.
func Split(recs []Record) (probs []map[string]float64, labels []string) {
	for _, r := range recs {
		if r.Error == "" {
			probs = append(probs, r.Probabilities)
			labels = append(labels, r.Label)
		}
	}
	return probs, labels
}
