// Package systemone is a minimal client for the llama.cpp /v1/systemone endpoint.
package systemone

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/mikolajsemeniuk/systemonebench/pkg/datasets"
)

// Client talks to one llama-server.
type Client struct {
	URL  string
	HTTP *http.Client
}

// New returns a client for the given base URL (e.g. http://localhost:8081).
func New(url string) Client {
	return Client{URL: url, HTTP: &http.Client{Timeout: 30 * time.Minute}}
}

// Result is the model answer to one choice question.
type Result struct {
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Latency       time.Duration      `json:"latency"`
}

// Choice asks one "choice" question. Options are sent in the given order
// (the server keeps JSON object order, so order matters for the model).
func (c Client) Choice(state json.RawMessage, instructions string, options []datasets.Option) (Result, error) {
	body, err := request(state, instructions, options)
	if err != nil {
		return Result{}, err
	}
	start := time.Now()
	resp, err := c.HTTP.Post(c.URL+"/v1/systemone", "application/json", bytes.NewReader(body))
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return Result{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return Result{}, fmt.Errorf("%s: %s", resp.Status, raw)
	}
	var out struct {
		Answers struct {
			Q Result `json:"q"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return Result{}, err
	}
	out.Answers.Q.Latency = time.Since(start)
	return out.Answers.Q, nil
}

// request builds the JSON body by hand so that option order is preserved
// (encoding/json sorts map keys).
func request(state json.RawMessage, instructions string, options []datasets.Option) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"state":`)
	b.Write(state)
	b.WriteString(`,"questions":{"q":{"type":"choice","instructions":`)
	if err := writeString(&b, instructions); err != nil {
		return nil, err
	}
	b.WriteString(`,"criteria":{`)
	for i, o := range options {
		if i > 0 {
			b.WriteByte(',')
		}
		if err := writeString(&b, o.Name); err != nil {
			return nil, err
		}
		b.WriteByte(':')
		if o.Description == "" {
			b.WriteString("null")
		} else if err := writeString(&b, o.Description); err != nil {
			return nil, err
		}
	}
	b.WriteString(`}}}}`)
	return b.Bytes(), nil
}

func writeString(b *bytes.Buffer, s string) error {
	enc, err := json.Marshal(s)
	if err != nil {
		return err
	}
	b.Write(enc)
	return nil
}
