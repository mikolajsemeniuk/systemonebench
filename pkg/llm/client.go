// Package llm is a minimal client for an OpenAI-compatible /v1/chat/completions
// server (llama-server with a chat model). Used as the LLM reference baseline.
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client talks to one chat server.
type Client struct {
	URL  string
	HTTP *http.Client
}

// New returns a client for the given base URL (e.g. http://localhost:8090).
func New(url string) Client {
	return Client{URL: url, HTTP: &http.Client{Timeout: 30 * time.Minute}}
}

// Complete sends one user message and returns the assistant text (greedy).
func (c Client) Complete(prompt string, maxTokens int) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":  maxTokens,
		"temperature": 0,
	})
	resp, err := c.HTTP.Post(c.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%s: %s", resp.Status, raw)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("no choices: %s", raw)
	}
	return out.Choices[0].Message.Content, nil
}
