// datasets downloads the benchmark datasets from Hugging Face and splits them
// into calibration / test. Output: pkg/datasets/<name>.{options.json,calibration.jsonl,test.jsonl}
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	rowsAPI = "https://datasets-server.huggingface.co/rows"
	bfclURL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/"
)

// Example is one dataset row; State is a string or an object.
type Example struct {
	ID    int             `json:"id"`
	State json.RawMessage `json:"state"`
	Label string          `json:"label"`
}

// Dataset is a loader: returns rows and options (option -> description or nil).
type Dataset struct {
	Name string
	Load func() ([]Example, map[string]*string)
}

var emotionLabels = []string{"sadness", "joy", "love", "anger", "fear", "surprise"}

var xnliLabels = []string{"entailment", "neutral", "contradiction"}

var xnliDesc = map[string]*string{
	"entailment":    ptr("the hypothesis is definitely true given the premise"),
	"neutral":       ptr("the hypothesis might be true given the premise"),
	"contradiction": ptr("the hypothesis is definitely false given the premise"),
}

var datasets = []Dataset{
	{"emotion", func() ([]Example, map[string]*string) {
		var out []Example
		for _, r := range rows("dair-ai/emotion", "split", "test") {
			out = append(out, Example{State: str(r["text"]), Label: emotionLabels[int(r["label"].(float64))]})
		}
		return out, names(emotionLabels)
	}},
	{"xnli", func() ([]Example, map[string]*string) {
		var out []Example
		for _, r := range rows("facebook/xnli", "en", "test") {
			state, _ := json.Marshal(map[string]any{"premise": r["premise"], "hypothesis": r["hypothesis"]})
			out = append(out, Example{State: state, Label: xnliLabels[int(r["label"].(float64))]})
		}
		return out, xnliDesc
	}},
	{"massive", func() ([]Example, map[string]*string) {
		var out []Example
		for _, r := range rows("mteb/amazon_massive_intent", "en", "test") {
			out = append(out, Example{State: str(r["text"]), Label: r["label"].(string)})
		}
		return out, labelsOf(out)
	}},
	{"banking77", func() ([]Example, map[string]*string) {
		var out []Example
		for _, r := range rows("mteb/banking77", "default", "test") {
			out = append(out, Example{State: str(r["text"]), Label: r["label_text"].(string)})
		}
		return out, labelsOf(out)
	}},
	{"clinc150", func() ([]Example, map[string]*string) {
		intents := map[int]string{}
		for _, r := range rows("DeepPavlov/clinc150", "intents", "intents") {
			intents[int(r["id"].(float64))] = r["name"].(string)
		}
		var out []Example
		for _, r := range rows("DeepPavlov/clinc150", "default", "test") {
			if r["label"] == nil { // out-of-scope rows (1000), skipped
				continue
			}
			out = append(out, Example{State: str(r["utterance"]), Label: intents[int(r["label"].(float64))]})
		}
		opts := map[string]*string{}
		for _, n := range intents {
			opts[n] = nil
		}
		return out, opts
	}},
	{"bfcl", func() ([]Example, map[string]*string) {
		var out []Example
		opts := map[string]*string{}
		for _, f := range []string{"BFCL_v3_simple.json", "BFCL_v3_live_simple.json"} {
			for _, line := range strings.Split(strings.TrimSpace(string(get(bfclURL+f))), "\n") {
				var x struct {
					Question [][]struct{ Content string }         `json:"question"`
					Function []struct{ Name, Description string } `json:"function"`
				}
				if err := json.Unmarshal([]byte(line), &x); err != nil {
					log.Fatal(err)
				}
				fn := x.Function[0]
				if _, ok := opts[fn.Name]; !ok {
					opts[fn.Name] = ptr(fn.Description)
				}
				out = append(out, Example{State: str(x.Question[0][0].Content), Label: fn.Name})
			}
		}
		return out, opts
	}},
}

func main() {
	out := flag.String("out", "pkg/datasets", "output directory")
	cal := flag.Float64("cal", 0.5, "calibration fraction")
	seed := flag.Int64("seed", 0, "shuffle seed")
	flag.Parse()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	for _, ds := range datasets {
		data, opts := ds.Load()
		for i := range data {
			data[i].ID = i
		}
		rand.New(rand.NewSource(*seed)).Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
		n := int(float64(len(data)) * *cal)
		writeJSON(filepath.Join(*out, ds.Name+".options.json"), opts)
		writeJSONL(filepath.Join(*out, ds.Name+".calibration.jsonl"), data[:n])
		writeJSONL(filepath.Join(*out, ds.Name+".test.jsonl"), data[n:])
		fmt.Printf("%s: %d rows, %d options, calibration %d, test %d\n", ds.Name, len(data), len(opts), n, len(data)-n)
	}
}

// rows pages through the datasets-server /rows API.
func rows(dataset, config, split string) []map[string]any {
	var out []map[string]any
	for off, total := 0, 100; off < total; off += 100 {
		q := url.Values{"dataset": {dataset}, "config": {config}, "split": {split},
			"offset": {fmt.Sprint(off)}, "length": {fmt.Sprint(min(100, total-off))}}
		var page struct {
			Rows  []struct{ Row map[string]any } `json:"rows"`
			Total int                            `json:"num_rows_total"`
		}
		if err := json.Unmarshal(get(rowsAPI+"?"+q.Encode()), &page); err != nil {
			log.Fatal(err)
		}
		for _, r := range page.Rows {
			out = append(out, r.Row)
		}
		total = page.Total
	}
	return out
}

// get downloads a URL with retries (the datasets-server returns random errors).
func get(u string) []byte {
	for attempt := 1; attempt <= 12; attempt++ {
		resp, err := http.Get(u)
		if err == nil && resp.StatusCode == http.StatusOK {
			b, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err == nil {
				return b
			}
		} else if err == nil {
			resp.Body.Close()
		}
		time.Sleep(time.Duration(5*attempt) * time.Second)
	}
	log.Fatalf("failed %s", u)
	return nil
}

func writeJSON(path string, v any) {
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		log.Fatal(err)
	}
}

func writeJSONL(path string, exs []Example) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	for _, e := range exs {
		b, _ := json.Marshal(e)
		w.Write(b)
		w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		log.Fatal(err)
	}
}

func str(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

func ptr(s string) *string { return &s }

func names(labels []string) map[string]*string {
	m := map[string]*string{}
	for _, l := range labels {
		m[l] = nil
	}
	return m
}

func labelsOf(exs []Example) map[string]*string {
	m := map[string]*string{}
	for _, e := range exs {
		m[e.Label] = nil
	}
	return m
}
