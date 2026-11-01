// llm is the LLM reference baseline: one chat completion with all k options
// listed in the prompt (gold + k-1 random distractors, same sampling as
// diagnose-k). The answer is matched to an option name.
// Output: output/llm/<model>.<dataset>.k<k>.<split>.jsonl
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mikolajsemeniuk/systemonebench/pkg/datasets"
	"github.com/mikolajsemeniuk/systemonebench/pkg/llm"
	"github.com/mikolajsemeniuk/systemonebench/pkg/metrics"
	"github.com/mikolajsemeniuk/systemonebench/pkg/records"
)

func main() {
	model := flag.String("model", "qwen-7b", "model name")
	url := flag.String("url", "http://localhost:8090", "chat server base URL")
	name := flag.String("dataset", "clinc150", "dataset name")
	split := flag.String("split", "test", "calibration or test")
	k := flag.Int("k", 150, "number of options (0 = all)")
	limit := flag.Int("limit", 500, "examples (0 = all)")
	seed := flag.Int64("seed", 0, "seed for distractors")
	workers := flag.Int("workers", 4, "concurrent examples")
	out := flag.String("out", "output", "output directory")
	flag.Parse()

	ds, ok := datasets.ByName(*name)
	if !ok {
		log.Fatalf("unknown dataset %q", *name)
	}
	exs := ds.Test
	if *split == "calibration" {
		exs = ds.Calibration
	}
	if *limit > 0 && *limit < len(exs) {
		exs = exs[:*limit]
	}
	if *k == 0 || *k > len(ds.Options) {
		*k = len(ds.Options)
	}
	client := llm.New(*url)
	recs := make([]records.Record, len(exs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, *workers)
	for i, ex := range exs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			options := sample(ds.Options, ex.Label, *k, rand.New(rand.NewSource(*seed*1_000_003+int64(ex.ID)*1_000+int64(*k))))
			start := time.Now()
			text, err := client.Complete(prompt(ex.State, ds.Instructions, options), 64)
			recs[i] = records.Record{ID: ex.ID, Label: ex.Label, Calls: 1,
				LatencyMS: float64(time.Since(start).Microseconds()) / 1000}
			if err != nil {
				recs[i].Error = err.Error()
				return
			}
			choice := match(text, options)
			recs[i].Choice = choice
			recs[i].Probabilities = map[string]float64{choice: 1} // no probabilities from an LLM answer
		}()
	}
	wg.Wait()

	path := filepath.Join(*out, "llm", fmt.Sprintf("%s.%s.k%d.%s.jsonl", *model, ds.Name, *k, *split))
	if err := records.Write(path, recs); err != nil {
		log.Fatal(err)
	}
	probs, labels := records.Split(recs)
	fmt.Printf("%s %s k=%d %s: n=%d ok=%d acc=%.3f\n", *model, ds.Name, *k, *split, len(recs), len(probs), metrics.Accuracy(probs, labels))
}

// prompt lists the options as "name: description" lines and asks for one name.
func prompt(state json.RawMessage, instructions string, options []datasets.Option) string {
	var b strings.Builder
	b.WriteString(instructions)
	b.WriteString("\n\nInput:\n")
	b.Write(state)
	b.WriteString("\n\nOptions:\n")
	for _, o := range options {
		b.WriteString("- ")
		b.WriteString(o.Name)
		if o.Description != "" {
			b.WriteString(": ")
			b.WriteString(o.Description)
		}
		b.WriteString("\n")
	}
	b.WriteString("\nAnswer with the option name only.")
	return b.String()
}

// match returns the option whose name appears in the answer (longest match),
// or "" when none does.
func match(text string, options []datasets.Option) string {
	text = strings.ToLower(strings.TrimSpace(text))
	best := ""
	for _, o := range options {
		n := strings.ToLower(o.Name)
		if strings.Contains(text, n) && len(n) > len(best) {
			best = o.Name
		}
	}
	return best
}

// sample returns gold + k-1 random distractors, sorted by name (same as cmd/diagnose-k).
func sample(all []datasets.Option, gold string, k int, rng *rand.Rand) []datasets.Option {
	out := []datasets.Option{}
	for _, i := range rng.Perm(len(all)) {
		if all[i].Name == gold {
			continue
		}
		out = append(out, all[i])
		if len(out) == k-1 {
			break
		}
	}
	for _, o := range all {
		if o.Name == gold {
			out = append(out, o)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
