// baseline asks the model once per example with all options in alphabetical order.
// Output: output/baseline/<model>.<dataset>.<split>.jsonl
package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"sync"

	"github.com/mikolajsemeniuk/systemonebench/pkg/datasets"
	"github.com/mikolajsemeniuk/systemonebench/pkg/metrics"
	"github.com/mikolajsemeniuk/systemonebench/pkg/records"
	"github.com/mikolajsemeniuk/systemonebench/pkg/systemone"
)

func main() {
	model := flag.String("model", "laya", "model name (used in output file name)")
	url := flag.String("url", "http://localhost:8081", "llama-server base URL")
	name := flag.String("dataset", "emotion", "dataset name")
	split := flag.String("split", "test", "calibration or test")
	limit := flag.Int("limit", 0, "max examples (0 = all)")
	workers := flag.Int("workers", 4, "concurrent requests")
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

	client := systemone.New(*url)
	recs := make([]records.Record, len(exs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, *workers)
	for i, ex := range exs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			recs[i] = decide(client, ds, ex)
		}()
	}
	wg.Wait()

	path := filepath.Join(*out, "baseline", fmt.Sprintf("%s.%s.%s.jsonl", *model, ds.Name, *split))
	if err := records.Write(path, recs); err != nil {
		log.Fatal(err)
	}
	probs, labels := records.Split(recs)
	fmt.Printf("%s %s %s: n=%d ok=%d acc=%.3f\n", *model, ds.Name, *split, len(recs), len(probs), metrics.Accuracy(probs, labels))
}

func decide(c systemone.Client, ds datasets.Dataset, ex datasets.Example) records.Record {
	r := records.Record{ID: ex.ID, Label: ex.Label, Calls: 1}
	res, err := c.Choice(ex.State, ds.Instructions, ds.Options)
	if err != nil {
		r.Error = err.Error()
		return r
	}
	r.Choice = res.Choice
	r.Probabilities = res.Probabilities
	r.LatencyMS = float64(res.Latency.Microseconds()) / 1000
	return r
}
