// tournament runs one method (single, permute, tournament, adaptive) on a
// dataset with k options (gold + random distractors, same sampling as diagnose-k).
// Output: output/tournament/<model>.<dataset>.k<k>.<method>.jsonl
package main

import (
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
	"github.com/mikolajsemeniuk/systemonebench/pkg/methods"
	"github.com/mikolajsemeniuk/systemonebench/pkg/metrics"
	"github.com/mikolajsemeniuk/systemonebench/pkg/records"
	"github.com/mikolajsemeniuk/systemonebench/pkg/systemone"
)

func main() {
	model := flag.String("model", "laya", "model name")
	url := flag.String("url", "http://localhost:8081", "llama-server base URL")
	name := flag.String("dataset", "clinc150", "dataset name")
	split := flag.String("split", "test", "calibration or test")
	k := flag.Int("k", 150, "number of options (0 = all)")
	method := flag.String("method", "tournament", "single | permute | tournament | adaptive | chain")
	r := flag.Int("r", 8, "permute: number of shuffles")
	g := flag.Int("g", 12, "tournament/adaptive: group size")
	eps := flag.Float64("eps", 0.1, "adaptive: options with p >= eps pass a round")
	limit := flag.Int("limit", 500, "examples (0 = all)")
	seed := flag.Int64("seed", 0, "seed for distractors and shuffles")
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
	tag := *method
	switch *method {
	case "permute":
		tag = fmt.Sprintf("permute-r%d", *r)
	case "tournament":
		tag = fmt.Sprintf("tournament-g%d", *g)
	case "adaptive":
		tag = fmt.Sprintf("adaptive-g%d-eps%s", *g, noDot(*eps))
	case "chain":
		tag = fmt.Sprintf("chain-g%d-eps%s", *g, noDot(*eps))
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
			// same distractors as diagnose-k for this example and k
			options := sample(ds.Options, ex.Label, *k, rand.New(rand.NewSource(*seed*1_000_003+int64(ex.ID)*1_000+int64(*k))))
			rng := rand.New(rand.NewSource(*seed*7_919 + int64(ex.ID)))
			ask := func(o []datasets.Option) (map[string]float64, error) {
				res, err := client.Choice(ex.State, ds.Instructions, o)
				return res.Probabilities, err
			}
			start := time.Now()
			var res methods.Result
			var err error
			switch *method {
			case "single":
				res, err = methods.Single(ask, options)
			case "permute":
				res, err = methods.Permute(ask, options, *r, rng)
			case "tournament":
				res, err = methods.Tournament(ask, options, *g, rng)
			case "adaptive":
				res, err = methods.Adaptive(ask, options, *g, *eps, rng)
			case "chain":
				res, err = methods.Chain(ask, options, *g, *eps, rng)
			default:
				log.Fatalf("unknown method %q", *method)
			}
			recs[i] = records.Record{ID: ex.ID, Label: ex.Label, Calls: res.Calls,
				LatencyMS: float64(time.Since(start).Microseconds()) / 1000}
			if err != nil {
				recs[i].Error = err.Error()
				return
			}
			recs[i].Choice = metrics.Argmax(res.Probabilities)
			recs[i].Probabilities = res.Probabilities
		}()
	}
	wg.Wait()

	path := filepath.Join(*out, "tournament", fmt.Sprintf("%s.%s.k%d.%s.%s.jsonl", *model, ds.Name, *k, tag, *split))
	if err := records.Write(path, recs); err != nil {
		log.Fatal(err)
	}
	probs, labels := records.Split(recs)
	calls := 0
	for _, rec := range recs {
		calls += rec.Calls
	}
	fmt.Printf("%s %s k=%d %s %s: n=%d ok=%d acc=%.3f ece=%.3f calls/decision=%.1f\n",
		*model, ds.Name, *k, tag, *split, len(recs), len(probs), metrics.Accuracy(probs, labels),
		metrics.ECE(probs, labels, 10), float64(calls)/float64(max(len(recs), 1)))
}

// noDot formats eps without dots (file names use dots as separators): 0.05 -> 0p05.
func noDot(eps float64) string { return strings.ReplaceAll(fmt.Sprint(eps), ".", "p") }

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
