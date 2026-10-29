// diagnose-k asks each example with k options (gold + k-1 random distractors),
// once in alphabetical order and once reversed, for several k.
// Output: output/diagnose-k/<model>.<dataset>.k<k>.<order>.jsonl
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mikolajsemeniuk/systemonebench/pkg/datasets"
	"github.com/mikolajsemeniuk/systemonebench/pkg/methods"
	"github.com/mikolajsemeniuk/systemonebench/pkg/records"
	"github.com/mikolajsemeniuk/systemonebench/pkg/systemone"
)

func main() {
	model := flag.String("model", "laya", "model name")
	url := flag.String("url", "http://localhost:8081", "llama-server base URL")
	name := flag.String("dataset", "clinc150", "dataset name")
	ks := flag.String("k", "3,6,12,24,59,77,150,255", "numbers of options")
	limit := flag.Int("limit", 500, "examples per k (0 = all test examples)")
	seed := flag.Int64("seed", 0, "distractor sampling seed")
	workers := flag.Int("workers", 4, "concurrent requests")
	out := flag.String("out", "output", "output directory")
	flag.Parse()

	ds, ok := datasets.ByName(*name)
	if !ok {
		log.Fatalf("unknown dataset %q", *name)
	}
	exs := ds.Test
	if *limit > 0 && *limit < len(exs) {
		exs = exs[:*limit]
	}
	client := systemone.New(*url)
	for _, s := range strings.Split(*ks, ",") {
		k, _ := strconv.Atoi(s)
		if k > len(ds.Options) {
			continue
		}
		for _, order := range []string{"sorted", "reversed"} {
			recs := run(client, ds, exs, k, order, *seed, *workers)
			path := filepath.Join(*out, "diagnose-k", fmt.Sprintf("%s.%s.k%d.%s.jsonl", *model, ds.Name, k, order))
			if err := records.Write(path, recs); err != nil {
				log.Fatal(err)
			}
			probs, labels := records.Split(recs)
			acc := 0.0
			for i, p := range probs {
				if argmax(p) == labels[i] {
					acc++
				}
			}
			fmt.Printf("%s %s k=%d %s: n=%d ok=%d acc=%.3f\n", *model, ds.Name, k, order, len(recs), len(probs), acc/float64(max(len(probs), 1)))
		}
	}
}

func run(c systemone.Client, ds datasets.Dataset, exs []datasets.Example, k int, order string, seed int64, workers int) []records.Record {
	recs := make([]records.Record, len(exs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	for i, ex := range exs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			// same distractors for both orders: seed depends on example and k only
			options := sample(ds.Options, ex.Label, k, rand.New(rand.NewSource(seed*1_000_003+int64(ex.ID)*1_000+int64(k))))
			ask := func(o []datasets.Option) (map[string]float64, error) {
				r, err := c.Choice(ex.State, ds.Instructions, o)
				return r.Probabilities, err
			}
			start := time.Now()
			var res methods.Result
			var err error
			if order == "reversed" {
				res, err = methods.Reversed(ask, options)
			} else {
				res, err = methods.Single(ask, options)
			}
			recs[i] = records.Record{ID: ex.ID, Label: ex.Label, Calls: res.Calls,
				LatencyMS: float64(time.Since(start).Microseconds()) / 1000}
			if err != nil {
				recs[i].Error = err.Error()
				return
			}
			recs[i].Choice = argmax(res.Probabilities)
			recs[i].Probabilities = res.Probabilities
		}()
	}
	wg.Wait()
	return recs
}

// sample returns gold + k-1 random distractors, sorted by name.
func sample(all []datasets.Option, gold string, k int, rng *rand.Rand) []datasets.Option {
	out := []datasets.Option{}
	perm := rng.Perm(len(all))
	for _, i := range perm {
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

func argmax(p map[string]float64) string {
	best, bestP := "", -1.0
	for k, v := range p {
		if v > bestP || (v == bestP && k < best) {
			best, bestP = k, v
		}
	}
	return best
}
