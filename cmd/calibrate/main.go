// calibrate runs the adaptive tournament for every eps in a grid on one split.
// cmd/eval-calibrate then picks eps per alpha from the calibration split and
// checks survival on the test split.
// Output: output/calibrate/<model>.<dataset>.k<k>.<method>-g<g>.eps<eps>.<split>.jsonl
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
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
	split := flag.String("split", "calibration", "calibration or test")
	k := flag.Int("k", 150, "number of options (0 = all)")
	method := flag.String("method", "adaptive", "adaptive | chain")
	g := flag.Int("g", 12, "group size")
	grid := flag.String("eps", "0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5", "eps grid")
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
	client := systemone.New(*url)
	for _, s := range strings.Split(*grid, ",") {
		eps, err := strconv.ParseFloat(s, 64)
		if err != nil {
			log.Fatal(err)
		}
		tag := fmt.Sprintf("%s-g%d.eps%s", *method, *g, strings.ReplaceAll(s, ".", "p"))
		path := filepath.Join(*out, "calibrate", fmt.Sprintf("%s.%s.k%d.%s.%s.jsonl", *model, ds.Name, *k, tag, *split))
		if _, err := os.Stat(path); err == nil {
			fmt.Println("skip, exists:", path)
			continue
		}
		recs := run(client, ds, exs, *k, *method, *g, eps, *seed, *workers)
		if err := records.Write(path, recs); err != nil {
			log.Fatal(err)
		}
		probs, labels := records.Split(recs)
		calls := 0
		for _, rec := range recs {
			calls += rec.Calls
		}
		fmt.Printf("%s %s k=%d %s %s: n=%d ok=%d acc=%.3f survival=%.3f calls/decision=%.1f\n",
			*model, ds.Name, *k, tag, *split, len(recs), len(probs), metrics.Accuracy(probs, labels),
			metrics.Survival(probs, labels), float64(calls)/float64(max(len(recs), 1)))
	}
}

func run(client systemone.Client, ds datasets.Dataset, exs []datasets.Example, k int, method string, g int, eps float64, seed int64, workers int) []records.Record {
	recs := make([]records.Record, len(exs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	for i, ex := range exs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			// same distractors and shuffles as cmd/tournament
			options := sample(ds.Options, ex.Label, k, rand.New(rand.NewSource(seed*1_000_003+int64(ex.ID)*1_000+int64(k))))
			rng := rand.New(rand.NewSource(seed*7_919 + int64(ex.ID)))
			ask := func(o []datasets.Option) (map[string]float64, error) {
				res, err := client.Choice(ex.State, ds.Instructions, o)
				return res.Probabilities, err
			}
			start := time.Now()
			var res methods.Result
			var err error
			if method == "chain" {
				res, err = methods.Chain(ask, options, g, eps, rng)
			} else {
				res, err = methods.Adaptive(ask, options, g, eps, rng)
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
	return recs
}

// sample returns gold + k-1 random distractors, sorted by name (same as cmd/tournament).
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
