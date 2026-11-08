// energy measures GPU power draw (nvidia-smi) while a model answers one
// decision per example, sequentially, and reports joules per decision.
// Run on the GPU machine (GX10), not on the Mac.
// Output: output/energy/<model>.<dataset>.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/mikolajsemeniuk/systemonebench/pkg/datasets"
	"github.com/mikolajsemeniuk/systemonebench/pkg/methods"
	"github.com/mikolajsemeniuk/systemonebench/pkg/metrics"
	"github.com/mikolajsemeniuk/systemonebench/pkg/systemone"
)

// Summary is one model x dataset energy measurement.
type Summary struct {
	Model         string  `json:"model"`
	Dataset       string  `json:"dataset"`
	Method        string  `json:"method"` // single or tournament-g<G>
	N             int     `json:"n"`
	Calls         int     `json:"calls"` // total model calls over the run
	DurationS     float64 `json:"duration_s"`
	IdleWatts     float64 `json:"idle_watts"`    // mean power before any request
	MeanWatts     float64 `json:"mean_watts"`    // mean power during the run
	TotalJoules   float64 `json:"total_joules"`  // integrated power during the run
	ActiveJoules  float64 `json:"active_joules"` // total minus idle*duration
	JoulesPerCall float64 `json:"joules_per_call"`
}

func main() {
	model := flag.String("model", "laya", "model name (used in output file name)")
	url := flag.String("url", "http://localhost:8081", "llama-server base URL")
	name := flag.String("dataset", "emotion", "dataset name")
	limit := flag.Int("limit", 100, "number of examples (sequential)")
	method := flag.String("method", "single", "single or tournament")
	g := flag.Int("g", 24, "group size for -method tournament")
	seed := flag.Int64("seed", 0, "shuffle seed for the tournament")
	idleS := flag.Int("idle", 5, "seconds to measure idle GPU power before the run")
	interval := flag.Duration("interval", 200*time.Millisecond, "nvidia-smi poll interval")
	out := flag.String("out", "output", "output directory")
	flag.Parse()

	if _, err := metrics.ReadPowerW(); err != nil {
		log.Fatalf("nvidia-smi not usable (run this on the GPU machine, not the Mac): %v", err)
	}

	ds, ok := datasets.ByName(*name)
	if !ok {
		log.Fatalf("unknown dataset %q", *name)
	}
	exs := ds.Test
	if *limit > 0 && *limit < len(exs) {
		exs = exs[:*limit]
	}

	idle := metrics.StartPowerSampler(*interval)
	time.Sleep(time.Duration(*idleS) * time.Second)
	idleWatts := metrics.MeanWatts(idle.Stop())

	client := systemone.New(*url)
	sampler := metrics.StartPowerSampler(*interval)
	start := time.Now()
	n, calls := 0, 0
	for _, ex := range exs {
		ask := func(o []datasets.Option) (map[string]float64, error) {
			res, err := client.Choice(ex.State, ds.Instructions, o)
			return res.Probabilities, err
		}
		var res methods.Result
		var err error
		switch *method {
		case "single":
			res, err = methods.Single(ask, ds.Options)
		case "tournament":
			res, err = methods.Tournament(ask, ds.Options, *g, rand.New(rand.NewSource(*seed*7_919+int64(ex.ID))))
		default:
			log.Fatalf("unknown method %q", *method)
		}
		if err != nil {
			log.Printf("call failed, skipping: %v", err)
			continue
		}
		n++
		calls += res.Calls
	}
	duration := time.Since(start)
	samples := sampler.Stop()

	total := metrics.JoulesTrapezoid(samples)
	active := total - idleWatts*duration.Seconds()
	tag := *method
	if *method == "tournament" {
		tag = fmt.Sprintf("tournament-g%d", *g)
	}
	s := Summary{
		Model: *model, Dataset: ds.Name, Method: tag, N: n, Calls: calls,
		DurationS: duration.Seconds(), IdleWatts: idleWatts,
		MeanWatts: metrics.MeanWatts(samples), TotalJoules: total,
		ActiveJoules: active,
	}
	if n > 0 {
		s.JoulesPerCall = active / float64(n)
	}

	path := filepath.Join(*out, "energy", fmt.Sprintf("%s.%s.%s.json", *model, ds.Name, tag))
	if err := writeJSON(path, s); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s %s %s: n=%d calls=%d idle=%.1fW mean=%.1fW active=%.1fJ -> %.3f J/decision\n",
		s.Model, s.Dataset, s.Method, s.N, s.Calls, s.IdleWatts, s.MeanWatts, s.ActiveJoules, s.JoulesPerCall)
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
