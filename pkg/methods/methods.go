// Package methods contains inference-time strategies for choice questions.
// Every method takes the options and an Ask function and returns the final
// probabilities plus the number of model calls used.
package methods

import (
	"math/rand"

	"github.com/mikolajsemeniuk/systemonebench/pkg/datasets"
)

// Ask asks the model one choice question with the given options (in this order).
type Ask func(options []datasets.Option) (map[string]float64, error)

// Result is the output of a method.
type Result struct {
	Probabilities map[string]float64
	Calls         int
}

// Single asks once with the options as given.
func Single(ask Ask, options []datasets.Option) (Result, error) {
	p, err := ask(options)
	return Result{p, 1}, err
}

// Reversed asks once with the options in reverse order.
func Reversed(ask Ask, options []datasets.Option) (Result, error) {
	return Single(ask, reverse(options))
}

// Permute asks R times with shuffled options and averages the probabilities.
func Permute(ask Ask, options []datasets.Option, r int, rng *rand.Rand) (Result, error) {
	var runs []map[string]float64
	for range r {
		p, err := ask(shuffle(options, rng))
		if err != nil {
			return Result{}, err
		}
		runs = append(runs, p)
	}
	return Result{Average(runs), r}, nil
}

// Tournament splits the shuffled options into groups of g, keeps the winner of
// each group and asks a final question among the winners. Repeats while more
// than g candidates remain. Returns the final probabilities (over finalists).
func Tournament(ask Ask, options []datasets.Option, g int, rng *rand.Rand) (Result, error) {
	return Adaptive(ask, options, g, 2, rng) // eps > 1: only the winner passes
}

// Adaptive is Tournament where every option with probability >= eps passes a
// round (the winner always passes). Lower eps keeps more candidates.
func Adaptive(ask Ask, options []datasets.Option, g int, eps float64, rng *rand.Rand) (Result, error) {
	return run(ask, options, g, eps, false, rng)
}

// Chain is Adaptive where the final probability of an option is multiplied by
// the probabilities it got in every earlier round (then renormalised). A group
// winner with p=0.51 enters the final weaker than one with p=0.99.
func Chain(ask Ask, options []datasets.Option, g int, eps float64, rng *rand.Rand) (Result, error) {
	return run(ask, options, g, eps, true, rng)
}

func run(ask Ask, options []datasets.Option, g int, eps float64, chain bool, rng *rand.Rand) (Result, error) {
	calls := 0
	score := map[string]float64{} // product of probabilities over earlier rounds
	for _, o := range options {
		score[o.Name] = 1
	}
	cands := shuffle(options, rng)
	for len(cands) > g {
		var next []datasets.Option
		for _, group := range groups(cands, g) {
			p, err := ask(group)
			if err != nil {
				return Result{}, err
			}
			calls++
			winner := argmax(p)
			for _, o := range group {
				score[o.Name] *= p[o.Name]
				if o.Name == winner || p[o.Name] >= eps {
					next = append(next, o)
				}
			}
		}
		if len(next) == len(cands) { // nothing eliminated: force winners only
			next = winners(ask, cands, g, &calls)
		}
		cands = shuffle(next, rng)
	}
	p, err := ask(cands)
	if err != nil {
		return Result{}, err
	}
	if chain {
		p = Multiply(p, score)
	}
	return Result{p, calls + 1}, nil
}

// Multiply returns p[k]*w[k] renormalised to sum 1 over the keys of p.
func Multiply(p, w map[string]float64) map[string]float64 {
	out := map[string]float64{}
	sum := 0.0
	for k, v := range p {
		out[k] = v * w[k]
		sum += out[k]
	}
	if sum == 0 {
		return p
	}
	for k := range out {
		out[k] /= sum
	}
	return out
}

// Average returns the mean of the probability maps (missing keys count as 0).
func Average(runs []map[string]float64) map[string]float64 {
	out := map[string]float64{}
	for _, p := range runs {
		for k, v := range p {
			out[k] += v / float64(len(runs))
		}
	}
	return out
}

func winners(ask Ask, cands []datasets.Option, g int, calls *int) []datasets.Option {
	var out []datasets.Option
	for _, group := range groups(cands, g) {
		p, err := ask(group)
		if err != nil {
			return cands[:g]
		}
		*calls++
		for _, o := range group {
			if o.Name == argmax(p) {
				out = append(out, o)
			}
		}
	}
	return out
}

func groups(options []datasets.Option, g int) [][]datasets.Option {
	var out [][]datasets.Option
	for i := 0; i < len(options); i += g {
		out = append(out, options[i:min(i+g, len(options))])
	}
	return out
}

func shuffle(options []datasets.Option, rng *rand.Rand) []datasets.Option {
	out := append([]datasets.Option(nil), options...)
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

func reverse(options []datasets.Option) []datasets.Option {
	out := make([]datasets.Option, len(options))
	for i, o := range options {
		out[len(options)-1-i] = o
	}
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
