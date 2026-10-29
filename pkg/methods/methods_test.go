package methods

import (
	"math/rand"
	"testing"

	"github.com/mikolajsemeniuk/systemonebench/pkg/datasets"
)

// fake model: option "gold" always gets 0.9, the rest share the remainder.
func fake(options []datasets.Option) (map[string]float64, error) {
	p := map[string]float64{}
	rest := 1.0
	for _, o := range options {
		if o.Name == "gold" {
			p[o.Name] = 0.9
			rest = 0.1
		}
	}
	n := len(options)
	if rest == 0.1 {
		n--
	}
	for _, o := range options {
		if o.Name != "gold" {
			p[o.Name] = rest / float64(n)
		}
	}
	return p, nil
}

func opts(n int) []datasets.Option {
	out := []datasets.Option{{Name: "gold"}}
	for i := 1; i < n; i++ {
		out = append(out, datasets.Option{Name: string(rune('a' + i))})
	}
	return out
}

func TestTournamentFindsGold(t *testing.T) {
	r, err := Tournament(fake, opts(20), 4, rand.New(rand.NewSource(1)))
	if err != nil || argmax(r.Probabilities) != "gold" {
		t.Fatalf("got %v %v", r, err)
	}
	if r.Calls != 5+2+1 { // 20/4=5 groups -> 5 winners -> 2 groups -> 2 winners -> final
		t.Fatalf("calls = %d", r.Calls)
	}
}

func TestAdaptiveKeepsMore(t *testing.T) {
	r, err := Adaptive(fake, opts(20), 4, 0.01, rand.New(rand.NewSource(1)))
	if err != nil || argmax(r.Probabilities) != "gold" {
		t.Fatalf("got %v %v", r, err)
	}
}

func TestPermuteAverages(t *testing.T) {
	r, _ := Permute(fake, opts(5), 3, rand.New(rand.NewSource(1)))
	if r.Calls != 3 || r.Probabilities["gold"] < 0.89 {
		t.Fatalf("got %v", r)
	}
}

func TestChainKeepsGold(t *testing.T) {
	r, err := Chain(fake, opts(20), 4, 2, rand.New(rand.NewSource(1)))
	if err != nil || argmax(r.Probabilities) != "gold" || r.Calls != 8 {
		t.Fatalf("got %v %v", r, err)
	}
	sum := 0.0
	for _, v := range r.Probabilities {
		sum += v
	}
	if sum < 0.999 || sum > 1.001 {
		t.Fatalf("sum = %f", sum)
	}
}
