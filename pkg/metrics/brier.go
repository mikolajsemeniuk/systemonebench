package metrics

// Brier is the multi-class Brier score: mean over examples of sum over options of (p - onehot)^2.
func Brier(probs []map[string]float64, labels []string) float64 {
	if len(probs) == 0 {
		return 0
	}
	sum := 0.0
	for i, p := range probs {
		for k, v := range p {
			y := 0.0
			if k == labels[i] {
				y = 1
			}
			sum += (v - y) * (v - y)
		}
		if _, ok := p[labels[i]]; !ok {
			sum += 1 // label missing from options: full penalty
		}
	}
	return sum / float64(len(probs))
}
