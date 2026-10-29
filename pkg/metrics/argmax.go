// Package metrics holds one metric per file. All take predicted probabilities
// (option -> probability) and gold labels of the same length.
package metrics

// Argmax returns the option with the highest probability (ties: lexicographically first).
func Argmax(p map[string]float64) string {
	best, bestP := "", -1.0
	for k, v := range p {
		if v > bestP || (v == bestP && k < best) {
			best, bestP = k, v
		}
	}
	return best
}

// Confidence returns the highest probability.
func Confidence(p map[string]float64) float64 {
	return p[Argmax(p)]
}

// Margin returns top1 - top2 probability.
func Margin(p map[string]float64) float64 {
	top1, top2 := 0.0, 0.0
	for _, v := range p {
		if v > top1 {
			top1, top2 = v, top1
		} else if v > top2 {
			top2 = v
		}
	}
	return top1 - top2
}
