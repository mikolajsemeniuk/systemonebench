package metrics

import "math"

// ECE is the expected calibration error with equal-width confidence bins.
func ECE(probs []map[string]float64, labels []string, bins int) float64 {
	if len(probs) == 0 {
		return 0
	}
	count := make([]int, bins)
	conf := make([]float64, bins)
	hit := make([]float64, bins)
	for i, p := range probs {
		c := Confidence(p)
		b := min(int(c*float64(bins)), bins-1)
		count[b]++
		conf[b] += c
		if Argmax(p) == labels[i] {
			hit[b]++
		}
	}
	ece := 0.0
	for b := range bins {
		if count[b] > 0 {
			n := float64(count[b])
			ece += n / float64(len(probs)) * math.Abs(conf[b]/n-hit[b]/n)
		}
	}
	return ece
}
