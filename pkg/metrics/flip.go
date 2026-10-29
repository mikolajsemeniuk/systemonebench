package metrics

// FlipRate is the fraction of examples where argmax differs between two runs.
func FlipRate(a, b []map[string]float64) float64 {
	if len(a) == 0 {
		return 0
	}
	flips := 0
	for i := range a {
		if Argmax(a[i]) != Argmax(b[i]) {
			flips++
		}
	}
	return float64(flips) / float64(len(a))
}
