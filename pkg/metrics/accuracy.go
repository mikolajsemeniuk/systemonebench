package metrics

// Accuracy is the fraction of examples where argmax equals the label.
func Accuracy(probs []map[string]float64, labels []string) float64 {
	if len(probs) == 0 {
		return 0
	}
	hits := 0
	for i, p := range probs {
		if Argmax(p) == labels[i] {
			hits++
		}
	}
	return float64(hits) / float64(len(probs))
}
