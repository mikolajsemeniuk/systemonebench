package metrics

// Survival is the fraction of examples whose label is among the finalists
// (keys of the final probability map). 1 - Survival is the conformal risk.
func Survival(probs []map[string]float64, labels []string) float64 {
	if len(probs) == 0 {
		return 0
	}
	hits := 0
	for i, p := range probs {
		if _, ok := p[labels[i]]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(probs))
}
