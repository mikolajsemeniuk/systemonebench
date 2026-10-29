package metrics

// MacroF1 is the unweighted mean of per-class F1 over classes present in labels or predictions.
func MacroF1(probs []map[string]float64, labels []string) float64 {
	tp, fp, fn := map[string]int{}, map[string]int{}, map[string]int{}
	for i, p := range probs {
		pred, gold := Argmax(p), labels[i]
		if pred == gold {
			tp[gold]++
		} else {
			fp[pred]++
			fn[gold]++
		}
	}
	classes := map[string]bool{}
	for _, m := range []map[string]int{tp, fp, fn} {
		for c := range m {
			classes[c] = true
		}
	}
	if len(classes) == 0 {
		return 0
	}
	sum := 0.0
	for c := range classes {
		denom := float64(2*tp[c] + fp[c] + fn[c])
		if denom > 0 {
			sum += 2 * float64(tp[c]) / denom
		}
	}
	return sum / float64(len(classes))
}
