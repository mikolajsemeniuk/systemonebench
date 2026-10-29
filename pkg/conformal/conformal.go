// Package conformal picks the threshold eps of the adaptive tournament so that
// the gold option reaches the final with probability >= 1 - alpha.
//
// Survival is not monotone in eps (too small eps = many rounds = many chances to
// lose the gold; too large eps = plain tournament), so conformal risk control
// cannot be applied directly. Instead the calibration split is cut in two:
//
//	half A picks eps:      largest eps whose miss rate on A is <= alpha
//	half B certifies eps:  (misses_B + 1) / (n_B + 1) <= alpha
//
// Only one eps is tested on B, so the certificate needs no monotonicity.
package conformal

import "sort"

// Candidate is one eps evaluated on one half of the calibration split.
type Candidate struct {
	Eps    float64
	N      int // examples
	Misses int // gold not among finalists
}

// Risk is the finite-sample corrected miss rate of a candidate.
func Risk(c Candidate) float64 { return float64(c.Misses+1) / float64(c.N+1) }

// Pick returns the largest eps with empirical miss rate <= alpha on half A.
// ok is false when no candidate qualifies.
func Pick(a []Candidate, alpha float64) (eps float64, ok bool) {
	sorted := append([]Candidate(nil), a...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Eps > sorted[j].Eps })
	for _, c := range sorted {
		if float64(c.Misses)/float64(max(c.N, 1)) <= alpha {
			return c.Eps, true
		}
	}
	return 0, false
}

// Certify reports whether eps passes the corrected risk bound on half B.
func Certify(b []Candidate, eps, alpha float64) bool {
	for _, c := range b {
		if c.Eps == eps {
			return Risk(c) <= alpha
		}
	}
	return false
}
