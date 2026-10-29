package metrics

import (
	"sort"
	"time"
)

// Percentile returns the p-th percentile (0-100) of the durations.
func Percentile(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	i := int(p / 100 * float64(len(s)-1))
	return s[i]
}
