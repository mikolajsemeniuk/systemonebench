// energy measures GPU power draw via nvidia-smi and integrates it to joules.
package metrics

import (
	"bytes"
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// PowerSample is one nvidia-smi power.draw reading.
type PowerSample struct {
	T     time.Time
	Watts float64
}

// ReadPowerW runs one nvidia-smi query and returns the GPU power draw in watts.
func ReadPowerW() (float64, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return 0, err
	}
	return strconv.ParseFloat(strings.TrimSpace(out.String()), 64)
}

// PowerSampler polls nvidia-smi on a fixed interval until Stop is called.
type PowerSampler struct {
	samples []PowerSample
	cancel  context.CancelFunc
	done    chan struct{}
}

// StartPowerSampler begins polling GPU power draw every interval.
func StartPowerSampler(interval time.Duration) *PowerSampler {
	ctx, cancel := context.WithCancel(context.Background())
	s := &PowerSampler{cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(s.done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case t := <-ticker.C:
				if w, err := ReadPowerW(); err == nil {
					s.samples = append(s.samples, PowerSample{T: t, Watts: w})
				}
			}
		}
	}()
	return s
}

// Stop halts polling and returns the collected samples.
func (s *PowerSampler) Stop() []PowerSample {
	s.cancel()
	<-s.done
	return s.samples
}

// JoulesTrapezoid integrates watts over time (trapezoidal rule) to joules.
// Samples must be sorted by time; unsorted or single-sample input returns 0.
func JoulesTrapezoid(samples []PowerSample) float64 {
	var j float64
	for i := 1; i < len(samples); i++ {
		dt := samples[i].T.Sub(samples[i-1].T).Seconds()
		j += (samples[i].Watts + samples[i-1].Watts) / 2 * dt
	}
	return j
}

// MeanWatts is the plain average power draw across samples.
func MeanWatts(samples []PowerSample) float64 {
	if len(samples) == 0 {
		return 0
	}
	var sum float64
	for _, s := range samples {
		sum += s.Watts
	}
	return sum / float64(len(samples))
}
