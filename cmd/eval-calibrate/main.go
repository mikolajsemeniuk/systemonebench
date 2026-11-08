// eval-calibrate reads output/calibrate/*.jsonl, picks eps per alpha on the
// calibration split (pkg/conformal) and reports survival / accuracy / calls of
// that eps on the test split.
// Output: output/eval/calibrate.json, paper/calibrate.gen.tex
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/mikolajsemeniuk/systemonebench/pkg/conformal"
	"github.com/mikolajsemeniuk/systemonebench/pkg/metrics"
	"github.com/mikolajsemeniuk/systemonebench/pkg/records"
)

// Point is one eps on one split. MissesA/MissesB and NA/NB are the two halves
// of the calibration split (even / odd example id).
type Point struct {
	Eps      float64 `json:"eps"`
	N        int     `json:"n"`
	Misses   int     `json:"misses"`
	NA       int     `json:"n_a,omitempty"`
	MissesA  int     `json:"misses_a,omitempty"`
	NB       int     `json:"n_b,omitempty"`
	MissesB  int     `json:"misses_b,omitempty"`
	Survival float64 `json:"survival"`
	Accuracy float64 `json:"accuracy"`
	Calls    float64 `json:"calls_per_decision"`
}

// Pick is the eps chosen for one alpha and its test-split result.
type Pick struct {
	Alpha        float64 `json:"alpha"`
	Eps          float64 `json:"eps"`
	Found        bool    `json:"found"`     // some eps had miss rate <= alpha on half A
	Certified    bool    `json:"certified"` // (misses_B+1)/(n_B+1) <= alpha
	TestSurvival float64 `json:"test_survival"`
	TestAccuracy float64 `json:"test_accuracy"`
	TestCalls    float64 `json:"test_calls_per_decision"`
}

// Row is one model × dataset × k × method-g.
type Row struct {
	Model       string  `json:"model"`
	Dataset     string  `json:"dataset"`
	K           int     `json:"k"`
	Method      string  `json:"method"`
	Calibration []Point `json:"calibration"`
	Test        []Point `json:"test"`
	Picks       []Pick  `json:"picks"`
}

func main() {
	in := flag.String("in", "output/calibrate", "input directory")
	alphas := flag.String("alpha", "0.01,0.05,0.1,0.2", "alpha grid")
	outJSON := flag.String("json", "output/eval/calibrate.json", "summary json")
	outTex := flag.String("tex", "paper/calibrate.gen.tex", "LaTeX table")
	flag.Parse()

	files, _ := filepath.Glob(filepath.Join(*in, "*.jsonl"))
	rows := map[string]Row{}
	for _, f := range files {
		// <model>.<dataset>.k<k>.<method>-g<g>.eps<eps>.<split>.jsonl; model may contain dots
		parts := strings.Split(strings.TrimSuffix(filepath.Base(f), ".jsonl"), ".")
		n := len(parts)
		k, _ := strconv.Atoi(strings.TrimPrefix(parts[n-4], "k"))
		eps, _ := strconv.ParseFloat(strings.ReplaceAll(strings.TrimPrefix(parts[n-2], "eps"), "p", "."), 64)
		key := strings.Join(parts[:n-2], ".")
		row := rows[key]
		row.Model, row.Dataset, row.K, row.Method = strings.Join(parts[:n-5], "."), parts[n-5], k, parts[n-3]
		recs, err := records.Read(f)
		if err != nil {
			log.Fatal(err)
		}
		pt := point(eps, recs)
		if parts[n-1] == "calibration" {
			row.Calibration = append(row.Calibration, pt)
		} else {
			row.Test = append(row.Test, pt)
		}
		rows[key] = row
	}
	var out []Row
	for _, row := range rows {
		sort.Slice(row.Calibration, func(i, j int) bool { return row.Calibration[i].Eps < row.Calibration[j].Eps })
		sort.Slice(row.Test, func(i, j int) bool { return row.Test[i].Eps < row.Test[j].Eps })
		for _, a := range strings.Split(*alphas, ",") {
			alpha, _ := strconv.ParseFloat(a, 64)
			row.Picks = append(row.Picks, pick(row, alpha))
		}
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Dataset+out[i].Model+out[i].Method < out[j].Dataset+out[j].Model+out[j].Method
	})
	if err := os.MkdirAll(filepath.Dir(*outJSON), 0o755); err != nil {
		log.Fatal(err)
	}
	b, _ := json.MarshalIndent(out, "", " ")
	if err := os.WriteFile(*outJSON, b, 0o644); err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*outTex, []byte(tex(out)), 0o644); err != nil {
		log.Fatal(err)
	}
	for _, r := range out {
		fmt.Printf("%s %s k=%d %s\n", r.Model, r.Dataset, r.K, r.Method)
		for _, p := range r.Calibration {
			fmt.Printf("  cal  eps=%.3f survival=%.3f acc=%.3f calls=%.1f\n", p.Eps, p.Survival, p.Accuracy, p.Calls)
		}
		for _, p := range r.Test {
			fmt.Printf("  test eps=%.3f survival=%.3f acc=%.3f calls=%.1f\n", p.Eps, p.Survival, p.Accuracy, p.Calls)
		}
		for _, p := range r.Picks {
			fmt.Printf("  alpha=%.2f -> eps=%.3f found=%v certified=%v test survival=%.3f (target >= %.2f) acc=%.3f calls=%.1f\n",
				p.Alpha, p.Eps, p.Found, p.Certified, p.TestSurvival, 1-p.Alpha, p.TestAccuracy, p.TestCalls)
		}
	}
}

func point(eps float64, recs []records.Record) Point {
	pt := Point{Eps: eps}
	calls := 0
	for _, r := range recs {
		if r.Error != "" {
			continue
		}
		calls += r.Calls
		pt.N++
		_, survived := r.Probabilities[r.Label]
		miss := 0
		if !survived {
			miss = 1
		}
		pt.Misses += miss
		if r.ID%2 == 0 {
			pt.NA++
			pt.MissesA += miss
		} else {
			pt.NB++
			pt.MissesB += miss
		}
	}
	probs, labels := records.Split(recs)
	pt.Survival = metrics.Survival(probs, labels)
	pt.Accuracy = metrics.Accuracy(probs, labels)
	pt.Calls = float64(calls) / float64(max(pt.N, 1))
	return pt
}

func pick(row Row, alpha float64) Pick {
	var a, b []conformal.Candidate
	for _, p := range row.Calibration {
		a = append(a, conformal.Candidate{Eps: p.Eps, N: p.NA, Misses: p.MissesA})
		b = append(b, conformal.Candidate{Eps: p.Eps, N: p.NB, Misses: p.MissesB})
	}
	eps, found := conformal.Pick(a, alpha)
	out := Pick{Alpha: alpha, Eps: eps, Found: found, Certified: found && conformal.Certify(b, eps, alpha)}
	for _, p := range row.Test {
		if p.Eps == eps {
			out.TestSurvival, out.TestAccuracy, out.TestCalls = p.Survival, p.Accuracy, p.Calls
		}
	}
	return out
}

func tex(rows []Row) string {
	var alphas []float64
	seen := map[float64]bool{}
	for _, r := range rows {
		for _, p := range r.Picks {
			if !seen[p.Alpha] {
				seen[p.Alpha] = true
				alphas = append(alphas, p.Alpha)
			}
		}
	}
	sort.Float64s(alphas)
	var b strings.Builder
	b.WriteString("% generated by cmd/eval-calibrate, do not edit\n")
	b.WriteString("\\begin{longtable}{lll")
	for range alphas {
		b.WriteString("rr")
	}
	b.WriteString("}\n\\toprule\n")
	head := " & & "
	for _, a := range alphas {
		head += fmt.Sprintf(" & \\multicolumn{2}{c}{$\\alpha=%g$}", a)
	}
	head += " \\\\\n"
	for i := range alphas {
		head += fmt.Sprintf("\\cmidrule(lr){%d-%d}", 4+2*i, 5+2*i)
	}
	head += "\nDataset & Model & $G$"
	for range alphas {
		head += " & $\\varepsilon$ & Surv."
	}
	head += " \\\\\n\\midrule\n"
	b.WriteString(head + "\\endfirsthead\n\\toprule\n" + head + "\\endhead\n")
	prev := ""
	for _, r := range rows {
		if prev != "" && prev != r.Dataset {
			b.WriteString("\\midrule\n")
		}
		prev = r.Dataset
		g := strings.TrimPrefix(r.Method, "adaptive-g")
		fmt.Fprintf(&b, "%s & %s & %s", esc(r.Dataset), esc(r.Model), g)
		for _, a := range alphas {
			cell := " & -- & --"
			for _, p := range r.Picks {
				if p.Alpha != a || !p.Found {
					continue
				}
				surv := fmt.Sprintf("%.3f", p.TestSurvival)
				if p.Certified {
					surv = "\\textbf{" + surv + "}"
				} else {
					surv = "(" + surv + ")"
				}
				cell = fmt.Sprintf(" & %g & %s", p.Eps, surv)
			}
			b.WriteString(cell)
		}
		b.WriteString(" \\\\\n")
	}
	b.WriteString("\\bottomrule\n\\end{longtable}\n")
	return b.String()
}

func esc(s string) string { return strings.ReplaceAll(s, "_", "\\_") }
