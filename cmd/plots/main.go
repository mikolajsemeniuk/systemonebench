// plots turns output/eval/*.json into pgfplots figures in paper/plots/*.gen.tex.
// Each file is one tikzpicture meant for \input in the paper.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type diagnoseRow struct {
	Model, Dataset string
	K              int
	Accuracy       float64 `json:"accuracy"`
	ECE            float64 `json:"ece"`
	Flip           float64 `json:"flip_rate"`
	P50            float64 `json:"p50_ms"`
}

type tournamentRow struct {
	Model, Dataset, Method string
	K                      int
	Accuracy               float64 `json:"accuracy"`
	ECE                    float64 `json:"ece"`
	Calls                  float64 `json:"calls_per_decision"`
	P50                    float64 `json:"p50_ms"`
}

type calibrateRow struct {
	Model, Dataset, Method string
	K                      int
	Calibration            []point `json:"calibration"`
	Test                   []point `json:"test"`
}

type point struct {
	Eps      float64 `json:"eps"`
	Survival float64 `json:"survival"`
	Calls    float64 `json:"calls_per_decision"`
}

// xy is one point of a series.
type xy struct{ X, Y float64 }

func main() {
	in := flag.String("in", "output/eval", "eval directory")
	out := flag.String("out", "paper/plots", "output directory")
	flag.Parse()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}

	var diag []diagnoseRow
	read(filepath.Join(*in, "diagnose-k.json"), &diag)
	for _, ds := range []string{"clinc150", "bfcl", "massive", "banking77"} {
		for _, m := range []struct {
			name, label string
			y           func(diagnoseRow) float64
		}{
			{"accuracy", "Accuracy", func(r diagnoseRow) float64 { return r.Accuracy }},
			{"flip", "Flip rate (sorted vs reversed)", func(r diagnoseRow) float64 { return r.Flip }},
			{"ece", "ECE", func(r diagnoseRow) float64 { return r.ECE }},
		} {
			series := map[string][]xy{}
			for _, r := range diag {
				if r.Dataset == ds {
					series[r.Model] = append(series[r.Model], xy{float64(r.K), m.y(r)})
				}
			}
			write(filepath.Join(*out, fmt.Sprintf("%s-%s-vs-k.gen.tex", m.name, ds)),
				lines(series, "number of options $k$", m.label, "xmode=log, log basis x=2"))
		}
	}

	// Laya on BFCL with and without descriptions
	nodesc := map[string][]xy{}
	for _, r := range diag {
		if r.Model == "laya" && (r.Dataset == "bfcl" || r.Dataset == "bfcl-nodesc") {
			nodesc[r.Dataset] = append(nodesc[r.Dataset], xy{float64(r.K), r.Accuracy})
		}
	}
	write(filepath.Join(*out, "accuracy-laya-bfcl-nodesc-vs-k.gen.tex"), lines(nodesc, "number of options $k$", "Accuracy (Laya)", "xmode=log, log basis x=2"))

	var tour []tournamentRow
	read(filepath.Join(*in, "tournament.json"), &tour)
	for _, ds := range []string{"clinc150", "bfcl", "massive", "banking77"} {
		acc, ece := map[string][]xy{}, map[string][]xy{}
		for _, r := range tour {
			if r.Dataset == ds && !strings.HasPrefix(r.Method, "chain") {
				acc[r.Model] = append(acc[r.Model], xy{r.Calls, r.Accuracy})
				ece[r.Model] = append(ece[r.Model], xy{r.Calls, r.ECE})
			}
		}
		write(filepath.Join(*out, fmt.Sprintf("accuracy-vs-calls-%s.gen.tex", ds)), lines(acc, "calls per decision", "Accuracy", "xmode=log"))
		write(filepath.Join(*out, fmt.Sprintf("ece-vs-calls-%s.gen.tex", ds)), lines(ece, "calls per decision", "ECE", "xmode=log"))
	}
	for _, ds := range []string{"clinc150", "bfcl", "massive", "banking77"} {
		series := map[string][]xy{}
		labels := map[string][]string{}
		for _, r := range tour {
			if r.Dataset == ds {
				series[r.Model] = append(series[r.Model], xy{r.P50, r.Accuracy})
				labels[r.Model] = append(labels[r.Model], r.Method)
			}
		}
		write(filepath.Join(*out, fmt.Sprintf("pareto-%s.gen.tex", ds)),
			scatter(series, labels, "latency p50 (ms)", "Accuracy", "xmode=log"))
	}

	var cal []calibrateRow
	read(filepath.Join(*in, "calibrate.json"), &cal)
	for _, r := range cal {
		series := map[string][]xy{}
		for _, p := range r.Calibration {
			series["calibration"] = append(series["calibration"], xy{p.Eps, p.Survival})
		}
		for _, p := range r.Test {
			series["test"] = append(series["test"], xy{p.Eps, p.Survival})
		}
		write(filepath.Join(*out, fmt.Sprintf("survival-%s-%s-%s.gen.tex", r.Model, r.Dataset, r.Method)),
			lines(series, "$\\varepsilon$", "Survival of the gold option", "xmode=log"))
	}
	// survival vs calls per decision, one line per G, per model x dataset (test split)
	groups := map[string]map[string][]xy{}
	for _, r := range cal {
		key := r.Model + "-" + r.Dataset
		if groups[key] == nil {
			groups[key] = map[string][]xy{}
		}
		for _, p := range r.Test {
			groups[key][r.Method] = append(groups[key][r.Method], xy{p.Calls, p.Survival})
		}
	}
	for key, series := range groups {
		write(filepath.Join(*out, fmt.Sprintf("survival-vs-calls-%s.gen.tex", key)),
			lines(series, "calls per decision", "Survival (test)", "xmode=log"))
	}
}

func read(path string, v any) {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		log.Fatal(err)
	}
}

func write(path, tex string) {
	if err := os.WriteFile(path, []byte("% generated by cmd/plots, do not edit\n"+tex), 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)
}

// lines draws one line per series, sorted by x.
func lines(series map[string][]xy, xlabel, ylabel, extra string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "\\begin{tikzpicture}\n\\begin{axis}[xlabel={%s}, ylabel={%s}, %s, legend pos=south west, legend style={font=\\scriptsize}, label style={font=\\small}, tick label style={font=\\scriptsize}, grid=major, width=\\plotwidth, height=0.72\\plotwidth]\n", xlabel, ylabel, extra)
	for _, name := range keys(series) {
		pts := series[name]
		sort.Slice(pts, func(i, j int) bool { return pts[i].X < pts[j].X })
		b.WriteString("\\addplot coordinates {")
		for _, p := range pts {
			fmt.Fprintf(&b, " (%s,%.4f)", num(p.X), p.Y)
		}
		fmt.Fprintf(&b, " };\n\\addlegendentry{%s}\n", esc(name))
	}
	b.WriteString("\\end{axis}\n\\end{tikzpicture}\n")
	return b.String()
}

// scatter draws one mark set per series with a label at each point.
func scatter(series map[string][]xy, labels map[string][]string, xlabel, ylabel, extra string) string {
	colors := map[string]string{"laya": "black", "kev-0.8b": "blue", "kev-4b": "orange", "kev-9b": "red"}
	families := []struct{ prefix, mark, label string }{
		{"single", "*", "single"},
		{"permute", "square*", "permute"},
		{"tournament", "triangle*", "tournament"},
		{"adaptive", "diamond*", "adaptive"},
		{"chain", "o", "chain"},
	}
	family := func(method string) int {
		for i, f := range families {
			if strings.HasPrefix(method, f.prefix) {
				return i
			}
		}
		return 0
	}
	var b strings.Builder
	fmt.Fprintf(&b, "\\begin{tikzpicture}\n\\begin{axis}[xlabel={%s}, ylabel={%s}, %s, legend style={at={(0.98,0.03)}, anchor=south east, legend columns=3, font=\\scriptsize, draw=gray!50}, label style={font=\\small}, tick label style={font=\\scriptsize}, grid=major, width=\\plotwidth, height=0.55\\plotwidth]\n", xlabel, ylabel, extra)
	for _, name := range keys(series) {
		color := colors[name]
		if color == "" {
			color = "gray"
		}
		fmt.Fprintf(&b, "\\addlegendimage{only marks, mark=*, %s}\n\\addlegendentry{%s}\n", color, esc(name))
		for i, p := range series[name] {
			fmt.Fprintf(&b, "\\addplot[only marks, mark=%s, mark size=2.2pt, %s, forget plot] coordinates { (%s,%.4f) };\n", families[family(labels[name][i])].mark, color, num(p.X), p.Y)
		}
	}
	for _, f := range families {
		fmt.Fprintf(&b, "\\addlegendimage{only marks, mark=%s, black}\n\\addlegendentry{%s}\n", f.mark, f.label)
	}
	b.WriteString("\\end{axis}\n\\end{tikzpicture}\n")
	return b.String()
}

func keys(m map[string][]xy) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func num(x float64) string {
	if x == math.Trunc(x) {
		return fmt.Sprintf("%d", int64(x))
	}
	return fmt.Sprintf("%g", x)
}

func esc(s string) string { return strings.ReplaceAll(s, "_", "\\_") }
