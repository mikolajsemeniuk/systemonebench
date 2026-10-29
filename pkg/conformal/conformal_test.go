package conformal

import "testing"

func TestPickCertify(t *testing.T) {
	a := []Candidate{{0.01, 100, 20}, {0.05, 100, 4}, {0.1, 100, 6}, {0.2, 100, 30}}
	b := []Candidate{{0.01, 100, 18}, {0.05, 100, 5}, {0.1, 100, 12}, {0.2, 100, 28}}
	// largest eps with miss rate <= 0.1 on A is 0.1 (6/100); on B (12+1)/101 > 0.1 -> not certified
	eps, ok := Pick(a, 0.1)
	if !ok || eps != 0.1 || Certify(b, eps, 0.1) {
		t.Fatalf("got %v %v cert=%v", eps, ok, Certify(b, eps, 0.1))
	}
	// alpha 0.05: A picks 0.05 (4/100); B (5+1)/101 = 0.059 > 0.05 -> not certified; alpha 0.06 certifies
	eps, ok = Pick(a, 0.05)
	if !ok || eps != 0.05 || Certify(b, eps, 0.05) || !Certify(b, eps, 0.06) {
		t.Fatalf("got %v %v", eps, ok)
	}
	if _, ok := Pick(a, 0.01); ok {
		t.Fatal("expected no candidate")
	}
}
