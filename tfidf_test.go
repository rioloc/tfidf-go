package tfidf

import (
	"math"
	"testing"
)

const tol = 1e-6

// Helper: compare slices with tolerance
func almostEqualSlices(a, b []float64, eps float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}

func TestTf(t *testing.T) {
	tests := []struct {
		name     string
		vocab    []string
		tokens   [][]string
		expected [][]float64
	}{
		{
			name:  "Simple TF",
			vocab: []string{"the", "cat", "sat"},
			tokens: [][]string{
				{"the", "cat", "sat"},
				{"the", "cat", "sat", "the"},
			},
			expected: [][]float64{
				{1, 1, 1},
				{2, 1, 1},
			},
		},
		{
			name:  "Missing term",
			vocab: []string{"dog"},
			tokens: [][]string{
				{"the", "cat"},
			},
			expected: [][]float64{
				{0},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Tf(tt.vocab, tt.tokens)
			for i := range tt.expected {
				if !almostEqualSlices(tt.expected[i], got[i], tol) {
					t.Errorf("row %d: got %v, want %v", i, got[i], tt.expected[i])
				}
			}
		})
	}
}

func TestIdf(t *testing.T) {
	tests := []struct {
		name      string
		vocab     []string
		tokens    [][]string
		smoothing bool
		wantFunc  func([]float64) bool
	}{
		{
			name:  "Smoothing on",
			vocab: []string{"the"},
			tokens: [][]string{
				{"the"}, {"the"}, {"cat"},
			},
			smoothing: true,
			wantFunc: func(idf []float64) bool {
				want := math.Log(4.0/3.0) + 1
				return math.Abs(idf[0]-want) < tol
			},
		},
		{
			name:  "Smoothing off",
			vocab: []string{"rare"},
			tokens: [][]string{
				{"the"}, {"the"}, {"rare"},
			},
			smoothing: false,
			wantFunc: func(idf []float64) bool {
				want := math.Log(3.0/1.0) + 1
				return math.Abs(idf[0]-want) < tol
			},
		},
		{
			name:      "Empty corpus",
			vocab:     []string{"term"},
			tokens:    [][]string{},
			smoothing: true,
			wantFunc: func(idf []float64) bool {
				return idf[0] == 1.0
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idf := Idf(tt.vocab, tt.tokens, tt.smoothing)
			if !tt.wantFunc(idf) {
				t.Errorf("Idf failed: got %v", idf)
			}
		})
	}
}

func TestTfIdfVectorizer_TfIdf(t *testing.T) {
	tests := []struct {
		name      string
		tf        [][]float64
		idf       []float64
		normLevel NLevel
		want      [][]float64
	}{
		{
			name: "NoNorm",
			tf: [][]float64{
				{1, 2},
				{3, 4},
			},
			idf:       []float64{1, 0.5},
			normLevel: NoNorm,
			want: [][]float64{
				{1, 1},
				{3, 2},
			},
		},
		{
			name: "L2Norm",
			tf: [][]float64{
				{1, 1},
			},
			idf:       []float64{1, 1},
			normLevel: L2Norm,
			want: [][]float64{
				{1 / math.Sqrt(2), 1 / math.Sqrt(2)},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vec := NewTfIdfVectorizer(WithNormLevel(tt.normLevel))
			got, err := vec.TfIdf(tt.tf, tt.idf)
			if err != nil {
				t.Fatalf("TfIdf error: %v", err)
			}
			for i := range tt.want {
				if !almostEqualSlices(tt.want[i], got[i], tol) {
					t.Errorf("row %d: got %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}
