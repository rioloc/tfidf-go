package similarity

import (
	"errors"
	"math"
	"strings"
	"testing"

	"math/rand"

	"github.com/rioloc/tfidf-go"
	"github.com/rioloc/tfidf-go/token"
)

const tol = 2e-1

func TestCosineSimilarity_Do(t *testing.T) {

	tests := []struct {
		name           string
		input          string
		documents      []string
		expectedScores []float64
		expectedErr    error
	}{
		{
			name:      "Successful calculation - 1",
			input:     "apple banana",
			documents: []string{"apple orange", "banana grape"},
			expectedScores: []float64{
				0.5,
				0.5,
			},
			expectedErr: nil,
		},
		{
			name:      "Successful calculation - 2",
			input:     "apple banana orange",
			documents: []string{"apple apple banana", "banana orange grape", "apple grape orange banana"},
			expectedScores: []float64{
				0.775,
				0.667,
				0.866,
			},
			expectedErr: nil,
		},
		{
			name:  "Successful calculation - 3",
			input: "data science machine learning",
			documents: []string{
				"data mining data analysis",
				"machine learning deep learning",
				"big data science and analytics",
				"data science machine",
			},
			expectedScores: []float64{
				0.408,
				0.612,
				0.5,
				0.86,
			},
			expectedErr: nil,
		},
		{
			name:           "Empty documents list",
			input:          "test",
			documents:      []string{},
			expectedScores: []float64{}, // No documents, so no scores
			expectedErr:    errors.New("empty TF matrix"),
		},
		{
			name:           "No common terms (resulting in zero vectors for query against vocab)",
			input:          "apple",
			documents:      []string{"orange", "grape"},
			expectedScores: []float64{0.0, 0.0}, // "apple" has no common terms with "orange" or "grape"
			expectedErr:    nil,
		},
		{
			name:           "Query with zero magnitude (empty after tokenization)",
			input:          "",
			documents:      []string{"apple"},
			expectedScores: []float64{0.0}, // Cosine similarity with a zero vector is 0
			expectedErr:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// tokenizer and vectorizer can be mocked as well
			// but for this scenario it is fine to stick with the library ones
			// which are quite simple and straightforward implementations
			// this also improves the testing coverage on them
			tokenOpts := []token.TokenizerOption{
				token.WithNormalizeFunc(func(s string) string {
					return strings.ToLower(s)
				}),
			}
			tokenizer := token.NewTokenizer(tokenOpts...)
			vectorizer := tfidf.NewTfIdfVectorizer()

			cs := NewCosineSimilarity(tokenizer, vectorizer)
			scores, err := cs.Do(tt.input, tt.documents)

			if tt.expectedErr != nil {
				if err == nil || err.Error() != tt.expectedErr.Error() {
					t.Errorf("Do() error = %v, wantErr %v", err, tt.expectedErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("Do() unexpected error: %v", err)
			}
			if len(scores) != len(tt.expectedScores) {
				t.Fatalf("Do() scores length = %v, want %v", len(scores), len(tt.expectedScores))
			}
			for i := range scores {
				// Use a small epsilon for float comparison
				delta := math.Abs(scores[i] - tt.expectedScores[i])
				if delta > tol {
					t.Errorf("Do() score[%d] = %v, want %v, delta: %v, tol: %.4f)", i, scores[i], tt.expectedScores[i], delta, tol)
				}
			}

		})
	}
}

func Test_cosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		vec1 []float64
		vec2 []float64
		want float64
	}{
		{
			name: "Perfect match",
			vec1: []float64{1, 1, 1},
			vec2: []float64{1, 1, 1},
			want: 1.0,
		},
		{
			name: "No commonality (orthogonal)",
			vec1: []float64{1, 0, 0},
			vec2: []float64{0, 1, 0},
			want: 0.0,
		},
		{
			name: "Partial commonality",
			vec1: []float64{1, 1, 0},
			vec2: []float64{1, 0, 1},
			want: 0.5, // (1*1 + 1*0 + 0*1) / (sqrt(2) * sqrt(2)) = 1 / 2 = 0.5
		},
		{
			name: "Negative correlation",
			vec1: []float64{1, -1},
			vec2: []float64{-1, 1},
			want: -1.0, // (1*-1 + -1*1) / (sqrt(2) * sqrt(2)) = -2 / 2 = -1.0
		},
		{
			name: "One zero vector",
			vec1: []float64{0, 0, 0},
			vec2: []float64{1, 1, 1},
			want: 0.0,
		},
		{
			name: "Another zero vector",
			vec1: []float64{1, 1, 1},
			vec2: []float64{0, 0, 0},
			want: 0.0,
		},
		{
			name: "Both zero vectors",
			vec1: []float64{0, 0, 0},
			vec2: []float64{0, 0, 0},
			want: 0.0,
		},
		{
			name: "Different magnitudes, same direction",
			vec1: []float64{1, 1},
			vec2: []float64{2, 2},
			want: 1.0, // (1*2 + 1*2) / (sqrt(2) * sqrt(8)) = 4 / (sqrt(16)) = 4 / 4 = 1.0
		},
		{
			name: "Complex vectors",
			vec1: []float64{1, 2, 3},
			vec2: []float64{4, 5, 6},
			want: 0.9746318461970762, // Calculated value
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.vec1, tt.vec2)
			if math.Abs(got-tt.want) > tol {
				t.Errorf("cosineSimilarity() got = %v, want %v, math.Abs(got-want): %v", got, tt.want, math.Abs(got-tt.want))
			}
		})
	}
}

// sampleWords is a pool of words to generate varied documents
var sampleWords = []string{
	"data", "science", "machine", "learning", "deep", "neural", "networks",
	"analysis", "mining", "model", "training", "testing", "algorithm",
	"statistics", "probability", "regression", "classification", "clustering",
	"feature", "vector", "natural", "language", "processing", "sentiment",
	"information", "retrieval", "text", "summarization",
	"cloud", "computing", "big", "analytics", "artificial", "intelligence",
	"transformer", "sequence", "attention", "embedding", "optimization",
	"tensorflow", "pytorch", "gpu", "cpu", "performance", "benchmark",
	"dataset", "evaluation", "validation",
}

// generateDoc creates a document string of approximately length tokens,
// randomly picking words from sampleWords with some repetition.
func generateDoc(length int) string {
	var b strings.Builder
	for i := 0; i < length; i++ {
		word := sampleWords[rand.Intn(len(sampleWords))]
		b.WriteString(word)
		if i != length-1 {
			b.WriteByte(' ')
		}
	}
	return b.String()
}

func BenchmarkCosineSimilarity_Do_Small(b *testing.B) {
	input := generateDoc(5)

	documents := make([]string, 10)
	for i := range documents {
		documents[i] = generateDoc(5)
	}

	tokenOpts := []token.TokenizerOption{
		token.WithNormalizeFunc(func(s string) string {
			return strings.ToLower(s)
		}),
	}
	tokenizer := token.NewTokenizer(tokenOpts...)
	vectorizer := tfidf.NewTfIdfVectorizer()
	cs := NewCosineSimilarity(tokenizer, vectorizer)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cs.Do(input, documents)
	}
}

func BenchmarkCosineSimilarity_Do_Medium(b *testing.B) {
	input := generateDoc(20)

	documents := make([]string, 100)
	for i := range documents {
		documents[i] = generateDoc(20)
	}

	tokenOpts := []token.TokenizerOption{
		token.WithNormalizeFunc(func(s string) string {
			return strings.ToLower(s)
		}),
	}
	tokenizer := token.NewTokenizer(tokenOpts...)
	vectorizer := tfidf.NewTfIdfVectorizer()
	cs := NewCosineSimilarity(tokenizer, vectorizer)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cs.Do(input, documents)
	}
}

func BenchmarkCosineSimilarity_Do_Large(b *testing.B) {
	input := generateDoc(50)

	documents := make([]string, 1000)
	for i := range documents {
		documents[i] = generateDoc(50)
	}

	tokenOpts := []token.TokenizerOption{
		token.WithNormalizeFunc(func(s string) string {
			return strings.ToLower(s)
		}),
	}
	tokenizer := token.NewTokenizer(tokenOpts...)
	vectorizer := tfidf.NewTfIdfVectorizer()
	cs := NewCosineSimilarity(tokenizer, vectorizer)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cs.Do(input, documents)
	}
}
