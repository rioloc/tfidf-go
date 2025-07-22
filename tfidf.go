// Package tfidf provides TF-IDF (Term Frequency-Inverse Document Frequency)
// computation for text analysis and document similarity calculations.
//
// This package implements the standard TF-IDF algorithm with support for different
// normalization schemes (L1, L2, or no normalization) and optional smoothing for
// handling rare terms.
//
// Example usage:
//
//	import "github.com/rioloc/tfidf-go"
//	import "github.com/rioloc/tfidf-go/token"
//
//	// Tokenize documents
//	tokenizer := token.NewTokenizer(token.WithNormalize())
//	vocabulary, tokens, _ := tokenizer.Tokenize(documents)
//
//	// Calculate TF-IDF
//	tfMatrix := tfidf.Tf(vocabulary, tokens)
//	idfVector := tfidf.Idf(vocabulary, tokens, true) // with smoothing
//
//	vectorizer := tfidf.NewTfIdfVectorizer()
//	tfidfMatrix, _ := vectorizer.TfIdf(tfMatrix, idfVector)
package tfidf

import (
	"errors"
	"math"
)

// NLevel represents the normalization level to apply to TF-IDF vectors.
// Different normalization schemes are useful for different similarity metrics.
type NLevel int

const (
	// NoNorm does not apply any normalization and returns raw TF-IDF scores.
	// Use this when you want to preserve the original scale of TF-IDF values.
	NoNorm NLevel = iota

	// L1Norm applies L1 normalization based on the sum of absolute values.
	// This makes the sum of all vector components equal to 1.
	// Useful for probability-like interpretations.
	L1Norm

	// L2Norm applies L2 normalization based on Euclidean norm.
	// This makes the Euclidean length of the vector equal to 1.
	// Best choice for cosine similarity calculations (default).
	L2Norm
)

// TfIdfVectorizer builds TF-IDF matrices from term frequency and inverse document frequency data.
// It supports different normalization schemes to make documents comparable regardless of length.
type TfIdfVectorizer struct {
	// NormLevel sets the normalization level to apply to each document vector.
	// Defaults to L2Norm which is optimal for cosine similarity calculations.
	NormLevel NLevel
}

// TfIdfOption is a functional option for configuring TfIdfVectorizer.
type TfIdfOption func(*TfIdfVectorizer)

// NewTfIdfVectorizer creates a new TF-IDF vectorizer with the specified options.
// By default, it uses L2 normalization which is best for cosine similarity.
//
// Example:
//
//	vectorizer := NewTfIdfVectorizer() // Uses L2 normalization
//	vectorizer := NewTfIdfVectorizer(WithNormLevel(L1Norm)) // Uses L1 normalization
func NewTfIdfVectorizer(opts ...TfIdfOption) *TfIdfVectorizer {
	t := &TfIdfVectorizer{
		// Default to L2 normalization (best for cosine similarity)
		NormLevel: L2Norm,
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// WithNormLevel sets the normalization level for the TF-IDF vectorizer.
//
// Parameters:
//   - lvl: The normalization level (NoNorm, L1Norm, or L2Norm)
//
// Example:
//
//	vectorizer := NewTfIdfVectorizer(WithNormLevel(L1Norm))
func WithNormLevel(lvl NLevel) TfIdfOption {
	return func(t *TfIdfVectorizer) {
		t.NormLevel = lvl
	}
}

// TfIdf computes the TF-IDF matrix by multiplying term frequency and inverse document frequency vectors.
// The result is optionally normalized according to the vectorizer's NormLevel setting.
//
// Parameters:
//   - tfVec: Term frequency matrix [documents][terms] from Tf()
//   - idfVec: Inverse document frequency vector [terms] from Idf()
//
// Returns:
//   - tfIdfMat: TF-IDF matrix [documents][terms] with optional normalization applied
//   - err: Error if normalization fails or input dimensions don't match
//
// The TF-IDF score for term j in document i is calculated as: tfVec[i][j] * idfVec[j]
// After calculation, each document vector is normalized according to NormLevel.
func (t *TfIdfVectorizer) TfIdf(tfVec [][]float64, idfVec []float64) (tfIdfMat [][]float64, err error) {
	if len(tfVec) == 0 {
		return nil, errors.New("empty TF matrix")
	}
	if len(tfVec[0]) != len(idfVec) {
		return nil, errors.New("TF matrix and IDF vector dimensions don't match")
	}

	tfIdfMat = make([][]float64, len(tfVec))
	for i := range tfIdfMat {
		tfIdfMat[i] = make([]float64, len(tfVec[0]))
	}

	// Calculate TF-IDF: tf[i][j] * idf[j] for each document i and term j
	for i := range tfIdfMat {
		for j := range tfIdfMat[i] {
			tfIdfMat[i][j] = tfVec[i][j] * idfVec[j]
		}
		// Apply normalization to make documents comparable regardless of length
		tfIdfMat[i], err = t.doNormalize(tfIdfMat[i])
		if err != nil {
			return nil, err
		}
	}

	return tfIdfMat, nil
}

// doNormalize applies the specified normalization to a document vector.
// This makes documents comparable regardless of their length or absolute TF-IDF scale.
func (t *TfIdfVectorizer) doNormalize(vec []float64) ([]float64, error) {
	switch t.NormLevel {
	case L1Norm:
		return l1Normalize(vec), nil
	case L2Norm:
		return l2Normalize(vec), nil
	case NoNorm:
		return vec, nil
	default:
		return nil, errors.New("invalid normalization level")
	}
}

// l1Normalize scales the vector so that the sum of the absolute values of its components equals 1.
// This creates a probability-like distribution where all values sum to 1.
//
// Formula: v_normalized[i] = v[i] / (|v[0]| + |v[1]| + ... + |v[n]|)
//
// Returns the original vector unchanged if the L1 norm is zero (all elements are zero).
func l1Normalize(vec []float64) []float64 {
	var norm float64
	for _, val := range vec {
		norm += math.Abs(val)
	}
	if norm == 0 {
		return vec // Return unchanged if vector is all zeros
	}
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

// l2Normalize scales the TF-IDF vector so its Euclidean length equals 1.
// This normalization ensures that document similarity comparisons are fair and consistent,
// focusing on term usage patterns rather than document length or absolute TF-IDF scale.
//
// Formula:
//   - Euclidean norm: ||v|| = sqrt(v[0]² + v[1]² + ... + v[n]²)
//   - Normalized: v_normalized[i] = v[i] / ||v||
//
// This is the preferred normalization for cosine similarity calculations.
// Returns the original vector unchanged if the L2 norm is zero (all elements are zero).
func l2Normalize(vec []float64) []float64 {
	var norm float64
	for _, val := range vec {
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return vec // Return unchanged if vector is all zeros
	}
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

// Tf calculates the Term Frequency matrix for a collection of tokenized documents.
// Term frequency represents how often each term appears in each document.
//
// Parameters:
//   - vocabulary: Ordered list of unique terms across all documents
//   - tokens: Tokenized documents where tokens[i] contains all tokens for document i
//
// Returns:
//   - Term frequency matrix [documents][terms] where element [i][j] represents
//     the count of vocabulary[j] in document i
//
// The raw term counts are returned without any normalization. Common TF normalizations
// like tf[i][j] / |document_i| can be applied separately if needed.
//
// Example:
//
//	vocabulary := []string{"the", "cat", "sat"}
//	tokens := [][]string{{"the", "cat", "sat"}, {"the", "cat", "sat", "the"}}
//	tfMatrix := Tf(vocabulary, tokens)
//	// tfMatrix[0] = [1, 1, 1] (document 0: "the"=1, "cat"=1, "sat"=1)
//	// tfMatrix[1] = [2, 1, 1] (document 1: "the"=2, "cat"=1, "sat"=1)
func Tf(vocabulary []string, tokens [][]string) [][]float64 {
	// Initialize matrix: [num_documents][num_terms]
	termsCountMatrix := make([][]float64, len(tokens))
	for i := range termsCountMatrix {
		termsCountMatrix[i] = make([]float64, len(vocabulary))
	}

	// Count term frequencies for each document
	for i, tokensRow := range tokens {
		// Build frequency map for this document
		termsMap := make(map[string]int)
		for _, term := range tokensRow {
			termsMap[term]++
		}

		// Fill matrix row with term counts
		for j, token := range vocabulary {
			if val, found := termsMap[token]; found {
				termsCountMatrix[i][j] = float64(val)
			}
			// Note: missing terms remain 0 (default value)
		}
	}

	return termsCountMatrix
}

// Idf calculates the Inverse Document Frequency vector for a vocabulary across a document corpus.
// IDF measures how rare or common each term is across the entire collection.
// Terms that appear in fewer documents get higher IDF scores.
//
// Parameters:
//   - vocabulary: Ordered list of unique terms to calculate IDF for
//   - tokens: Tokenized documents used to calculate document frequencies
//   - smoothing: If true, applies add-one smoothing to prevent log(0) and reduce impact of very rare terms
//
// Returns:
//   - IDF vector [terms] where element [j] is the IDF score for vocabulary[j]
//
// Formulas:
//   - With smoothing: IDF(t) = log((N + 1) / (df(t) + 1)) + 1
//   - Without smoothing: IDF(t) = log(N / df(t)) + 1
//   - Where N = total documents, df(t) = documents containing term t
//
// The +1 constant is added to ensure all IDF values are positive.
// Smoothing is recommended to handle rare terms and prevent extreme IDF values.
//
// Time Complexity: O(total_tokens + vocabulary_size × num_documents)
// Space Complexity: O(total_unique_tokens) for document maps
//
// Example:
//
//	vocabulary := []string{"the", "cat", "rare"}
//	tokens := [][]string{{"the", "cat"}, {"the", "dog"}, {"cat", "rare"}}
//	idfVec := Idf(vocabulary, tokens, true)
//	// "the" appears in 2/3 documents (common) -> lower IDF
//	// "rare" appears in 1/3 documents (rare) -> higher IDF
func Idf(vocabulary []string, tokens [][]string, smoothing bool) []float64 {
	idfVec := make([]float64, len(vocabulary))
	total := len(tokens)

	if total == 0 {
		// Handle edge case: no documents
		for i := range idfVec {
			idfVec[i] = 1.0 // Default IDF value
		}
		return idfVec
	}

	// Pre-compute document maps for O(1) term lookup instead of O(doc_length)
	// This optimization converts O(vocabulary × documents × avg_doc_length) 
	// to O(total_tokens + vocabulary × documents)
	docMaps := make([]map[string]struct{}, len(tokens))
	for i, doc := range tokens {
		docMaps[i] = make(map[string]struct{})
		for _, token := range doc {
			docMaps[i][token] = struct{}{}
		}
	}

	// Calculate IDF for each term in vocabulary
	for j, term := range vocabulary {
		docCount := 0
		// Count documents containing this term
		for _, docMap := range docMaps {
			if _, found := docMap[term]; found {
				docCount++
			}
		}

		if smoothing {
			// Add-one smoothing: prevents log(0) and reduces impact of very rare terms
			idfVec[j] = math.Log(float64(total+1)/float64(docCount+1)) + 1
			continue
		}
		if docCount == 0 {
			// Handle terms not found in any document (shouldn't happen with proper vocabulary)
			// Assign high IDF score as these terms are extremely rare
			idfVec[j] = math.Log(float64(total)) + 1
			continue
		}
		// Standard IDF formula
		idfVec[j] = math.Log(float64(total)/float64(docCount)) + 1
	}

	return idfVec
}
