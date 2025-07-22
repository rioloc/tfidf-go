package similarity

import (
	"math"

	"github.com/rioloc/tfidf-go"
)

// tokenizer is an interface that defines the Tokenize method.
// This allows for different tokenization strategies to be used.
type tokenizer interface {
	Tokenize(documents []string) ([]string, [][]string, error)
}

// vectorizer is an interface that defines the TfIdf method.
// This allows for different TF-IDF vectorization strategies to be used.
type vectorizer interface {
	TfIdf(tfVec [][]float64, idfVec []float64) (tfIdfMat [][]float64, err error)
}

// CosineSimilarity struct holds the tokenizer and vectorizer implementations.
// It is designed to calculate cosine similarity between an input string and a set of documents.
type CosineSimilarity struct {
	tokenizer  tokenizer
	vectorizer vectorizer
}

// NewCosineSimilarity is a constructor function that returns a new CosineSimilarity instance.
// It takes a tokenizer and a vectorizer as arguments, allowing for dependency injection.
func NewCosineSimilarity(tokenizer tokenizer, vectorizer vectorizer) *CosineSimilarity {
	return &CosineSimilarity{
		tokenizer:  tokenizer,
		vectorizer: vectorizer,
	}
}

// Do calculates the cosine similarity between an input string and a slice of documents.
// It returns a slice of float64, where each element is the cosine similarity score
// between the input string and the corresponding document.
func (c *CosineSimilarity) Do(input string, documents []string) ([]float64, error) {
	// Tokenize the provided documents to create a vocabulary and tokenized representations.
	vocabulary, tokens, err := c.tokenizer.Tokenize(documents)
	if err != nil {
		return nil, err
	}
	// Calculate Term Frequency (TF) for the documents.
	tfVec := tfidf.Tf(vocabulary, tokens)
	// Calculate Inverse Document Frequency (IDF) for the vocabulary.
	idfVec := tfidf.Idf(vocabulary, tokens, true)

	// Calculate TF-IDF vectors for the documents.
	tfIdfVec, err := c.vectorizer.TfIdf(tfVec, idfVec)
	if err != nil {
		return nil, err
	}

	// Tokenize the input string to generate its tokens.
	_, queryTokens, err := c.tokenizer.Tokenize([]string{input})
	if err != nil {
		return nil, err
	}
	// Calculate Term Frequency (TF) for the input string using the same vocabulary.
	tf := tfidf.Tf(vocabulary, queryTokens)
	// Calculate TF-IDF vector for the input string.
	tfIdf, err := c.vectorizer.TfIdf(tf, idfVec)
	if err != nil {
		return nil, err
	}

	// Initialize a slice to store the cosine similarity scores.
	scores := make([]float64, len(documents))
	// Iterate through each document's TF-IDF vector and calculate its cosine similarity with the input string's TF-IDF vector.
	for i, vec := range tfIdfVec {
		scores[i] = cosineSimilarity(tfIdf[0], vec)
	}
	return scores, nil
}

// cosineSimilarity calculates the cosine similarity between two given vectors (vec1 and vec2).
// It returns a float64 representing the similarity score.
func cosineSimilarity(vec1, vec2 []float64) float64 {
	var dot, normA, normB float64
	// Calculate the dot product of the two vectors and their magnitudes.
	for i := range vec1 {
		dot += vec1[i] * vec2[i]
		normA += vec1[i] * vec1[i] // Sum of squares for vec1
		normB += vec2[i] * vec2[i] // Sum of squares for vec2
	}
	// If either vector has a zero magnitude, return 0.0 to avoid division by zero.
	if normA == 0 || normB == 0 {
		return 0.0
	}
	// Calculate and return the cosine similarity.
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
