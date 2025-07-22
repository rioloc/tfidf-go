package token

import (
	"regexp"
	"slices" // Importing the slices package for sorting.
)

const (
	// defaultTokenPattern is the default regular expression used for tokenization.
	// `(?i)` enables case-insensitive matching.
	// `\p{L}{2,}` matches words with two Unicode letters or more.
	defaultTokenPattern = `(?i)\p{L}{2,}`
)

// Tokenizer is a simple tokenizer implementation based on regular expressions.
type Tokenizer struct {
	tokenPattern  string              // The regular expression pattern used to find tokens.
	normalizeFunc func(string) string // An optional function to normalize tokens (e.g., convert to lowercase).
}

// TokenizerOption is a function type that allows for configuring the Tokenizer.
type TokenizerOption func(*Tokenizer)

// WithTokenPattern is a functional option to set a custom token pattern for the Tokenizer.
func WithTokenPattern(pattern string) TokenizerOption {
	return func(t *Tokenizer) {
		t.tokenPattern = pattern
	}
}

// WithNormalizeFunc is a functional option to set a normalization function for the Tokenizer.
// This function will be applied to each token after extraction.
func WithNormalizeFunc(fn func(string) string) TokenizerOption {
	return func(t *Tokenizer) {
		t.normalizeFunc = fn
	}
}

// NewTokenizer is a constructor function that creates and returns a new Tokenizer instance.
// It accepts a variable number of TokenizerOption functions to configure the tokenizer.
func NewTokenizer(opts ...TokenizerOption) *Tokenizer {
	t := &Tokenizer{
		tokenPattern: defaultTokenPattern, // Initialize with the default pattern.
	}

	// Apply all provided options to the tokenizer.
	for _, opt := range opts {
		opt(t)
	}

	return t
}

// Tokenize takes a slice of documents and returns a vocabulary (unique tokens)
// and a 2D slice representing the tokens for each document.
func (t *Tokenizer) Tokenize(documents []string) ([]string, [][]string, error) {
	tokens := make([][]string, len(documents))
	// Process each document individually.
	for i, doc := range documents {
		tkns, err := t.tokenize(doc) // Call the internal tokenize method for a single document.
		if err != nil {
			return nil, nil, err
		}
		tokens[i] = tkns
	}
	// Generate the vocabulary from all tokens and return it along with the tokenized documents.
	return vocabulary(tokens), tokens, nil

}

// tokenize extracts tokens from a single document string based on the tokenizer's pattern.
// If a normalize function is set, it applies normalization to each extracted term.
func (t *Tokenizer) tokenize(doc string) ([]string, error) {
	r, err := regexp.Compile(t.tokenPattern) // Compile the regular expression pattern.
	if err != nil {
		return nil, err
	}
	// Find all strings that match the token pattern in the document.
	terms := r.FindAllString(doc, -1)

	// If no normalization function is provided, return the terms as-is.
	if t.normalizeFunc == nil {
		return terms, nil
	}

	// If normalization is enabled, apply the normalizeFunc to every extracted term.
	normalizedTerms := make([]string, len(terms))
	for j, term := range terms {
		normalizedTerms[j] = t.normalizeFunc(term)
	}

	return normalizedTerms, nil
}

// vocabulary extracts all unique tokens from a 2D slice of tokens (documents)
// and returns them as a sorted slice of strings.
// Example:
// Input: [ ["this", "is", "good", "example"], ["alSo", "THIS", "can", "good", "example"] ]
// Returns: ["also", "can", "example", "good", "is", "this"] (sorted)
func vocabulary(tokens [][]string) []string {
	// Use a map to efficiently store unique terms.
	termsMap := make(map[string]struct{})
	var terms []string // Slice to store the unique terms in order of discovery.
	for _, doc := range tokens {
		for _, token := range doc {
			// If the token is already in the map, skip it.
			if _, f := termsMap[token]; f {
				continue
			}
			// Add the token to the slice and mark it as seen in the map.
			terms = append(terms, token)
			termsMap[token] = struct{}{}
		}
	}
	slices.Sort(terms) // Sort the unique terms alphabetically.
	return terms
}
