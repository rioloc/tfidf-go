package token

import (
	"slices" // Importing the slices package for sorting.
	"unicode"
)

// Tokenizer is a simple tokenizer implementation based on regular expressions.
type Tokenizer struct {
	normalizeFunc func(string) string // An optional function to normalize tokens (e.g., convert to lowercase).
}

// TokenizerOption is a function type that allows for configuring the Tokenizer.
type TokenizerOption func(*Tokenizer)

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
	t := &Tokenizer{}

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
		tkns := t.tokenize(doc)
		if t.normalizeFunc != nil {
			for j, term := range tkns {
				tkns[j] = t.normalizeFunc(term)
			}
		}
		tokens[i] = tkns
	}
	return vocabulary(tokens), tokens, nil
}

// tokenize extracts tokens from a single document string based on the tokenizer's pattern.
// If a normalize function is set, it applies normalization to each extracted term.
func (t *Tokenizer) tokenize(doc string) []string {

	// estimate the number of tokens to avoid reallocations
	// this is a rough estimate, but it's good enough for most cases
	// we divide by 8 because we expect to find 8 tokens per word
	// the number 8 is an arbitrary, but reasonable, estimate for the average length of a word
	tokens := make([]string, 0, len(doc)/8)

	// zero allocation tokenization
	start := -1
	for i, r := range doc {
		if unicode.IsLetter(r) {
			if start == -1 {
				start = i
			}
			continue
		}
		if start != -1 && i-start > 1 {
			tokens = append(tokens, t.doNormalize(doc[start:i]))
		}
		start = -1
	}
	// handle trailing token
	if start != -1 && len(doc)-start > 1 {
		tokens = append(tokens, t.doNormalize(doc[start:]))
	}

	return tokens
}

// doNormalize applies the normalization function to a token if it is defined
func (t *Tokenizer) doNormalize(token string) string {
	if t.normalizeFunc != nil {
		return t.normalizeFunc(token)
	}
	return token
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
