package main

import (
	"fmt"
	"strings"

	tfidf "github.com/rioloc/tfidf-go"
	"github.com/rioloc/tfidf-go/token"
)

func main() {
	documents := []string{
		"this is a sample document",
		"this document is another example",
		"and this is a different one",
		"WHILE this Is Not NORMALized ProperLY",
		"and in this example the word example is written at least to times",
	}

	tokenOpts := []token.TokenizerOption{
		token.WithNormalizeFunc(func(s string) string {
			return strings.ToLower(s)
		}),
	}

	tokenizer := token.NewTokenizer(tokenOpts...)
	vocabulary, tokens, err := tokenizer.Tokenize(documents)
	if err != nil {
		panic(err)
	}

	tfVec := tfidf.Tf(vocabulary, tokens)
	idfVec := tfidf.Idf(vocabulary, tokens, true)

	vectorizer := tfidf.NewTfIdfVectorizer()
	tfidfMat, err := vectorizer.TfIdf(tfVec, idfVec)
	if err != nil {
		panic(err)
	}

	fmt.Println(vocabulary)
	fmt.Println(tfidfMat)
}
