package main

import (
	"fmt"
	"strings"

	"github.com/rioloc/tfidf-go"
	"github.com/rioloc/tfidf-go/similarity"
	"github.com/rioloc/tfidf-go/token"
)

var (
	documents = []string{
		"All animals are equal but some animals are more equal than others",
		"Big Brother is watching you",
		"If you want a picture of the future imagine a boot stamping on a human face forever",
		"To be or not to be that is the question",
		"All the world’s a stage and all the men and women merely players",
	}

	queries = []string{
		"equality among animals",
		"constant surveillance",
		"future oppression",
		"the meaning of life",
		"life is a stage",
	}
)

func main() {
	tokenOpts := []token.TokenizerOption{
		token.WithNormalizeFunc(func(s string) string {
			return strings.ToLower(s)
		}),
	}
	tokenizer := token.NewTokenizer(tokenOpts...)
	vectorizer := tfidf.NewTfIdfVectorizer()

	csm := similarity.NewCosineSimilarity(tokenizer, vectorizer)

	var err error
	scores := make([][]float64, len(queries))
	for i, query := range queries {
		scores[i], err = csm.Do(query, documents)
		if err != nil {
			panic(err)
		}
	}

	prettyPrint(documents, queries, scores)
}

func prettyPrint(documents, queries []string, scores [][]float64) {
	rows := make([][]string, len(scores))
	for i := range scores {
		rows[i] = make([]string, len(scores[0])+1)
	}

	for i := range scores {
		rows[i][0] = queries[i]
		for j := 1; j < len(rows[0]); j++ {
			rows[i][j] = fmt.Sprintf("%.4f", scores[i][j-1])
		}
	}

	headers := []string{"INPUT"}
	headers = append(headers, documents...)
	printTable(headers, rows)
}

func printTable(headers []string, rows [][]string) {
	// Determine column widths
	colWidths := make([]int, len(headers))
	for i, h := range headers {
		colWidths[i] = len(h)
	}
	for _, row := range rows {
		for i, cell := range row {
			if len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	// Build separator line
	sepLine := "+"
	for _, w := range colWidths {
		sepLine += strings.Repeat("-", w+2) + "+"
	}

	// Print header
	fmt.Println(sepLine)
	fmt.Print("|")
	for i, h := range headers {
		fmt.Printf(" %-*s |", colWidths[i], h)
	}
	fmt.Println()
	fmt.Println(sepLine)

	// Print rows
	for _, row := range rows {
		fmt.Print("|")
		for i, cell := range row {
			fmt.Printf(" %-*s |", colWidths[i], cell)
		}
		fmt.Println()
	}
	fmt.Println(sepLine)
}
