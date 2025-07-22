package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/rioloc/tfidf-go"
	"github.com/rioloc/tfidf-go/similarity"
	"github.com/rioloc/tfidf-go/token"
)

var (
	docNames = []string{"Hamlet", "Tom Sawyer", "Pride and Prejudice"}
	docPaths = []string{"./docs/hamlet.txt", "./docs/tom_sawyer.txt", "./docs/pride_and_prejudice.txt"}

	queries = []string{
		"To be or not to be",
		"Tom what's with that boy",
		"I hope he will like it",
	}
)

func main() {

	documents, err := loadData(docPaths)
	if err != nil {
		panic(err)
	}

	tokenOpts := []token.TokenizerOption{
		token.WithNormalizeFunc(func(s string) string {
			return strings.ToLower(s)
		}),
	}
	tokenizer := token.NewTokenizer(tokenOpts...)
	vectorizer := tfidf.NewTfIdfVectorizer()

	csm := similarity.NewCosineSimilarity(tokenizer, vectorizer)

	scores := make([][]float64, len(queries))
	for i, query := range queries {
		scores[i], err = csm.Do(query, documents)
		if err != nil {
			panic(err)
		}
	}

	prettyPrint(docNames, queries, scores)
}

func loadData(docs []string) ([]string, error) {
	contentSlice := make([]string, len(docs))
	for i, path := range docs {
		content, err := readFile(path)
		if err != nil {
			return nil, err
		}
		contentSlice[i] = content
	}
	return contentSlice, nil
}

func readFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return "", err
	}

	return string(content), nil
}

func prettyPrint(docNames, queries []string, scores [][]float64) {
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
	headers = append(headers, docNames...)
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
