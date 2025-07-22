# Cosine Similarity Example

```go
import "github.com/rioloc/tfidf-go"
import "github.com/rioloc/tfidf-go/similarity"
import "github.com/rioloc/tfidf-go/token"

...

documents := []string{
	"All animals are equal but some animals are more equal than others",
	"Big Brother is watching you",
	"If you want a picture of the future imagine a boot stamping on a human face forever",
	"To be or not to be that is the question",
	"All the world’s a stage and all the men and women merely players",
}

queries := []string{
	"equality among animals",
	"constant surveillance",
	"future oppression",
	"the meaning of life",
	"life is a stage",
}

// Instantiate Tokenizer
tokenOpts := []token.TokenizerOption{
	token.WithNormalizeFunc(func(s string) string {
		return strings.ToLower(s)
	}),
}
tokenizer := token.NewTokenizer(tokenOpts...)

// Instantiate TF-IDF Vectorizer
vectorizer := tfidf.NewTfIdfVectorizer()

// Instantiate CosineSimilarity
csm := similarity.NewCosineSimilarity(tokenizer, vectorizer)

scores := make([][]float64, len(queries))
for i, query := range queries {
	// Calculate cosine similarity scores against the documents
	scores[i], err = csm.Do(query, documents)
	...
}
```

## Output

```text
+-----------------------+------------------------------+----------------------------+--------------------------------------+--------------------------------+-----------------------------+
| INPUT                 | All animals are equal ...    | Big Brother is watch...    | If you want a picture of the fut...  | To be or not to be ...         | All the world’s a stage ... |
+-----------------------+------------------------------+----------------------------+--------------------------------------+--------------------------------+-----------------------------+
| equality among animals| 0.4760                       | 0.0000                     | 0.0000                               | 0.0000                         | 0.0000                      |
| constant surveillance | 0.0000                       | 0.0000                     | 0.0000                               | 0.0000                         | 0.0000                      |
| future oppression     | 0.0000                       | 0.0000                     | 0.2763                               | 0.0000                         | 0.0000                      |
| the meaning of life   | 0.0000                       | 0.0000                     | 0.3325                               | 0.1030                         | 0.1964                      |
| life is a stage       | 0.0000                       | 0.2443                     | 0.0000                               | 0.1400                         | 0.2051                      |
+-----------------------+------------------------------+----------------------------+--------------------------------------+--------------------------------+-----------------------------+
