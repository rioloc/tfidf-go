# tfidf-go

A simple, idiomatic Go library for computing **TF-IDF** vectors, with **L1/L2 normalization**, custom tokenization, and cosine similarity.

- Pure Go — no dependencies beyond the standard library.
- Easy tokenization with Unicode-ready patterns.
- TF, IDF, and TF-IDF matrix calculations.
- Cosine Similarity implementation, including support for user defined Tokenizer and TF-IDF Vectorizer.
- Inspired on Python _scikit-learn_ class `sklearn.feature_extraction.text.TfidfVectorizer`.
	- If enabled, uses same smoothing for IDF
	- Defaults to L2 Normalization (Euclidean Norm) for TF-IDF

## TF-IDF Usage
```go
import "github.com/rioloc/tfidf-go"
import "github.com/rioloc/tfidf-go/token"

...

documents := []string{
    "this is a sample document",
    "this document is another example",
    "and this is a different one",
    "WHILE this Is Not NORMALized ProperLY",
    "and in this example the word example is written at least to times",
}

// Tokenize documents
tokenOpts := []token.TokenizerOption{
	token.WithNormalizeFunc(func(s string) string {
		return strings.ToLower(s)
	}),
}
vocabulary, tokens, _ := tokenizer.Tokenize(documents)

// Calculate TF-IDF
tfMatrix := tfidf.Tf(vocabulary, tokens)
idfVector := tfidf.Idf(vocabulary, tokens, true) // with smoothing

vectorizer := tfidf.NewTfIdfVectorizer()
tfidfMatrix, _ := vectorizer.TfIdf(tfMatrix, idfVector)
```

Produces a result like

```text
[and another at different document example in is least normalized not one properly sample the this times to while word written]
[
  [0 0 0 0 0.556074875921159 0 0 0.3284267796124245 0 0 0 0 0 0.6892404756223269 0 0.3284267796124245 0 0 0 0 0]
  [0 0.6023717267407068 0 0 0.4859897162935888 0.4859897162935888 0 0.2870333553240574 0 0 0 0 0 0 0 0.2870333553240574 0 0 0 0 0] 
  [0.4578566690891173 0 0 0.5675015398728065 0 0 0 0.2704175244456293 0 0 0 0.5675015398728065 0 0 0 0.2704175244456293 0 0 0 0 0] 
  [0 0 0 0 0 0 0 0.22578084262477424 0 0.47382645087819175 0.47382645087819175 0 0.47382645087819175 0 0 0.22578084262477424 0 0 0.47382645087819175 0 0]
]
```

If we calculate the TF-IDF matrix for the same documents with `sklearn`, we get comparable results
```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "this is a sample document",
    "this document is another example",
    "and this is a different one",
    "WHILE this Is Not NORMALized ProperLY",
    "and in this example the word example is written at least to times",
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```


```text
['and' 'another' 'at' 'different' 'document' 'example' 'in' 'is' 'least'
 'normalized' 'not' 'one' 'properly' 'sample' 'the' 'this' 'times' 'to'
 'while' 'word' 'written']
[[0.         0.         0.         0.         0.55607488 0.
  0.         0.32842678 0.         0.         0.         0.
  0.         0.68924048 0.         0.32842678 0.         0.
  0.         0.         0.        ]
 [0.         0.60237173 0.         0.         0.48598972 0.48598972
  0.         0.28703336 0.         0.         0.         0.
  0.         0.         0.         0.28703336 0.         0.
  0.         0.         0.        ]
 [0.45785667 0.         0.         0.56750154 0.         0.
  0.         0.27041752 0.         0.         0.         0.56750154
  0.         0.         0.         0.27041752 0.         0.
  0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.22578084 0.         0.47382645 0.47382645 0.
  0.47382645 0.         0.         0.22578084 0.         0.
  0.47382645 0.         0.        ]
 [0.2357807  0.         0.2922441  0.         0.         0.4715614
  0.2922441  0.13925588 0.2922441  0.         0.         0.
  0.         0.         0.2922441  0.13925588 0.2922441  0.2922441
  0.         0.2922441  0.2922441 ]]
```


## Cosine Similarity Usage
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
Will produce the following cosine similarity scores

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

```

- The previous example can be found in the example https://github.com/rioloc/tfidf-go/blob/main/examples/cosine_similarity_2
- Another example of cosine similarity scores calculation can be found in  https://github.com/rioloc/tfidf-go/blob/main/examples/cosine_similarity_1

## Performance Analysis  and Considerations
At the state of the art, by running benchmark tests within _similarity_ package via `go test -bench=.`, with the following parameters, it is possible to have an overview on the performances.

- Small: 10 documents, ~5 tokens each; ~5 tokens input; 
- Medium: 100 documents, ~20 tokens each; ~20 tokens input; 
- Large: 1000 documents, ~50 tokens each; ~50 tokens input; 

```text
goos: darwin
goarch: amd64
pkg: github.com/rioloc/tfidf-go/similarity
cpu: Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz
BenchmarkCosineSimilarity_Do_Small-8    	  49926	    27438 ns/op	  14854 B/op	     77 allocs/op
BenchmarkCosineSimilarity_Do_Medium-8   	   1587	   734638 ns/op	 377003 B/op	    975 allocs/op
BenchmarkCosineSimilarity_Do_Large-8    	     74	 15078496 ns/op	7413658 B/op	  12506 allocs/op
PASS
ok  	github.com/rioloc/tfidf-go/similarity	4.343s
```
