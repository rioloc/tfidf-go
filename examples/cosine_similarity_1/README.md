# Cosine Similarity Example

This example loads a set of documents from the `docs` directory and calculates cosine similarity for a set of queries, like:

```go
queries = []string{
    "To be or not to be",
    "Tom what's with that boy",
    "I hope he will like it",
}
```

## Output

```bash
+--------------------------+--------+------------+---------------------+
| INPUT                    | Hamlet | Tom Sawyer | Pride and Prejudice |
+--------------------------+--------+------------+---------------------+
| To be or not to be       | 0.2086 | 0.0448     | 0.1457              |
| Tom what's with that boy | 0.0546 | 0.3103     | 0.0412              |
| I hope he will like it   | 0.0295 | 0.0221     | 0.3607              |
+--------------------------+--------+------------+---------------------+
```