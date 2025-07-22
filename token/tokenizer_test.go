package token

import "testing"

func TestTokenizer_Tokenize(t *testing.T) {
	tests := []struct {
		name         string
		docs         []string
		normalize    bool
		expectedSize int
	}{
		{
			name:         "Simple tokenization",
			docs:         []string{"This is good"},
			normalize:    false,
			expectedSize: 3,
		},
		{
			name:         "Lowercase normalization",
			docs:         []string{"This THIS this"},
			normalize:    true,
			expectedSize: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := []TokenizerOption{}
			if tt.normalize {
				opts = append(opts, WithNormalizeFunc(func(s string) string { return "this" }))
			}
			tokenizer := NewTokenizer(opts...)
			vocab, _, err := tokenizer.Tokenize(tt.docs)
			if err != nil {
				t.Fatalf("Tokenize error: %v", err)
			}
			if len(vocab) != tt.expectedSize {
				t.Errorf("got %d tokens, want %d", len(vocab), tt.expectedSize)
			}
		})
	}
}
