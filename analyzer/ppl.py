import math
from collections import Counter

def calculate_ngram_probabilities(corpus, n):
    ngrams = []
    for sentence in corpus:
        tokens = sentence.split()
        print(f"tokens: {tokens}")
        sentence_ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        print(f"sentence_ngrams: {sentence_ngrams}")
        ngrams.extend(sentence_ngrams)
        print(f"ngram: {ngrams}")
    ngram_counts = Counter(ngrams)
    context_counts = Counter([ngram[:-1] for ngram in ngrams])
    probabilities = {}
    for ngram in ngram_counts:
        context = ngram[:-1]
        print(f"ngram_counts[ngram]: {ngram_counts[ngram]}")
        print(f"context_counts[context]: {context_counts[context]}")
        probabilities[ngram] = ngram_counts[ngram] / context_counts[context]

    return probabilities


def calculate_perplexity(corpus, probabilities, n):
    log_prob_sum = 0
    token_count = 0

    for sentence in corpus:
        tokens = sentence.split()
        sentence_ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        for ngram in sentence_ngrams:
            if ngram in probabilities:
                log_prob_sum += math.log(probabilities[ngram])
            else:
                # If ngram is not in the training data, assign a very small probability
                log_prob_sum += math.log(1e-6)
            token_count += 1

    # Calculate perplexity
    perplexity = math.exp(-log_prob_sum / token_count)
    return perplexity


# Example usage
corpus = ["The cat chased the mouse under the table",
          "The mouse found a piece of cheese",
          "The dog barked at the cat loudly"]

n = 2  # Bigram model
probabilities = calculate_ngram_probabilities(corpus, n)
print(f"probabilities: {probabilities}")
perplexity = calculate_perplexity(corpus, probabilities, n)
print(f"Perplexity: {perplexity}")