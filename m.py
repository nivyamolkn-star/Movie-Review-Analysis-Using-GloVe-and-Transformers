#=========================================================
# MOVIE REVIEW ANALYSIS USING GLOVE + TRANSFORMERS
# =========================================================

# =========================================================
# PART 1 — GLOVE
# =========================================================

# Download GloVe
!wget http://nlp.stanford.edu/data/glove.6B.zip

# Extract files
!unzip -q glove.6B.zip


# Import NumPy
import numpy as np

# GloVe file
glove_path = "glove.6B.100d.txt"

# Empty dictionary
glove = {}


# Load GloVe vectors
with open(glove_path, "r", encoding="utf8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        glove[word] = vector

print("Number of words loaded:", len(glove))


# =========================================================
# GET VECTOR
# =========================================================
word = "movie"
print("\nVector for:", word)
print(glove[word])

# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(a, b):
    return np.dot(a, b) / ( np.linalg.norm(a) *np.linalg.norm(b) )

# Compare movie and film
similarity = cosine_similarity(glove["movie"], glove["film"])
print("\nSimilarity between movie and film:",similarity)


# Compare actor and actress
similarity = cosine_similarity(glove["actor"], glove["actress"])
print("\nSimilarity between actor and actress:",similarity)

# =========================================================
# CREATE REUSABLE FUNCTION
# =========================================================

def compare_words(word1, word2):
    similarity = cosine_similarity(glove[word1], glove[word2])
    print("\nSimilarity between", word1, "and", word2, ":",similarity )

compare_words("movie", "film")
compare_words("story", "book")


# =========================================================
# PART 2 — TRANSFORMERS
# =========================================================

!pip install transformers torch --quiet
from transformers import pipeline
# Create sentiment-analysis pipeline
classifier = pipeline( "sentiment-analysis")

# =========================================================
# TEST SENTENCE
# =========================================================

sentence = (
    "The movie was surprisingly good "
    "and I enjoyed it."
)
result = classifier(sentence)

print("\nSentiment result:")
print(result)


# =========================================================
# TEST MULTIPLE REVIEWS
# =========================================================

reviews = [
    "The movie was fantastic.",
    "The movie was boring.",
    "The acting was excellent.",
    "The story was terrible.",
    "I really enjoyed this film.",
    "I hated this movie."
]

for review in reviews:
    result = classifier(review)
    print("\nReview:", review)
    print("Prediction:", result)


# =========================================================
# USER INPUT
# =========================================================

user_sentence = input("\nEnter your own movie review: ")
result = classifier(user_sentence)

print("\nTransformer Result:")
print(result)
