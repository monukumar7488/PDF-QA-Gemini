from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

sentences = [
    "Robot sensors detect objects.",
    "Sensors help robots understand their environment.",
    "The weather is hot today."
]

embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])