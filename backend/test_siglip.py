
from PIL import Image
from services.embedding_service import EmbeddingService

service = EmbeddingService()

image = Image.open("cycle.jpg").convert("RGB")
vector = service.embed_image(image)

print("Embedding dimension:", len(vector))
print("Expected dimension:", service.embedding_dim)
print("First 5 values:", vector[:5])
print("Vector length:", sum(x * x for x in vector) ** 0.5)

assert len(vector) == service.embedding_dim
assert abs(sum(x * x for x in vector) ** 0.5 - 1.0) < 1e-4

print("SigLIP embedding test passed.")
