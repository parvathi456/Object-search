
# import open_clip
# import torch
# import torch.nn.functional as F
# from PIL import Image


# class EmbeddingService:
#     def __init__(self):
#         self.device = (
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         self.model, _, self.preprocess = (
#             open_clip.create_model_and_transforms(
#                 "ViT-B-32",
#                 pretrained="laion2b_s34b_b79k",
#             )
#         )

#         self.model = self.model.to(self.device)
#         self.model.eval()

#         self.embedding_dimension = (
#             self.model.visual.output_dim
#         )

#     @torch.inference_mode()
#     def embed_image(self, image: Image.Image) -> list[float]:
#         """
#         Convert a PIL image into a normalized embedding.
#         The same method must be used for indexed crops
#         and reference query images.
#         """
#         image = image.convert("RGB")
#         tensor = self.preprocess(image).unsqueeze(0)
#         tensor = tensor.to(self.device)

#         features = self.model.encode_image(tensor)
#         features = F.normalize(features, p=2, dim=1)

#         return features[0].cpu().tolist()



import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class EmbeddingService:
    def __init__(self):
        self.model_name = "google/siglip-base-patch16-224"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading SigLIP on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        ).to(self.device)

        self.model.eval()

        self.embedding_dim = self.model.config.vision_config.hidden_size

        print(
            f"SigLIP loaded. Embedding dimension: "
            f"{self.embedding_dim}"
        )

    @torch.inference_mode()
    def embed_image(self, image):
        """
        Generate a normalized SigLIP image embedding.
        """

        inputs = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        # Extract pooled image features directly from the vision encoder.
        outputs = self.model.vision_model(
            pixel_values=pixel_values
        )

        features = outputs.pooler_output

        # Normalize the embedding for cosine similarity search.
        features = F.normalize(features, p=2, dim=-1)

        return features[0].cpu().tolist()
