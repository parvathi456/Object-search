
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


class EmbeddingService:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k",
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.embedding_dimension = (
            self.model.visual.output_dim
        )

    @torch.inference_mode()
    def embed_image(self, image: Image.Image) -> list[float]:
        """
        Convert a PIL image into a normalized embedding.
        The same method must be used for indexed crops
        and reference query images.
        """
        image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0)
        tensor = tensor.to(self.device)

        features = self.model.encode_image(tensor)
        features = F.normalize(features, p=2, dim=1)

        return features[0].cpu().tolist()
