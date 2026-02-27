import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class BankingFaceModel:
    def __init__(self, device):
        self.device = device
        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(device)

    def embed(self, image_tensor):
        """
        image_tensor: (1, 3, 160, 160) normalized to [-1, 1]
        returns: (1, 512) L2-normalized embedding
        """
        with torch.no_grad():
            emb = self.model(image_tensor)
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    @staticmethod
    def compute_confidence(embedding, template):
        """
        cosine similarity mapped to [0,1]
        """
        similarity = F.cosine_similarity(embedding, template)
        confidence = (similarity + 1) / 2
        return confidence.item()
