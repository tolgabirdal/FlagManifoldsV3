import torch
from torch import nn
from NetworkHeads import FlagNetHead

class PrototypicalFlagNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalFlagNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support1, z_support2 = self.backbone.forward(support_images)
        z_query1, z_query2 = self.backbone.forward(query_images)
    
        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Infer the number shots from the labels of the support set
        n_shot = len(torch.where(support_labels==support_labels[0])[0])

        scores = FlagNetHead(z_query1, z_query2, z_support1, z_support2, support_labels, n_way, n_shot)

        return scores