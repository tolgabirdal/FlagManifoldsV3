import torch
from torch import nn
from NetworkHeads import ProtoNetHead, SubspaceNetHead


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module, head: str):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.head = head

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
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
    
        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Infer the number shots from the labels of the support set
        n_shot = len(torch.where(support_labels==support_labels[0])[0])

        if self.head == 'ProtoNet':
            scores = ProtoNetHead(z_query, z_support, support_labels, n_way, n_shot)
        elif self.head == 'SubspaceNet':
            scores = SubspaceNetHead(z_query, z_support, support_labels, n_way, n_shot)
        else:
            print('head not recognized')

        return scores