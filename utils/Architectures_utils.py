
import torch
import torch.nn as nn
from vit_pytorch import ViT
from einops.layers.torch import Rearrange

class EquivariantLayer(nn.Module):
    """
    A permutation-equivariant layer for DeepSets.
    """
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.phi_self = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        self.phi_mean = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

    def forward(self, x):
        """
        x: [B, N, d] - Input batch
        Returns: [B, N, d] - Equivariant output
        """
        self_term = self.phi_self(x)
        sum_term = self.phi_mean(x.mean(dim=-2, keepdim=True))
    
        return self_term + sum_term

class InvariantLayer(nn.Module):
    """
    A permutation-invariant layer that maps [B, N, d] -> [B, d_out]
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out)
        )

    def forward(self, x):
        """
        x: [B, N, d] - Input batch
        Returns: [B, d_out] - Aggregated representation
        """
        x = x.mean(dim=-2)  # Mean-pooling aggregation
        return self.rho(x)

class DeepSets(nn.Module):
    """
    Full DeepSets model with K equivariant layers and one invariant layer.
    """
    def __init__(self, d_in, d_hidden, d_out, K=2):
        super().__init__()
        self.equivariant_layers = nn.ModuleList([
            EquivariantLayer(d_in if i == 0 else d_hidden, d_hidden)
            for i in range(K)
        ])
        self.invariant_layer = InvariantLayer(d_hidden, d_out)

    def forward(self, x):
        """
        x: [B, N, d_in] - Input batch
        Returns: [B, d_out] - Set-level representation
        """
        for layer in self.equivariant_layers:
            x = layer(x)
        return self.invariant_layer(x)



class SdInvVit(ViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., DS_model=DeepSets):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        
        # Override patch embedding method
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) c (p1 p2)', p1 = patch_height, p2 = patch_width), # [1 x num_patches x p1*p2 x c]
            DS_model(d_in=patch_height*patch_width, d_hidden=dim, d_out=dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

