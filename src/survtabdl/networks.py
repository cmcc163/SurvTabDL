"""Scalar log-hazard heads for the manuscript's tabular architectures."""
import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, n_features, cat_idx, cat_dims, params, row_attention):
        super().__init__()
        from ._vendor.saint.pretrainmodel import SAINT
        self.cat_idx = list(cat_idx)
        self.num_idx = [i for i in range(n_features) if i not in cat_idx]
        self.backbone = SAINT(
            categories=(1, *cat_dims), num_continuous=len(self.num_idx),
            dim=params.get("dim", 16), depth=params.get("depth", 2),
            heads=params.get("heads", 2), attn_dropout=0.1,
            ff_dropout=params.get("dropout", 0.1), attentiontype="colrow" if row_attention else "col",
            cont_embeddings="MLP", final_mlp_style="sep", y_dim=1,
            mlp_hiddle=params.get("embedding_hidden", 100), final_hiddle=params.get("head_hidden", 128),
        )

    def forward(self, x):
        b = self.backbone
        cls = torch.zeros((len(x), 1), dtype=torch.long, device=x.device)
        cats = torch.cat([cls, x[:, self.cat_idx].long()], dim=1)
        cat_tokens = b.embeds(cats + b.categories_offset)
        cont_tokens = None
        if self.num_idx:
            cont_tokens = torch.stack([layer(x[:, i:i+1]) for layer, i in zip(b.simple_MLP, self.num_idx)], dim=1)
        return b(cat_tokens, cont_tokens).reshape(-1)


class NodeHead(nn.Module):
    def __init__(self, n_features, params):
        super().__init__()
        from ._vendor.node.arch import DenseBlock
        from ._vendor.node.nn_utils import entmax15, entmoid15
        self.block = DenseBlock(n_features, layer_dim=params.get("layer_dim", 32),
                                num_layers=params.get("num_layers", 2), depth=params.get("tree_depth", 4),
                                tree_dim=1, flatten_output=False,
                                choice_function=entmax15, bin_function=entmoid15)

    def forward(self, x):
        return self.block(x)[..., 0].mean(dim=1)


class TabNetHead(nn.Module):
    def __init__(self, n_features, cat_idx, cat_dims, params):
        super().__init__()
        try:
            from pytorch_tabnet.tab_network import TabNet
        except ImportError as exc:
            raise ImportError('Install SurvTabDL with the "tabnet" extra.') from exc
        dim = params.get("n_d", 16)
        self.backbone = TabNet(input_dim=n_features, output_dim=1, n_d=dim, n_a=dim,
                              n_steps=params.get("n_steps", 3), cat_idxs=cat_idx, cat_dims=cat_dims,
                              cat_emb_dim=[2] * len(cat_idx), group_attention_matrix=torch.eye(n_features),
                              virtual_batch_size=params.get("virtual_batch_size", 128))
        self.sparsity_loss = None

    def forward(self, x):
        out, self.sparsity_loss = self.backbone(x)
        return out.reshape(-1)


def build_network(name, n_features, cat_idx, cat_dims, params):
    if name == "MLP":
        layers = []
        hidden = params.get("hidden_dim", 64)
        for _ in range(params.get("n_layers", 2)):
            layers.extend([nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(params.get("dropout", 0.1))])
            n_features = hidden
        return nn.Sequential(*layers, nn.Linear(n_features, 1))
    if name == "Cox":
        return nn.Linear(n_features, 1, bias=False)
    if name in ("SAINT", "FT-Transformer"):
        return AttentionHead(n_features, cat_idx, cat_dims, params, name == "SAINT")
    if name == "NODE":
        return NodeHead(n_features, params)
    if name == "TabNet":
        return TabNetHead(n_features, cat_idx, cat_dims, params)
    raise ValueError(f"Unknown neural architecture: {name}")
