import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(
            self,
            cat_features_size,
            num_features_size,
            dropout=0.3
    ):
        super(Model, self).__init__()

        self.embedding_size = 4
        self.cat_hidden_size = 64

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(feature_size, self.embedding_size, padding_idx=0)
                for feature_size in cat_features_size
            ]
        )
        self.cat_proj = nn.Sequential(
            nn.Linear(self.embedding_size * len(cat_features_size), self.cat_hidden_size),
            nn.LayerNorm(self.cat_hidden_size),
        )

        self.hidden_size = 64

        # encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features_size + self.cat_hidden_size, momentum=0.1, affine=False),
            nn.Dropout(dropout),
            nn.Linear(num_features_size + self.cat_hidden_size, self.hidden_size * 2),
            nn.BatchNorm1d(self.hidden_size * 2, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.BatchNorm1d(self.hidden_size // 2, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1),
        )

        self.apply(self.init_weights)

    def init_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0., std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        if isinstance(module, nn.LayerNorm) and module.weight is not None:
            module.weight.data.fill_(1.)

        if isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, cat_features, num_features):

        batch_size = cat_features.shape[0]

        cat_emb = []
        for cat_feature_idx in range(cat_features.shape[1]):
            cat_emb.append(self.cat_embeddings[cat_feature_idx](cat_features[:, [cat_feature_idx]]))

        cat_emb = torch.cat(cat_emb, dim=1)
        cat_emb = self.cat_proj(cat_emb.reshape(batch_size, -1))

        hidden_states = torch.cat([cat_emb, num_features], dim=1)
        pred = self.encoder(hidden_states)

        return pred
