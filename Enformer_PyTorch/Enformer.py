from AttentionEtc import *

class Enformer(nn.Module):
    def __init__(self, dim=1536, depth=11, heads=8, output_heads=dict(human = 5313), target_length=896, attn_dim_key=64, dropout_rate=0.4, attn_dropout=0.05, pos_dropout=0.01):
        super().__init__()
        self.dim = dim

        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(4, dim//2, 15, padding = 'same'),
            Residual(ConvBlock(dim//2)),
            AttentionPool(dim//2, pool_size = 2)
        )

        # conv tower
        filter_list = exponential_linspace_int(dim//2, dim, num = 6, divisible_by = 128)
        filter_list = [dim//2, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer
        transformer = []
        for _ in range(depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    MultiheadAttention(
                        dim,
                        heads = heads,
                        dim_key = attn_dim_key,
                        dim_value = dim // heads,
                        dropout = attn_dropout,
                        pos_dropout = pos_dropout,
                        num_rel_pos_features = dim // heads,
                    ),
                    nn.Dropout(dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

        # target cropping
        self.target_length = target_length
        self.crop_final = TargetLengthCrop(target_length)

        # final pointwise
        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], dim*2, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(dropout_rate / 8),
            GELU()
        )

        # trunk sequential module
        self._trunk = nn.Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        # final heads
        self.add_heads(**output_heads)

    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    def forward(self, x):
        x = self._trunk(x)
        out = map_values(lambda fn: fn(x), self._heads)
        return out
