import rootutils
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange


logging.basicConfig(level=logging.INFO)
# import from src after this line
root_path = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


def scale_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    d_k = q.size(-1)  # b h l d
    attn = einsum("...ld,...td->...lt", q, k) / \
        torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # ... len, len
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn = F.softmax(attn, dim=-1)  # ... seq_len, seq_len
    values = einsum("...lt,...td->...ld", attn, v)
    return values  # ... seq_len, d_k


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # stacked q, k, v projections for efficiency
        self.qkv_proj = nn.Linear(input_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_length, input_dim = x.size()
        # (batch_size, seq_length, embed_dim * 3)
        qkv: torch.Tensor = self.qkv_proj(x)
        qkv = rearrange(qkv, "b l (h n hd) -> n b h l hd",
                        h=self.num_heads, hd=self.head_dim)
        # (batch_size, num_heads, seq_length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (batch_size, num_heads, seq_length, head_dim)
        values = scale_dot_product_attention(q, k, v, mask=mask)
        values = rearrange(values, "b h l hd -> b l (h hd)")

        return values


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        self.attn = MultiHeadAttention(input_dim, input_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask=mask)
        x = self.norm1(x)

        x = x + self.feedforward(x)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


@hydra.main(version_base=None, config_path=config_path, config_name="transformer")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
