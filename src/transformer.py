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
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


def scale_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    d_k = q.size(-1)
    attn = torch.matmul(q, k.transpose(-2, -1))
    attn /= torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # ... seq_len, seq_len
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn = F.softmax(attn, dim=-1)  # ... seq_len, seq_len
    return torch.matmul(attn, v)  # ... seq_len, d_k


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, embed_dim * 3)  # stacked q, k, v projections for efficiency
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_length, input_dim = x.size()
        qkv: torch.Tensor = self.qkv_proj(x)  # (batch_size, seq_length, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # (batch_size, num_heads, seq_length, head_dim)

        values = scale_dot_product_attention(q, k, v, mask=mask)  # (batch_size, num_heads, seq_length, head_dim)
        values = values.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_dim)
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        return values


@hydra.main(version_base=None, config_path=config_path, config_name="transformer")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
