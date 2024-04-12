import rootutils
import unittest
import torch
import lightning as L


root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


from src.transformer import (
    scale_dot_product_attention,
    MultiHeadAttention,
    EncoderBlock,
    TransformerEncoder
)


class TestTransformer(unittest.TestCase):
    def test_scale_dot_product_attention(self):
        seq_len, d_k = 3, 2
        L.seed_everything(42)
        q = torch.randn(seq_len, d_k)
        k = torch.randn(seq_len, d_k)
        v = torch.randn(seq_len, d_k)
        values = scale_dot_product_attention(q, k, v)
        self.assertEqual(values.size(), (seq_len, d_k))

    def test_multi_head_attention(self):
        batch_size = 10
        seq_len = 4
        input_dim, embed_dim, num_heads = 3, 10, 2
        attn = MultiHeadAttention(input_dim, embed_dim, num_heads)
        x = torch.randn((batch_size, seq_len, input_dim))
        values = attn(x)
        self.assertEqual(values.size(), (batch_size, seq_len, embed_dim))

    def test_multi_head_attention_raise(self):
        input_dim, embed_dim, num_heads = 3, 10, 3
        with self.assertRaises(AssertionError):
            MultiHeadAttention(input_dim, embed_dim, num_heads)

    def test_encoder_block(self):
        input_dim, num_heads, dim_ff = 15, 5, 6
        batch_size = 10
        seq_len = 4
        encoder = EncoderBlock(input_dim, num_heads, dim_ff)
        x = torch.randn((batch_size, seq_len, input_dim))
        values = encoder(x)
        self.assertEqual(values.size(), (batch_size, seq_len, input_dim))

    def test_transformer_encoder(self):
        input_dim, num_heads, dim_ff, num_layers = 15, 5, 6, 4
        batch_size = 10
        seq_len = 4
        encoder = TransformerEncoder(num_layers, input_dim=input_dim, num_heads=num_heads, dim_feedforward=dim_ff)
        x = torch.randn((batch_size, seq_len, input_dim))
        values = encoder(x)
        self.assertEqual(values.size(), (batch_size, seq_len, input_dim))


if __name__ == '__main__':
    unittest.main()
