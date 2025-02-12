import torch
from linformer import LinformerSelfAttention

attn = LinformerSelfAttention(
    dim = 73,
    seq_len = 128 ** 2,
    heads = 1,
    k = 16,
    one_kv_head = True,
    share_kv = True
)

x = torch.randn(1, 128 ** 2, 73)
context = torch.randn(1, 128 ** 2, 73)
attn(x, context) # (1, 2048, 512)