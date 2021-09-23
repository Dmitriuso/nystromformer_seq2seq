import torch
from nystrom_attention import NystromAttention

attn = NystromAttention(
    dim = 128,
    dim_head = 16,
    heads = 4,
    num_landmarks = 64,    # number of landmarks
    pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
    residual = True         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
)

# x = torch.randn(1, 16384, 512)
# mask = torch.ones(1, 16384).bool()
#
# attn(x, mask = mask) # (1, 16384, 512)