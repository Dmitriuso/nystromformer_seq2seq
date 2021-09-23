import torch
import torch.nn as nn
import torch.optim as optim
from layers.mha_layer import MultiHeadAttentionLayer
from layers.NystromAttn import attn
from layers.pff import PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 head_dim,
                 n_heads,
                 n_landmarks,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = attn
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src_len]

        #self attention
        _src = self.self_attention(src, src_mask)

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src