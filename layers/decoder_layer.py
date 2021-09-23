import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mha_layer import MultiHeadAttentionLayer
from layers.NystromAttn import attn
from layers.pff import PositionwiseFeedforwardLayer


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 head_dim,
                 n_heads,
                 n_landmarks,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = attn
        self.encoder_attention = attn
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]

        #self attention
        _trg = self.self_attention(trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #encoder attention
        _src = self.encoder_attention(enc_src, src_mask)

        #dropout, residual connection and layer norm
        max_len = max([trg.shape[1], _src.shape[1]])
        # cannot otherwise add trg and dropout(_src),
        # because src len and trg len dimensions are different
        trg = \
            self.enc_attn_layer_norm(
                F.pad(trg, (0, 0, 0, max_len - trg.shape[1])) +
                self.dropout(F.pad(_src, (0, 0, 0, max_len - _src.shape[1])))
            )

        #trg = [batch size, trg len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg