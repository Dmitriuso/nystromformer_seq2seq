import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx)

        #src_mask = [batch size, 1, 1, src len]
        # print(f'size of src_mask: {src_mask.shape}')

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        # trg_pad_mask = (trg != self.trg_pad_idx)

        #trg_pad_mask = [batch size, trg len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((batch_size, trg_len), device = self.device)).bool()

        #trg_sub_mask = [batch size, trg len]

        # trg_mask = trg_pad_mask & trg_sub_mask

        #trg_mask = [batch size, 1, trg len, trg len]
        # print(f'size of trg mask: {trg_mask.shape}')

        return trg_sub_mask

    def forward(self, src, trg):

        #src = [batch size, src len]
        # print(f'src : {src.shape}')

        #trg = [batch size, trg len]
        # print(f'trg : {trg.shape}')

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        #enc_src = [batch size, src len, hid dim]
        # print(f'enc_src : {enc_src.shape}')

        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        return output