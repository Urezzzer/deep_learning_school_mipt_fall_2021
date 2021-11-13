import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                           dropout=(dropout if n_layers > 1 else 0), bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim)
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim, temperature=5):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1)
        self.t = temperature

    def softmax__(self, x):
        e_x = torch.exp(x / self.t)
        return e_x / torch.sum(e_x, dim=1).unsqueeze(1)

    def forward(self, encoder_outputs, hidden):
        hidden = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.shape[0], 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs.permute(1, 0, 2), hidden), dim=2)))
        attention = self.v(energy).squeeze(2)

        return self.softmax__(attention)


class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=n_layers, dropout=(dropout if n_layers > 1 else 0))
        self.out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, encoder_outputs, prev_hidden):
        inp = inp.unsqueeze(0)
        embedded = self.dropout(self.embedding(inp))
        a = self.attention(encoder_outputs, prev_hidden).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, prev_hidden)
        prediction = self.out(torch.cat((embedded.squeeze(0), weighted.squeeze(0), output.squeeze(0)), dim=1))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim * (encoder.bidirectional + 1) == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch_size = trg.shape
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        enc_states, hidden = self.encoder(src)
        inp = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(inp, enc_states, hidden)
            outputs[t] = output
            teacher_force = np.random.uniform() < teacher_forcing_ratio
            top1 = output.argmax(-1)
            inp = trg[t] if teacher_force else top1

        return outputs
