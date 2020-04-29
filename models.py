import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FastWeights(nn.Module):
    def __init__(self,
                 input_size=10,
                 hidden_size=128,
                 out_size=10,
                 fast_lr=0.5,
                 decay_lr=0.95,
                 control=True,
                 layer_norm=False,
                 extra_depth=False,
                 device='cuda',
                 **kwargs):
        super(FastWeights, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fast_lr = fast_lr
        self.decay_lr = decay_lr
        self.control = control
        self.layer_norm = layer_norm
        self.extra_depth = extra_depth
        self.device = device

        self.W_ih = nn.Parameter(torch.FloatTensor(
            self.input_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.FloatTensor(
            self.hidden_size))
        self.register_parameter('W_ih', self.W_ih)
        self.register_parameter('b_ih', self.b_ih)

        self.W_hh = nn.Parameter(torch.FloatTensor(
            self.hidden_size, self.hidden_size))
        self.b_hh = nn.Parameter(torch.FloatTensor(
            self.hidden_size))
        self.register_parameter('W_hh', self.W_hh)
        self.register_parameter('b_hh', self.b_hh)

        self.W_y = nn.Parameter(torch.FloatTensor(
            self.hidden_size, self.out_size))
        self.b_y = nn.Parameter(torch.FloatTensor(
            self.out_size))
        self.register_parameter('W_y', self.W_y)
        self.register_parameter('b_y', self.b_y)

        self.reset_parameters()

        self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, input):
        bs = input.size(1)
        hidden = torch.zeros(bs, self.hidden_size, device=self.device)
        out = [hidden]
        for t in range(input.size(0)):
            if not self.control:
                if self.extra_depth:
                    # preliminary_vector
                    hidden = F.relu(
                        torch.einsum('hh,bh->bh', (self.W_hh, out[t])) +
                        self.b_hh + self.b_ih +
                        torch.einsum('ih,bi->bh', (self.W_ih, input[t]))
                    )
                    # memory matrix
                    if t < 1:
                        memory_matrix = torch.zeros(
                            bs, self.hidden_size, self.hidden_size, device=self.device
                        )
                    else:
                        for tau in range(t):
                            memory_matrix = memory_matrix + (self.decay_lr ** (t - tau)) * (
                                torch.einsum('bh,bj->bhj', (out[tau], out[tau]))
                            )
                    memory_matrix = self.fast_lr * memory_matrix
                    hidden = torch.einsum('hh,bh->bh', (self.W_hh, out[t])) + (
                        self.b_hh + self.b_ih +
                        torch.einsum('ih,bi->bh', (self.W_ih, input[t])) +
                        torch.einsum('bhh,bh->bh', (memory_matrix, hidden))
                    )
                else:
                    # memory matrix
                    if t < 1:
                        memory_matrix = torch.zeros(
                            bs, self.hidden_size, self.hidden_size, device=self.device
                        )
                    else:
                        for tau in range(t):
                            memory_matrix = memory_matrix + (self.decay_lr ** (t - tau)) * (
                                torch.einsum('bh,bj->bhj', (out[tau], out[tau]))
                            )
                    memory_matrix = self.fast_lr * memory_matrix
                    # compute hidden vector
                    hidden = torch.einsum('hh,bh->bh', (self.W_hh, out[t])) + (
                        self.b_hh + self.b_ih +
                        torch.einsum('ih,bi->bh', (self.W_ih, input[t])) +
                        torch.einsum('bhh,bh->bh', (memory_matrix, out[t]))
                    )
                if self.layer_norm:
                    hidden = F.relu(self.ln(hidden))
                else:
                    hidden = F.relu(hidden)
            else:
                hidden = F.relu(
                    torch.einsum('hh,bh->bh', (self.W_hh, out[t])) +
                    self.b_hh + self.b_ih +
                    torch.einsum('ih,bi->bh', (self.W_ih, input[t]))
                )
            out.append(hidden)
        out = out[-1]
        logits = (
            torch.einsum('ho,bh->bo', (self.W_y, out)) + self.b_y
        )
        return logits

    def reset_parameters(self):
        stdv = np.sqrt(2.0 / self.hidden_size)
        for name, weight in self.named_parameters():
            if 'W' in name:
                if name == 'W_hh':
                    nn.init.eye_(weight)
                    weight.data *= 0.05
                else:
                    nn.init.uniform_(weight, -stdv, stdv)
            else:
                nn.init.zeros_(weight)
