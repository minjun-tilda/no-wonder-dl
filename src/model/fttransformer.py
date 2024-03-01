from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch

# https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py#L61
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py#L7
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py#L62
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_in, d_hid * 2)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.swiglu = SwiGLU()
        self.layernorm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_1(x)
        x = self.swiglu(x)
        x = self.w_2(x)

        x = self.dropout(x)
        x += residual

        x = self.layernorm(x)

        return x

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py#L9   
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            q[:, :, mask, :] = 0
            k[:, :, mask, :] = 0
            v[:, :, mask, :] = 0

            mask = torch.FloatTensor([1 if x != mask+1 else 0 for x in range(q.shape[2])]).unsqueeze(-2).unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layernorm(q)

        return q, attn

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Layers.py#L10
class EncodingLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L48C1-L84C27
class Encoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncodingLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        src_seq= self.dropout(src_seq)
        enc_output = self.layernorm(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class NumericalEmbedder(nn.Module):
    def __init__(self, n_numerical, d_model):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(n_numerical, d_model))
        self.biases = nn.Parameter(torch.randn(n_numerical, d_model))

    def forward(self, x):
        x = x.unsqueeze(-1)
        return x * self.weights + self.biases

class Embedder(nn.Module):
    def __init__(self, sh_categorical, n_numerical, d_model):
        super().__init__()

        self.n_categorical = len(sh_categorical)
        self.n_uni_categorical = sum(sh_categorical)

        self.n_numerical = n_numerical

        if self.n_uni_categorical:
            categorical_offset = F.pad(torch.tensor(list(sh_categorical)), (1, 0), value = 0)
            categorical_offset = categorical_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categorical_offset', categorical_offset)

            self.categorical_embedder = nn.Embedding(self.n_uni_categorical, d_model)

        if n_numerical:
            self.numerical_embedder = NumericalEmbedder(n_numerical, d_model)


    def forward(self, x_categ, x_numer, mask=None):
        xs = []

        if self.n_categorical > 0:
            x_categ = self.categorical_embedder(x_categ + self.categorical_offset)
            xs.append(x_categ)
        
        if self.n_numerical > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        
        x = torch.cat(xs, dim = 1)

        if mask is not None:
            x[:, mask, :] = 0

        return x

# https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py#L113
class FTTransformer(nn.Module):
    def __init__(self, sh_categorical, n_numerical, n_layers, n_head, d_k, d_v, d_model, d_inner, d_out):
        super().__init__()
        self.input_length = len(sh_categorical) + n_numerical

        self.embedder = Embedder(sh_categorical, n_numerical, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.encoder = Encoder(n_layers, n_head, d_k, d_v, d_model, d_inner)
        
        self.layernorm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, x_categ, x_numer, mask:int=None, return_attns = False):
        x = self.embedder(x_categ, x_numer, mask)

        b = x.shape[0]
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim = 1)

        mask = torch.FloatTensor([1 if x != mask+1 else 0 for x in range(self.input_length + 1)])

        x, attns = self.encoder(x, mask, True)

        x = x[:, 0]

        x = self.layernorm(x)
        x = F.relu(x)
        x = self.linear(x)

        if return_attns:
            return x, attns
        return x