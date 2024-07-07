import torch
import torch.nn as nn
import copy
import math


def Layer_clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    '''
    定义attention, 由于不需要保存任何状态, 写成函数比写成类更合适
    attention(q, k, v) = softmax(qk^T/\sqrt{d})v
    '''
    d_h = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_h)
    # 为什么用transpose(-2, -1)而不直接.T转置:在key为三维时也可以正常运行
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
        # 将mask矩阵为0的地方变成-1e9, 这里不能使用torch.inf, exp(torch.inf)=0，后面计算softmax的时候会出现Nan
    point = scores.softmax(dim=-1)
    # 在最后一个维度上进行归一化
    if dropout is not None:
        point = nn.functional.dropout(point, dropout)
    return torch.matmul(point, value), point

def seq_mask(q_pad, k_pad, subsequent_mask=False):
    # q_pad以及k_pad指的是pad的位置序列, shape为[n ,q_pad]
    assert q_pad.device == k_pad.device
    n, q_len = q_pad
    n, k_len = k_pad
    mask_shape = (n, 1, q_len, k_len)
    if subsequent_mask:
        mask = torch.tril(torch.ones(mask_shape), diagonal=1)
        mask = (mask == 0).to(q_pad.device)
    else:
        mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)
        for i in range(n):
            mask[i, :, q_pad[i], :] = 1
            mask[i, :, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
    return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, h_num, d_model, dropout=0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        if d_model % h_num != 0:
            print('Error!')
            return
        self.d_h = d_model//h_num
        self.h_num = h_num
        self.proj = Layer_clone(nn.Linear(d_model, d_model), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.drop = dropout
        self._attn_point = None
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 维度匹配
            mask = mask.unsqueeze(1)
        
        batch = query.size(0)
        # projection
        query, key, value = [
            proj(x).view(batch, -1, self.h_num, self.d_h).transpose(1, 2)
            # --> batch, h_num, seq_len, d_h
            for proj, x in zip(self.proj, (query, key, value))
        ]
        
        attn_ans, self._attn_point = attention(query, key, value, mask, self.drop)
        # 将拆成多头的attn_ans reshape合并起来
        attn_ans = attn_ans.transpose(1, 2).contiguous().view(batch, -1, self.d_h*self.h_num)
        # transpose --> batch, seq_len, h_num, d_h
        # view --> batch, seq_len, d_model
        # contiguous使得tensor在内存中是连续的
        del query, key, value
        
        return self.linear(attn_ans)
    
    def get_attn_point(self):
        return self._attn
    
# ffn(x) = max(0, xW_1+b_1)W_2 + b_2
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.w1(x)
        x = self.drop(nn.functional.relu(x))
        # relu起了公式里max的作用
        x = self.w2(x)
        return x
        
        
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout) -> None:
        super(AddNorm, self).__init__()
        # b seq_len hidden, 在hidden上归一化
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # PreNorm
        # norm(x + f(norm(x)))
        return self.norm(x + self.drop(sublayer(self.norm(x))))


class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, h_num=8, d_ff=2048, dropout=0.1) -> None:
        super(TransformerLayer, self).__init__()
        self.drop = dropout
        self.attn = MultiHeadAttention(h_num, d_model, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
    
    def forward(self, x, mask=None):
        x = self.add_norm1(x, lambda x : self.attn(x, x, x, mask))
        x = self.add_norm2(x, self.ffn)
        return x
    
class PositionEncode(nn.Module):
    def __init__(self, d_model, drpoout=0, max_len=4096) -> None:
        super(PositionEncode, self).__init__()
        pos_code = torch.zeros(max_len, d_model)
        position_num = torch.arange(0, max_len).unsqueeze(1)
        # 加了一个维度
        fm = 10000*torch.exp(2*torch.arange(0, d_model, 2)/d_model)
        pos_code[:, 0::2] = torch.sin(position_num * fm)
        pos_code[:, 1::2] = torch.cos(position_num * fm)
        pos_code = pos_code.unsqueeze(0)
        # 在第0维上加了一个维度
        # register_buffer可以将pos_code加载到缓冲区且不进行梯度计算
        # 还可以使用self.pos_encode访问
        self.register_buffer("pos_encode", pos_code, False)
        # False表示不放在state_dict中
        self.drop = nn.Dropout(drpoout)
        
    def forward(self, x):
        # 输入的x的shape为  b x seq_len x d_model
        b, seq_len, d_model = x.shape
        assert seq_len <= self.pos_encode[1]
        assert d_model == self.pos_encode[2]
        # 缩放word_vec x
        rescaled_x = x * math.sqrt(d_model)
        rescaled_x = rescaled_x + self.pos_encode[:, : seq_len].requires_grad_(False)
        # drop可选，随机丢失一部分位置编码，增加模型的健壮性
        return self.drop(x)
        
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h_num=8, d_ff=2048, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(h_num, d_model, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        
    def forward(self, x, mask=None):
        # add_norm要传入一个函数，因此用lambda
        x = self.add_norm1(x, lambda x : self.mha(x, x, x, mask))
        return self.add_norm2(x, self.ffn)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h_num=8, d_ff=2048, dropout=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.mha = MultiHeadAttention(h_num, d_model, dropout)
        self.mix_mha = MultiHeadAttention(h_num, d_model, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)
    
    def forward(self, encoder_out, decoder_out, encoder_mask=None, decoder_mask=None):
        decoder_out = self.add_norm1(decoder_out, lambda x : self.mha(x, x, x, decoder_mask))
        # encoder的输出做KV
        decoder_out = self.add_norm2(decoder_out, lambda x : self.mix_mha(x, encoder_out, encoder_out, encoder_mask))
        return self.add_norm3(decoder_out, self.ffn)
        

class Encoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(Encoder).__init__()
        self.encoder = Layer_clone(layer, N)
        self.norm = nn.LayerNorm(layer.shape)
        # layer.shape是啥
        
    def forward(self, x, mask):
        # encoder需要什么mask？
        for layer in self.encoder:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(Decoder, self).__init__()
        self.decoder = Layer_clone(layer, N)
        self.norm = nn.LayerNorm(layer.shape)
        
    def forward(self, decoder_out, encoder_out, encoder_mask, decoder_mask):
        for layer in self.decoder:
            decoder_out = layer(decoder_out, encoder_out, encoder_mask, decoder_mask)
        return self.norm(decoder_out)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h_num=8, dropout=0.1, pad_val=0) -> None:
        super(Transformer, self).__init__()
        self.encoder_emb = nn.Sequential(
            nn.Linear(d_model, src_vocab_size),
            PositionEncode(d_model, dropout))
        self.encoder = Encoder(
            EncoderLayer(
                d_model, 
                h_num,
                d_ff,
                dropout
                ),
            N)
        self.decoder_emb = nn.Sequential(
            nn.Linear(d_model, tgt_vocab_size),
            PositionEncode(d_model, dropout))
        self.decoder = Decoder(
            DecoderLayer(
                d_model,
                h_num,
                d_ff,
                dropout
                ),
            N)
        self.out_layer = nn.Sequential(
            nn.Linear(d_model, tgt_vocab_size),
            # batch, seq_len, vb
            nn.Softmax(dim=-1)
        )
        self.pad_val = pad_val
        # pad的值, ==pad说明此位置上为填充
    def forward(self, encoder_input, decoder_out, encoder_mask=None, encoder_out_mask=None, decoder_out_mask=None):
        encoder_out = self.encode(encoder_input, encoder_mask)
        decoder_out = self.decode(encoder_out, decoder_out, encoder_out_mask, decoder_out_mask)
        return self.out_layer(decoder_out)
        # 输出每个位置上词典中的词的概率
    
    def encode(self, encoder_input, encoder_mask=None):
        if not encoder_mask:
            src_pad = (encoder_input == self.pad_val)
            encoder_mask = seq_mask(src_pad, src_pad)
        return self.encoder(encoder_input, encoder_mask)
    
    def decode(self, encoder_out, decoder_out, encoder_out_mask, decoder_out_mask):
        if not (encoder_out_mask and decoder_out_mask):
            src_pad = (encoder_out == self.pad_val)
            tgt_pad = (decoder_out == self.pad_val)
            decoder_out_mask = seq_mask(tgt_pad, tgt_pad, True)
            encoder_out_mask = seq_mask(tgt_pad, src_pad)
        return self.decoder(encoder_out, decoder_out, encoder_out_mask, decoder_out_mask)
    



