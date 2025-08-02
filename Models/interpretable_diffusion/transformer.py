import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp

'主要保存序列分解相关模块'


'用多项式回归建模趋势'
class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        
        trend_poly = 3
        
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,                          # 激活函数gelu
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):                 # 输入是transformer加conv后的结果
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)     # 应该是公式3的linear  不包括右边的均值  加均值在最后加上  在470行左右位置
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))  # 公式3里乘以C
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    
    
'用移动平均建模趋势'
class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


'用反傅里叶变换建模周期 选topk个基函数  应该是ETSformer的方法'
class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    
    
    
'用傅里叶基函数建模周期  应该是nbeats里的方法'
class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


    
    
class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    
    
    
'单个编码器模块'
class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)   # 扩散步嵌入 + 层归一化
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, d=None, mask=None, label_emb=None):
        a, att = self.attn( self.ln1(x, timestep, d, label_emb), mask=mask )  
        x = x + a
        x = x + self.mlp(self.ln2(x))          # 编码器里的FFN  only one really use encoder_output
        return x, att                    # 下一层的输入和注意力图

    
    
    
    
'整个编码器模块'
class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

        
    def forward(self, input, t, d=None, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, d=d, mask=padding_masks, label_emb=label_emb)
        return x


    
'单个解码器组件'
class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

        
    def forward(self, x, encoder_output, timestep, d=None, mask=None, label_emb=None):
        
        a, att = self.attn1(self.ln1(x, timestep, d, label_emb), mask=mask)     # 这四行就是一个transformer  两个注意力 一个交叉注意力 融入编码器信息
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep, d ), encoder_output, mask=mask)
        x = x + a
        
        #分成两路 一路计算周期和趋势  一路进入FFN
        x1, x2 = self.proj(x).chunk(2, dim=1)         # 这一路计算趋势和周期
        trend, season = self.trend(x1), self.seasonal(x2)
        
        x = x + self.mlp(self.ln2(x))              # 这一路进入FFN
        m = torch.mean(x, dim=1, keepdim=True)        # 需要算一次均值 与上面的x相减 作为下一个模块的输入
        
        return x - m, self.linear(m), trend, season     # x-m作为下一个模块的输入  m经过线性层应该是为了扩维 然后累积与趋势相加
    
    
    
    

'整个解码器'
class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512    
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])
    
    def forward(self, x, t,enc, d=None, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        
        for block_idx in range(len(self.blocks)):
            
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, d=d, mask=padding_masks, label_emb=label_emb)
            
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)         # mean累积后与趋势相加
        return x, mean, trend, season         # x是最后一个本应作为下一个组件的输入


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

    def forward(self, input, t, d=None, padding_masks=None):
        
        emb = self.emb(input)                   # 第一个mlp卷积
        
        inp_enc = self.pos_enc(emb)               # 第一个编码器的pe
        
        enc_cond = self.encoder(inp_enc, t, d=d, padding_masks=padding_masks)  # 输入编码器 监督信息

        inp_dec = self.pos_dec(emb)               # 解码器的pe
        
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, d=d, padding_masks=padding_masks)

        res = self.inverse(output)               # output是最后一个模块的输入 本来是用于下一个模块的输入
        res_m = torch.mean(res, dim=1, keepdim=True)   # 最后一个均值 即使是最后一个输入 也要计算均值 然后就相减 x-m 和DecoderBlock里一样
                                        # 体现为下面的 res - res_m 
        
        '加上res的操作应该是周期相加处的加号 上面的虚线输入'
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        
        '累积的mean与趋势相加 应该是公式3的右项  还要加上最后一个mean 因为这个mean没有来得及和之前的mean拼接'
        trend = self.combine_m(mean) + res_m + trend     

        return trend, season_error


if __name__ == '__main__':
    pass



