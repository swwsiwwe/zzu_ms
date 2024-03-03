import torch
import torch.nn as nn
import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile

# import mindspore
# import mindspore.nn as nn
# import mindspore.ops as ops
# import pandas as pd
# from mindspore import Parameter

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
         super().__init__()
         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, group=nin,has_bias=True, pad_mode='pad')
         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1,has_bias=True, pad_mode='pad')

    def construct(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class channelshrink(nn.Module):
    def __init__(self, dim, ratio, wh):
        super().__init__()
        self.dim = dim
        self.dim_out = dim//ratio
        self.dwconv = depthwise_separable_conv(nin=self.dim, nout=self.dim_out)
        self.wh = wh
        self.layernorm = nn.LayerNorm([self.dim_out, self.wh, self.wh])
        self.act = nn.GELU()

    def forward(self, x):
        # x = rearrange(x, 'B (W H) C -> B C W H', W=self.wh, H=self.wh)
        out = self.dwconv(x)
        out = self.layernorm(out)
        out = self.act(out)
        # out = rearrange(out, 'B C W H -> B (W H) C')

        return out


class SptialTopk(nn.Module):
    def __init__(self,topk, dimq, dimv,head):
        super().__init__()
        self.topk = topk
        self.dimq = dimq
        self.dimv = dimv
        self.nH = head

    def forward(self, q, qq, bias):
        x = q.mean(dim=-1)
        y = bias
        B, _, _ = q.shape
        val, idx = torch.topk(x, k=self.topk, dim=-1, largest=True, sorted=False)  # k smallest elements
        sorted_idx, new_idx = torch.sort(idx, dim=-1)
        p = torch.gather(idx, dim=-1, index=new_idx)
        ppp = p[:, :, None]
        pppq = ppp.expand(-1, -1, self.dimq)
        pp = rearrange(p, '(B H) wh -> B H wh', H=self.nH)
        pp = pp[:, :, None, :]
        pp = pp.expand(-1, -1, 49, -1)
        yy = y[None, :, :, :]
        yy = yy.expand(B//self.nH, -1, -1, -1)
        qal = torch.gather(qq, dim=1, index=pppq)
        yal = torch.gather(yy, dim=-1, index=pp)

        pppv = ppp.expand(-1, -1, self.dimv)
        val = torch.gather(q, dim=1, index=pppv)

        return qal, val, yal

class ChannelTopk(nn.Module):
    def __init__(self,topk, width, high, in_dim):
        super().__init__()
        self.topk = topk
        self.w = width
        self.h = high
        self.indim = in_dim

        self.compute = nn.Sequential(
            nn.Linear(self.indim*2, self.indim),
            nn.GELU(),
            nn.Softmax(dim=1)
        )



    def forward(self,q):

        x = q.mean(dim=1)
        x1, _ = q.max(dim=1)
        x = torch.cat((x, x1), dim=1)
        x = self.compute(x)
        val, idx = torch.topk(input=x, k=self.topk, dim=-1, largest=True, sorted=False)
        sorted_idx, new_idx = torch.sort(idx, dim=-1)
        p = torch.gather(idx, dim=-1, index=new_idx)
        ppp = p[:, None, :]
        ppp = ppp.expand(-1, 49, -1)
        val = torch.gather(q, dim=-1, index=ppp)
        return val


class Mlp(nn.Module):
    def __init__(self, in_feature, hidden_feature=None, out_feature=None, act_layer=nn.GELU,drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature, out_feature)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def window_partition(x, window_size):
    """

    :param x: (B, C, W, H)
    :param window_size:
    :return:
    """

    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.contiguous()
    x = x.view(-1, window_size, window_size, C)

    return x


def window_reverse(windows, window_size, H, W):
    """
    恢复窗口的函数
    :param windows:
    :param window_size:
    :param H:
    :param W:
    :return:
    """

    B = int((windows.shape[0])/(H*W/window_size/window_size))
    x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, W, H, -1)

    return x


class channelattention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., top_dim=64):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)  # 截断正太分布

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.top_dim = ChannelTopk(top_dim, self.window_size, self.window_size, in_dim=self.dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        qcha = kcha = self.top_dim(q)
        qcha = qcha * self.scale

        qcha = rearrange(qcha, 'b n (h d) -> b h n d', h=self.num_heads)
        kcha = rearrange(kcha, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        attn = qcha @ kcha.transpose(2, 3)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'b n wh d -> b wh (n d)', n=self.num_heads, wh=self.window_size*self.window_size)
        x = self.proj(x)  # 经过一个线性转换
        x = self.proj_drop(x)
        return x


class sptialattention(nn.Module):
    def __init__(self, dim, size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., top_sptial=64, ratio=2):
        super().__init__()
        self.dim = dim
        self.size = size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * size - 1) * (2 * size - 1), num_heads)
        )

        coords_h = torch.arange(self.size)
        coords_w = torch.arange(self.size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.size - 1
        relative_coords[:, :, 0] *= 2 * self.size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)  # 截断正太分布

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim//2, dim//2, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.sig = nn.Sigmoid()

        self.top_sptial = SptialTopk(top_sptial, dimq=(self.dim//self.num_heads)//2, dimv=(self.dim//self.num_heads), head=self.num_heads)
        self.top_channel = ChannelTopk(topk=(self.dim//self.num_heads)//2, width=self.size, high=self.size, in_dim=self.dim//self.num_heads)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.to_q(x)
        q = rearrange(q, 'B wh (n c) -> (B n) wh c', n=self.num_heads)
        qcha = self.top_channel(q)
        qcha1 = rearrange(qcha, '(B n) wh c -> B wh (n c)', n=self.num_heads)
        k = self.to_k(qcha1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.size * self.size, self.size * self.size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        qspt, vspt, index = self.top_sptial(q, qcha,relative_position_bias)
        vspt = rearrange(vspt, '(B n) wh c -> B wh (n c)', n=self.num_heads)
        vspt = self.to_v(vspt)

        qspt = qspt * self.scale

        qspt = rearrange(qspt, '(b n) wh d -> b n wh d', n=self.num_heads)
        k = rearrange(k, 'b wh (n d) -> b n wh d', n=self.num_heads)
        vspt = rearrange(vspt, 'b wh (n d) -> b n wh d', n=self.num_heads)
        attn = k @ qspt.transpose(2, 3)
        attn = attn + index

        attn1 = attn.mean(dim=-1)
        attn1 = self.sig(attn1)
        # attn1 = self.softmax(attn1)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = attn @ vspt
        attn1 = attn1[:, :, :, None]
        x = torch.mul(x, attn1)
        x = rearrange(x, 'b n wh d -> b wh (n d)', n=self.num_heads)
        x = self.proj(x)  # 经过一个线性转换
        x = self.proj_drop(x)
        return x


class BlackholeBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_dorp=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, top_channel=64, top_sptial=28):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn_sptial = sptialattention(dim, size=self.window_size, num_heads=self.num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_dorp, proj_drop=drop,
            top_sptial=top_sptial,
        )

        self.drop_path = DropPath(drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_feature=dim, hidden_feature=mlp_hidden_dim, out_feature=dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H*W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x1 = x

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # 如果shifted不为空则进行平移

        x_window = window_partition(shifted_x, self.window_size)
        x_window = x_window.view(-1, self.window_size*self.window_size, C)

        # spatial attention
        att_sptial = self.attn_sptial(x_window)
        att_sptial = att_sptial.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(att_sptial, self.window_size, H, W)
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x
        x1 = x1.view(B, H*W, C)

        # confusion
        # out = torch.concat((x, x1, shortcut), dim=-1)
        #
        # out = self.confusion1(out)
        # # out = rearrange(out, 'B (W H) C -> B C W H', W=self.input_resolution[0])
        # # out = self.confusion(out)
        # # out = rearrange(out, 'B C W H -> B (W H) C', W=self.input_resolution[0])

        # FFN
        x = shortcut + self.drop_path(x1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 截断正太分布
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """

        :param x:
        :return:
        """

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [batch*patch, head_num, window_size**2, head_dim]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [batch*patch, head_num, window_size**2, window_size**2]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 经过一个线性转换
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_dorp=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_dorp, proj_drop=drop
        )

        self.drop_path = DropPath(drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_feature=dim, hidden_feature=mlp_hidden_dim, out_feature=dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H*W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # 如果shifted不为空则进行平移

        x_window = window_partition(shifted_x, self.window_size)
        x_window = x_window.view(-1, self.window_size*self.window_size, C)
        # 得到shift_window后的窗口 [Batch*numPatch, window_size**2, Channel]

        att_window = self.attn(x_window, mask=self.attn_mask)
        # 得到自我局部注意力后的模型

        att_window = att_window.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(att_window, self.window_size, H, W)
        # 得到4维度的张良， 为了使得恢复到shift前到

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H*W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_dorp=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_dorp, proj_drop=drop
        )

        self.drop_path = DropPath(drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_feature=dim, hidden_feature=mlp_hidden_dim, out_feature=dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H*W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # 如果shifted不为空则进行平移

        x_window = window_partition(shifted_x, self.window_size)
        x_window = x_window.view(-1, self.window_size*self.window_size, C)
        # 得到shift_window后的窗口 [Batch*numPatch, window_size**2, Channel]

        att_window = self.attn(x_window, mask=self.attn_mask)
        # 得到自我局部注意力后的模型

        att_window = att_window.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(att_window, self.window_size, H, W)
        # 得到4维度的张良， 为了使得恢复到shift前到

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H*W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

# b = BlackholeBlock(dim=384, input_resolution=[14, 14], num_heads=8, window_size=7, shift_size=0, top_channel=128, top_sptial=28)
# a = torch.rand((1, 14*14, 384))
# c = b(a)
# print(c.size())