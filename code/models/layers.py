import copy, math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from torch import nn, einsum
from einops import rearrange, repeat

import pdb
from pprint import pprint

DEFAULT_THRESHOLD = 5e-3

class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None

class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > self.threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput

class SharableConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(SharableConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        # Give real-valued mask weights per task to manage the shared part from previous tasks.
        self.piggymask = None

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            # print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = Binarizer.apply
        elif threshold_fn == 'ternarizer':
            print('Calling ternarizer with threshold:', threshold)
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input, layer_info=None, name=None):
        if self.piggymask is not None:
            # Get binarized/ternarized mask from real-valued mask.
            mask_thresholded = self.threshold_fn(self.piggymask, self.info['threshold'])
            # Mask weights with above mask.
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight

        # Perform conv using modified weight.
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)

class SharableLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(SharableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
        self.weight = Parameter(torch.Tensor(
            out_features, in_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.piggymask = None

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer.apply
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input):
        if self.piggymask is not None:
            # pdb.set_trace()
            # Get binarized/ternarized mask from real-valued mask.
            mask_thresholded = self.threshold_fn(self.piggymask, self.info['threshold'])
            # Mask weights with above mask.
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight
        # Get output using modified weight.
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)

class SharableMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False, 
                 mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', threshold=None):
        super(SharableMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Shared weight matrices with binary masks
        self.in_proj = SharableLinear(embed_dim, 3 * embed_dim, bias=bias,
                                      mask_init=mask_init, mask_scale=mask_scale,
                                      threshold_fn=threshold_fn, threshold=threshold)
        self.out_proj = SharableLinear(embed_dim, embed_dim, bias=bias,
                                       mask_init=mask_init, mask_scale=mask_scale,
                                       threshold_fn=threshold_fn, threshold=threshold)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        batch_size, seq_length, _ = query.shape

        if self.batch_first:
            query, key, value = query.transpose(1, 0), key.transpose(1, 0), value.transpose(1, 0)

        # Linear projections
        qkv = self.in_proj(query)  # Project only query for self-attention
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape into multi-head format
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_weights if need_weights else None

# class SharableMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, bias=True, 
#                  mask_init = '1s', mask_scale=1e-2,
#                  threshold_fn='binarizer', threshold=None):
#         super(SharableMultiheadAttention, self).__init__()
#         assert embed_dim % num_heads == 0

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.threshold_fn = threshold_fn
#         self.mask_scale = mask_scale
#         self.mask_init = mask_init
#         #self.scale = self.d_k  ** -0.5
#         #self.linears = clones(SharableLinear(embed_dim))
#         #self.attn = None
#         #self.dropout = nn.Dropout(dropout)     #ToDo: 나중에 필요하게 되면 추가로
        
#         if threshold is None:
#             threshold = DEFAULT_THRESHOLD

#         self.info = {
#             'threshold_fn': threshold_fn,
#             'threshold': threshold,
#         }

#         # self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias) 
#         # self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         # self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.q_proj = SharableLinear(embed_dim, embed_dim, bias=bias,
#                                      mask_init=mask_init, mask_scale=mask_scale,
#                                      threshold_fn=threshold_fn, threshold=threshold)
#         self.k_proj = SharableLinear(embed_dim, embed_dim, bias=bias,
#                                      mask_init=mask_init, mask_scale=mask_scale,
#                                      threshold_fn=threshold_fn, threshold=threshold)
#         self.v_proj = SharableLinear(embed_dim, embed_dim, bias=bias,
#                                      mask_init=mask_init, mask_scale=mask_scale,
#                                      threshold_fn=threshold_fn, threshold=threshold)
#         self.out_proj = SharableLinear(embed_dim, embed_dim, bias=bias,
#                                        mask_init=mask_init, mask_scale=mask_scale,
#                                        threshold_fn=threshold_fn, threshold=threshold)

#         # self.piggymask = None   # 모든 layer의 piggymask를 하나로 퉁치기기
#         self.piggymask_q = None
#         self.piggymask_k = None
#         self.piggymask_v = None
#         self.piggymask_out = None

#         if threshold_fn == 'binarizer':
#             self.threshold_fn = Binarizer.apply
#         elif threshold_fn == 'ternarizer':
#             self.threshold_fn = Ternarizer(threshold=threshold)

#     def apply_piggymask(self, weight, mask):
#             """Apply the piggyback mask to the weight matrix."""
#             if mask is not None:
#                 mask_thresholded = self.threshold_fn(mask, self.info['threshold'])
#                 masked_weight = mask_thresholded * weight
#             else:
#                 masked_weight = weight
#             return masked_weight
    
#     def forward(self, query, key, value, attn_mask=None, need_weights=False):
#         # batch_size, seq_length, _ = query.size()
#         batch_size = query.shape[1]  # batch_size를 동적으로 가져옴
#         print(f'batch_size={batch_size}')
#         query_len, key_len = query.shape[0], key.shape[0]
#         print(query.shape, key.shape)
#         print(f'query_len={query_len}, key_len={key_len}')

#         assert self.embed_dim == self.num_heads * self.head_dim, \
#             f"embed_dim ({self.embed_dim}) must be equal to num_heads ({self.num_heads}) * head_dim ({self.head_dim})"

#         q_weight = self.apply_piggymask(self.q_proj.weight, self.piggymask_q)
#         k_weight = self.apply_piggymask(self.k_proj.weight, self.piggymask_k)
#         v_weight = self.apply_piggymask(self.v_proj.weight, self.piggymask_v)
#         out_weight = self.apply_piggymask(self.out_proj.weight, self.piggymask_out)


#         q = F.linear(query, q_weight, self.q_proj.bias)
#         k = F.linear(key, k_weight, self.k_proj.bias)
#         v = F.linear(value, v_weight, self.v_proj.bias)

#         # q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         # k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         # v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         print(f"q.shape before view: {q.shape}")  # 디버깅 추가
#         print(f"k.shape before view: {k.shape}")
#         print(f"v.shape before view: {v.shape}")
#         head_dim = self.embed_dim // self.num_heads
#         q = q.view(batch_size, query_len, self.num_heads, head_dim).transpose(1, 2)
#         k = k.view(batch_size, key_len, self.num_heads, head_dim).transpose(1, 2)
#         v = v.view(batch_size, key_len, self.num_heads, head_dim).transpose(1, 2)

#         print(f"q.shape after view: {q.shape}")  # 디버깅 추가
#         print(f"k.shape after view: {k.shape}")
#         print(f"v.shape after view: {v.shape}")
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
#         if attn_mask is not None:
#             attn_scores += attn_mask

#         attn_probs = F.softmax(attn_scores, dim=-1)
#         attn_output = torch.matmul(attn_probs, v)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
#         attn_output = F.linear(attn_output, out_weight, self.out_proj.bias)

#         if need_weights:
#             return attn_output, attn_probs
#         else:
#             return attn_output
    
#     def __repr__(self):
#         s = {'{name} (embed_dim={embed_dim}, num_heads={num_heads})'}
#         return s.format(name=self.__class__.__name__, **self.__dict__)
    
#     def set_piggymask(self, masks):
#         """Set Piggyback Masks for Shared Attention Weights"""
#         self.piggymask_q = Parameter(masks['q_proj'], requires_grad=True)
#         self.piggymask_k = Parameter(masks['k_proj'], requires_grad=True)
#         self.piggymask_v = Parameter(masks['v_proj'], requires_grad=True)
#         self.piggymask_out = Parameter(masks['out_proj'], requires_grad=True)

#     def reinitialize_piggymask(self):
#         """Reinitialize Piggyback Masks for a new task."""
#         self.piggymask_q = Parameter(torch.ones_like(self.q_proj.weight) * self.mask_scale, requires_grad=True)
#         self.piggymask_k = Parameter(torch.ones_like(self.k_proj.weight) * self.mask_scale, requires_grad=True)
#         self.piggymask_v = Parameter(torch.ones_like(self.v_proj.weight) * self.mask_scale, requires_grad=True)
#         self.piggymask_out = Parameter(torch.ones_like(self.out_proj.weight) * self.mask_scale, requires_grad=True)


# def exists(val):
#     return val is not None

# def default(val, d):
#     return val if exists(val) else d

# --- https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py ---
class Attention(nn.Module):
    def __init__(self, embed_dim, context_dim=None, num_heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = context_dim if context_dim is not None else embed_dim  # 기본값 설정
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        
        # Query, Key-Value 생성
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, embed_dim)

    def forward(self, query, context=None, mask=None):
        h = self.num_heads
        context = context if context is not None else query  # Cross-Attention이 아니면 Self-Attention

        # Query, Key, Value 생성
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Multi-head 차원 변환 (b: batch, n: seq_len, h: num_heads, d: head_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Attention Score 계산 (Scaled Dot-Product Attention)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # Masking 처리
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # Softmax 및 Dropout 적용
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # Attention 적용 후 원래 차원으로 변환
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    