import math
from Config import LLMConfig
from typing import List,Tuple,Optional  #定义数据类别时使用的类型提示
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # 取一些激活函数时需要用到
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast # 用于存储模型的输出，包括logits和past_key_values，方便做内存管理

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        # 初始化RMSNorm层
        super().__init__()
        self.eps = eps  # 设置一个小的常数，防止除零错误
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化可学习的权重参数

    def forward(self, x):
        # 前向传播函数，计算RMSNorm
        # 计算每个输入x的均方根（RMS）并进行归一化
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x)
        # 因为x是BF16格式，所以需要将其转换为float格式，以便进行计算，防止平方后溢出
        # rsqrt是求平方根的倒数，pow是求平方，mean是求均值，keepdim=True表示保持维度不变
        # mean(dim=-1, keepdim=True)表示对最后一个维度求均值，保持维度不变
        # .type_as(x)表示将结果转换为x的数据类型

def precompute_pos_cis(dim: int, theta: float, end: int = int(32 * 1024)):
    # 预计算位置编码（pos_cis）
    # dim：位置编码的维度
    # theta：用于计算频率的参数
    # end：位置编码的长度
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))  # 计算频率
    # **表示幂运算，torch.arange(0, dim, 2)表示生成一个从0到dim-1的等差数列，步长为2
    # [:(dim // 2)]表示取前dim // 2个元素

    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(0, end)  # 生成时间序列t

    # freqs.shape = [seq_len, dim // 2]
    # 计算m * \theta
    freqs = torch.outer(t, freqs).float()  # 计算外积以得到频率矩阵

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 pos_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 计算复数的极坐标表示（位置编码）
    # torch.polar是将复数转换为极坐标形式，返回的是极坐标的模和幅角
    # torch.ones_like(freqs)表示生成一个与freqs相同形状的全1张量
    # torch.polar(torch.ones_like(freqs), freqs)表示将全1张量转换为极坐标形式
    return pos_cis  # 返回预计算的位置编码

def apply_rotary_emb(xq, xk, pos_cis):
    # 应用旋转位置编码
    def unite_shape(pos_cis, x):
        # 统一pos_cis和x的形状，使它们能够相乘
        ndim = x.ndim  # 获取x的维度
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])  # 检查位置编码的形状是否与x形状一致
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # 扩展x的维度
        return pos_cis.view(*shape)  # 将位置编码扩展为与x相同的形状

    # xq和xk的形状为[bs, seq_len, n_kv_heads, head_dim]
    # 将xq和xk转换为复数形式，以便与位置编码相乘
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # torch.view_as_complex是将实数张量转换为复数张量，reshape(*xq.shape[:-1], -1, 2)表示将xq的最后两维度转换为2维度

    pos_cis = unite_shape(pos_cis, xq_)  # 统一位置编码和xq的形状
    # 将xq和xk与位置编码相乘，并转换回实数形式
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    # torch.view_as_real是将复数张量转换为实数张量，flatten(3)表示将张量的最后一维度展平

    return xq_out.type_as(xq), xk_out.type_as(xk)  # 返回旋转位置编码后的xq和xk

def repeat_kv(x: torch.Tensor, n_rep: int):
    # 将x的键值对（key/value）扩展n_rep倍
    # x的形状为[bs, seq_len, n_kv_heads, head_dim]

    bs, seq_len, n_kv_heads, head_dim = x.shape  # 获取输入张量x的形状
    if n_rep == 1:
        return x  # 如果n_rep为1，则返回原始的x
    # 扩展键值对，将n_rep重复并改变形状
    return (
        x[:, :, :, None, :].expand(bs, seq_len, n_kv_heads, n_rep, head_dim).reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
        # expand是张量的扩展操作，expand(bs, seq_len, n_kv_heads, n_rep, head_dim)表示将x的第3维度扩展n_rep倍
        # x[:, :, :, None, :]表示在第3维度后面添加一个维度，然后进行扩展
        # reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)表示将扩展后的张量进行形状重塑
        # 与直接repeat不同，expand是对张量进行扩展，repeat是对张量的元素进行复制，例如[1, 2, 3] repeat(2)=[1, 2, 3, 1, 2, 3]
        # 而先expend再reshape是对张量的维度进行扩展，例如[1, 2, 3] reshape(2)=[1, 1, 2, 2, 3, 3]
    )


class Attention(nn.Module):
    def __init__(self, args: LLMConfig):
        # 构造函数，初始化注意力层
        super().__init__()
        
        # 如果没有指定n_kv_heads，则使用nhead的值
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0  # 确保n_heads可以被n_kv_heads整除
        self.n_heads = args.n_heads  # 获取总的注意力头数
        self.n_rep = self.n_heads // self.n_kv_heads  # 计算每个kv组的重复次数
        self.head_dim = args.dim // self.n_heads  # 计算每个注意力头的维度
        
        # 定义查询、键、值的线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # 查询向量的线性变换
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 键向量的线性变换
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 值向量的线性变换
        
        # 定义输出的线性变换
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        
        # 定义Dropout层，防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        
        # 定义mask矩阵，用于自注意力计算中的遮蔽操作
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
        # torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
        # 表示生成一个形状为(1, 1, args.max_seq_len, args.max_seq_len)的张量，值全为负无穷
        #  float('-inf')表示负无穷
        mask = torch.triu(mask, diagonal=1)  # 生成上三角矩阵
        # torch.triu(mask, diagonal=1)表示生成一个上三角矩阵，对角线为1 torch.triu是取上三角矩阵  diagonal=1表示对角线的偏移量

        self.register_buffer('mask', mask, persistent=False)
        # 注册为buffer，使其不会作为模型参数更新, 掩码是固定值，无需保存到模型参数中,禁用持久化。

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                # 构成一个元组，包含键和值的张量
                use_cache: bool = False):
        # 前向传播函数，计算注意力输出
        # x此时的形状为[bsz, seq_len, dim]
        
        bsz, seq_len, _ = x.shape  # 获取输入x的批次大小、序列长度和特征维度
        # 通过线性变换计算查询、键、值向量
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 将查询、键、值的维度重新调整为 (bsz, seq_len, n_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        # view是张量的重塑操作，view(1, -1)表示将张量重塑为1行，-1表示自动计算列数


        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 如果存在past_key_value（即缓存的键和值），则将其与当前的键和值拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼接过去的键 访问用下标0来访问构建的Tuple的键
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼接过去的值
            # past_key_value[0]表示过去的键，past_key_value[1]表示过去的值，就是将现在的词向量复制出来的key和value直接拼接到缓存中
            # torch.cat是将两个张量拼接在一起，dim=1表示在第1维度上进行拼接，即在seq_len维度上进行拼接
            # 例如[bsz, seq_len1, dim]和[bsz, seq_len2, dim]拼接后为[bsz, seq_len1+seq_len2, dim]
            # xk和xv的形状为[bsz, seq_len, n_kv_heads, head_dim]
        
        # 如果使用缓存，返回拼接后的past_key_value，否则返回None
        past_kv = (xk, xv) if use_cache else None
        
        # 转置查询向量xq并对键和值进行重复，形成多头注意力
        # xq转置前的形状为[bsz, seq_len, n_heads, head_dim]
        xq, xk, xv = (
            xq.transpose(1, 2),  # 转置查询向量
            # transpose(1, 2)表示将第1维和第2维进行转置，即将seq_len和n_heads进行转置
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # 重复键向量并转置
            repeat_kv(xv, self.n_rep).transpose(1, 2)   # 重复值向量并转置
        )
        # repeat_kv是将键值对进行扩展，将n_kv_heads重复n_rep次，然后转置
        # 转置后的形状为[bsz, n_heads, seq_len, head_dim]
        
        # 计算注意力得分，使用缩放点积计算
        scores = (xq @ xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # transpose(-2, -1)表示将倒数第2维和倒数第1维进行转置，即将seq_len和head_dim进行转置
        # xk转置后的形状为[bsz, n_heads, head_dim, seq_len]
        # 这样计算得到的scores的形状为[bsz, n_heads, seq_len, seq_len]

        # 加上mask矩阵，用于遮蔽未来的信息（自回归生成）
        scores += self.mask[:, :, :seq_len, :seq_len]
        # 对得分进行softmax归一化，得到注意力权重
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)
        scores = self.attn_dropout(scores)  # 应用dropout
        
        # 计算加权值输出
        output = scores @ xv

        # 转置输出并恢复原始维度
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 最后通过wo线性层并应用dropout
        output = self.resid_dropout(self.wo(output))
        
        return output, past_kv  # 返回注意力输出和past_key_value（用于缓存）


class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        # 构造函数，初始化前馈网络层
        super().__init__()
        
        # 如果配置中没有指定hidden_dim，则计算默认值
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim  # 默认hidden_dim为dim的4倍
            hidden_dim = int(2 * hidden_dim / 3)  # 对hidden_dim进行调整
            # 将hidden_dim调整为config.multiple_of的倍数
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        # 定义第一个全连接层，将输入维度从dim映射到hidden_dim
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        # 定义第二个全连接层，将hidden_dim映射回dim
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        # 定义第三个全连接层，用于对输入进行变换
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        # 定义Dropout层，防止过拟合
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 前向传播函数，处理输入x并返回输出
        # 先通过w1层处理，再应用Silu激活函数，接着与w3(x)的输出相乘，最后通过w2层和dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))



class SpongeBobBlock(nn.Module):
    def __init__(self, layer_id: int, config: LLMConfig):
        # 构造函数，初始化一个SpongeBobBlock层
        super().__init__()
        self.n_heads = config.n_heads  # 获取注意力头的数量
        self.dim = config.dim  # 获取模型的维度
        self.head_dim = self.dim // self.n_heads  # 每个注意力头的维度
        self.attention = Attention(config)  # 定义注意力层
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)  # 定义注意力层的归一化层
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)  # 定义前馈网络的归一化层
        self.feed_forward = FeedForward(config)  # 定义前馈网络层

    def forward(self,
                x,
                pos_cis,
                past_key_value=None,
                use_cache=False):
        # 前向传播函数，处理输入并返回输出

        # 首先通过注意力层进行处理，注意力层包括归一化、计算注意力以及保存past_key_value
        h_attn, past_kv = self.attention(
            self.attention_norm(x),  # 对输入x进行注意力归一化
            pos_cis,  # 位置编码
            past_key_value=past_key_value,  # 上一轮的key_value
            use_cache=use_cache  # 是否使用缓存
        )

        # 通过残差连接将输入x和注意力输出h_attn相加
        h = x + h_attn

        # 通过前馈网络进行处理，先进行归一化，再通过feed_forward处理，并加到h上
        out = h + self.feed_forward(self.ffn_norm(h))

        # 返回输出和当前层的past_key_value
        return out, past_kv




class SpongeBob(PreTrainedModel):
    config_class = LLMConfig  # 定义配置类，SpongeBob模型的配置类是LLMConfig

    def __init__(self, params: LLMConfig = None):
        # 构造函数，初始化模型参数，若未传入params，则使用LLMConfig的默认值
        self.params = params or LLMConfig()  # 使用传入的参数或默认配置
        super().__init__(params)  # 调用父类的构造函数，传入配置参数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers  # 获取词汇表大小和层数
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.params.dim)  # 定义token嵌入层
        self.dropout = nn.Dropout(self.params.dropout)  # 定义Dropout层
        self.layers = nn.ModuleList([SpongeBobBlock(l, params) for l in range(self.n_layers)])  # 定义多个SpongeBobBlock层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 定义RMSNorm层
        self.output = nn.Linear(params.dim, self.vocab_size, bias=False)  # 定义输出线性层
        self.tok_embeddings.weight = self.output.weight  # 将token嵌入层的权重与输出层共享
        # 为什么能共享权重？因为token嵌入层和输出层的权重矩阵是一样的 为什么要共享权重？因为这样可以减少参数量
        # 预计算RoPE位置编码相关的值
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=self.params.dim // params.n_heads, theta=params.rope_theta
                                                ), persistent=False)
        self.OUT = CausalLMOutputWithPast()  # 用于存储模型的输出，包括logits和past_key_values
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                # input_ids: Optional[torch.Tensor] = None, 表示input_ids是一个可选的tensor，如果没有传入则为None
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args # **args表示 args是一个字典，包含了所有未命名的参数
                ):
        # 前向传播函数，处理输入并返回输出
        past_key_values = past_key_values or [None] * (self.n_layers)  # 若没有传入past_key_values，初始化为空
        start_pos = args.get('start_pos', 0)  # 获取起始位置（默认为0） 用于推理模式从哪里开始算
        
        # 获取输入的token嵌入表示，并应用Dropout
        h = self.dropout(self.tok_embeddings(input_ids))
        # 获取位置编码（根据start_pos获取对应的片段）
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []  # 用于存储每层的past_key_values

        # 遍历每一层进行前向传播
        for l, layer in enumerate(self.layers):
            # 通过当前层计算新的hidden状态和past_key_value
            h, past_kv = layer(h, pos_cis, past_key_value=past_key_values[l], use_cache=use_cache)
            past_kvs.append(past_kv)  # 将每一层的past_key_value添加到past_kvs列表中
        
        # 通过输出层得到logits 对最后的hidden状态进行归一化，并通过输出层得到logits logits是模型的输出
        logits = self.output(self.norm(h))  # 对最后的hidden状态进行归一化，并通过输出层得到logits
        self.OUT.__setitem__('logits', logits)  # 将logits放入输出字典
        # 为什么使用__setitem__方法将logits放入输出字典？因为CausalLMOutputWithPast是一个字典类，要通过__setitem__方法将logits放入其中
        self.OUT.__setitem__('past_key_values', past_kvs)  # 将past_key_values放入输出字典
        return self.OUT  # 返回包含logits和past_key_values的输出
    


    """推理部分的代码"""

    @torch.inference_mode()  # 表示该函数不会参与梯度计算（加速推理过程）
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
               stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
     # 生成文本的主函数
     # 参数说明：
     # input_ids: 输入的token id（形状为 [batch_size, seq_len]）
     # eos_token_id: 结束符的token id（默认为2）
     # max_new_tokens: 生成的最大token数量（默认为1024）
     # temperature: 控制采样多样性的温度值（默认为0.75）
     # top_p: 用于控制nucleus采样的阈值（默认为0.90）
     # stream: 是否进行流式生成（默认为False）
     # rp: 重复惩罚因子（默认为1.0）
     # use_cache: 是否使用缓存（默认为True）
     # pad_token_id: padding token的id（默认为0）
          return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
     # 流式生成的实现方法
     # 生成过程会分步进行，每次生成一个token，直到达到最大长度或遇到eos_token
     
          start, first_seq, past_kvs = input_ids.shape[1], True, None
          # 初始化序列的开始位置，first_seq标记是否为第一次生成，past_kvs存储缓存
          while input_ids.shape[1] < max_new_tokens - 1:
               # 循环直到生成足够数量的token
               if first_seq or not use_cache:
                    # 如果是第一次生成或者不使用缓存，则从头开始生成
                    out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
               else:
                    # 否则，只使用最后一个token进行生成
                    out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                              start_pos=input_ids.shape[1] - 1, **args)
               
               # 获取logits和past_key_values
               logits, past_kvs = out.logits[:, -1, :], out.past_key_values
               
               # 对已生成的tokens进行重复惩罚
               logits[:, list(set(input_ids.tolist()[0]))] /= rp
               # 对logits进行温度调整
               logits /= (temperature + 1e-9)
               
               if top_p is not None and top_p < 1.0:
                    # 如果top_p值不为None且小于1，则应用nucleus采样
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # 选择累积概率超过top_p的token进行过滤
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')  # 过滤掉不符合条件的token
               
               # 从logits中根据概率分布采样下一个token
               input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
               
               # 将新生成的token添加到input_ids中
               input_ids = torch.cat((input_ids, input_ids_next), dim=1)
               
               # 输出当前生成的token
               yield input_ids[:, start:]
               
               # 如果生成的token为eos_token，则停止生成
               if input_ids_next.item() == eos_token_id:
                    break
