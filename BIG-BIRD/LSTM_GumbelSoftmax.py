#!/usr/bin/env python
# coding: utf-8

# In[3]:


# The Transformer from ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) has been on a lot of people's minds over the last year. Besides producing major improvements in translation quality, it provides a new architecture for many other NLP tasks. The paper itself is very clearly written, but the conventional wisdom has been that it is quite difficult to implement correctly. 
# 
# In this post I present an "annotated" version of the paper in the form of a line-by-line implementation. I have reordered and deleted some sections from the original paper and added comments throughout. This document itself is a working notebook, and should be a completely usable implementation. In total there are 400 lines of library code which can process 27,000 tokens per second on 4 GPUs. 
# 
# To follow along you will first need to install [PyTorch](http://pytorch.org/). The complete notebook is also available on [github](https://github.com/harvardnlp/annotated-transformer) or on Google [Colab](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sharing). 
# 
# Note this is merely a starting point for researchers and interested developers. The code here is based heavily on our [OpenNMT](http://opennmt.net) packages. (If helpful feel free to [cite](#conclusion).) For other full-sevice implementations of the model check-out [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) (tensorflow) and [Sockeye](https://github.com/awslabs/sockeye) (mxnet).
# 
# - Alexander Rush ([@harvardnlp](https://twitter.com/harvardnlp) or srush@seas.harvard.edu)
# 

# # Prelims

# In[1]:


# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 


# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import torch.autograd as autograd
import wandb
# Table of Contents
# 
# 
# * Table of Contents                               
# {:toc}      

# > My comments are blockquoted. The main text is all from the paper itself.

# # Background

# The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention.
# 
# Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations. End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and
# language modeling tasks.
# 
# To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution. 

# # Model Architecture

# Most competitive neural sequence transduction models have an encoder-decoder structure [(cite)](https://arxiv.org/abs/1409.0473). Here, the encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$. Given $\mathbf{z}$, the decoder then generates an output sequence $(y_1,...,y_m)$ of symbols one element at a time. At each step the model is auto-regressive [(cite)](https://arxiv.org/abs/1308.0850), consuming the previously generated symbols as additional input when generating the next. 

# In[3]:


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# In[4]:


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, device):
        super(Generator, self).__init__()
        self.device = device
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x, mode = 'log_softmax', temp = 0.8):
        if(mode == 'gumbel_softmax'):
            return self.gumbel_softmax(self.proj(x),temp)
        elif(mode == 'log_softmax'):
            return F.log_softmax(self.proj(x), dim=-1)
        else:
            print("Generator mode undefined")
            exit(1);
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        values, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(logits.shape[0], -1), ind, values, y

class CriticNet(nn.Module):
    def __init__(self, d_model):
        super(CriticNet, self).__init__()
        self.proj = nn.Sequential(
                        nn.Linear(d_model, 2 * d_model),
                        nn.ReLU(),
                        nn.Linear(2 * d_model, 1),
                    )

    def forward(self, x):
        return self.proj(x).squeeze(1)   
# The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. 

# In[4]:


# ## Encoder and Decoder Stacks   
# 
# ### Encoder
# 
# The encoder is composed of a stack of $N=6$ identical layers. 

# In[5]:


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[6]:


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# We employ a residual connection [(cite)](https://arxiv.org/abs/1512.03385) around each of the two sub-layers, followed by layer normalization [(cite)](https://arxiv.org/abs/1607.06450).  

# In[7]:


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the output of each sub-layer, before it is added to the sub-layer input and normalized.  
# 
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.  

# In[8]:


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

# In[9]:


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ### Decoder
# 
# The decoder is also composed of a stack of $N=6$ identical layers.  
# 

# In[10]:


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

# In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.  Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.  

# In[11]:


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.  This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

# In[12]:


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# > Below the attention mask shows the position each tgt word (row) is allowed to look at (column). Words are blocked for attending to future words during training.

# In[13]:


# ### Attention                                                                                                                                                                                                                                                                             
# An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.                                                                                                                                                                                                                                                                                   
# 
# We call our particular attention "Scaled Dot-Product Attention".   The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.  We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.                                                                                                                                                                                                                                  
#                                                                                                                                                                      

# In[8]:


# 
# In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$.   The keys and values are also packed together into matrices $K$ and $V$.  We compute the matrix of outputs as:                      
#                                                                  
# $$                                                                         
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V               
# $$   

# In[14]:


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))              / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# The two most commonly used attention functions are additive attention [(cite)](https://arxiv.org/abs/1409.0473), and dot-product (multiplicative) attention.  Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.  While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.                                                                                             
# 
#                                                                         
# While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$ [(cite)](https://arxiv.org/abs/1703.03906). We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients  (To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$.  Then their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance $d_k$.). To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.          

#  

# In[6]:


# Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.                                            
# $$    
# \mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O    \\                                           
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)                                
# $$                                                                                                                 
# 
# Where the projections are parameter matrices $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.                                                                                                                                                                                             In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_k=d_v=d_{\text{model}}/h=64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality. 

# In[15]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# ### Applications of Attention in our Model                                                                                                                                                      
# The Transformer uses multi-head attention in three different ways:                                                        
# 1) In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.   This allows every position in the decoder to attend over all positions in the input sequence.  This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [(cite)](https://arxiv.org/abs/1609.08144).    
# 
# 
# 2) The encoder contains self-attention layers.  In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.   Each position in the encoder can attend to all positions in the previous layer of the encoder.                                                   
# 
# 
# 3) Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.  We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.  We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections.                                                                                                                                                                                                                                                      

# ## Position-wise Feed-Forward Networks                                                                                                                                                                                                                                                                                                                                                             
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
# In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.  This consists of two linear transformations with a ReLU activation in between.
# 
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$                                                                                                                                                                                                                                                         
#                                                                                                                                                                                                                                                         
# While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.  The dimensionality of input and output is $d_{\text{model}}=512$, and the inner-layer has dimensionality $d_{ff}=2048$. 

# In[16]:


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ## Embeddings and Softmax                                                                                                                                                                                                                                                                                           
# Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$.  We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.  In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [(cite)](https://arxiv.org/abs/1608.05859). In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$.                                                                                                                                 

# In[17]:


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ## Positional Encoding                                                                                                                             
# Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.  To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.  The positional encodings have the same dimension $d_{\text{model}}$ as the embeddings, so that the two can be summed.   There are many choices of positional encodings, learned and fixed [(cite)](https://arxiv.org/pdf/1705.03122.pdf). 
# 
# In this work, we use sine and cosine functions of different frequencies:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
# $$PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}})$$
# 
# $$PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})$$                                                                                                                                                                                                                                                        
# where $pos$ is the position and $i$ is the dimension.  That is, each dimension of the positional encoding corresponds to a sinusoid.  The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.  We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$. 
# 
# In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.  For the base model, we use a rate of $P_{drop}=0.1$. 
#                                                                                                                                                                                                                                                     
# 

# In[18]:


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2., dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

    
# > Below the positional encoding will add in a sine wave based on position. The frequency and offset of the wave is different for each dimension. 

# In[19]:



# We also experimented with using learned positional embeddings [(cite)](https://arxiv.org/pdf/1705.03122.pdf) instead, and found that the two versions produced nearly identical results.  We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.    

# ## Full Model
# 
# > Here we define a function from hyperparameters to a full model. 

# In[20]:


def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1, emb_share=False):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    src_emb = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_emb = src_emb if emb_share else nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        src_emb,
        tgt_emb,
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# In[21]:



# # Training
# 
# This section describes the training regime for our models.

# > We stop for a quick interlude to introduce some of the tools 
# needed to train a standard encoder decoder model. First we define a batch object that holds the src and target sentences for training, as well as constructing the masks. 

# ## Batches and Masking

# In[22]:


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum().item() # this was a tensor
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# > Next we create a generic training and scoring function to keep track of loss. We pass in a generic loss compute function that also handles parameter updates. 

# ## Training Loop

# In[23]:


# def run_epoch(data_iter, model, loss_compute):
#     "Standard Training and Logging Function"
#     start = time.time()
#     total_tokens = 0
#     total_loss = 0
#     tokens = 0
#     for i, batch in enumerate(data_iter):
#         out = model.forward(batch.src, batch.trg, 
#                             batch.src_mask, batch.trg_mask)
#         loss = loss_compute(out, batch.trg_y, batch.ntokens)
#         total_loss += loss
#         total_tokens += batch.ntokens
#         tokens += batch.ntokens
#         if i % 50 == 1:
#             elapsed = time.time() - start
#             print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
#                     (i, loss / batch.ntokens, tokens / elapsed))
#             start = time.time()
#             tokens = 0
#     return total_loss / total_tokens



class Stats:
    "Log stats for a run."
    
    def __init__(self, writer=None):
        self.tokens = 0
        self.loss = 0
        self.steps = 0
        self.start = time.time()
        self.writer = writer
        self.step_loss = 0
        self.step_tokens = 0
        self.last_time = self.start
        self.history = []
        
    def update(self, loss, tokens, log=False):
        self.loss += loss
        self.tokens += tokens
        self.steps += 1
        tps = tokens / (time.time () - self.last_time)
        step_avg_loss =  loss / tokens
        if self.writer is not None:
            self.writer.add_scalar("data/loss", step_avg_loss)
            self.writer.add_scalar("data/tokens_per_sec", tps)
        
        self.history.append(float(step_avg_loss))
        
        if log:
            print(
                "Step: %d Loss: %f Tokens / Sec: %f"
                % (self.steps, step_avg_loss, tps), end='\r'
            )
        self.last_time = time.time()


# ## Training Data and Batching
# 
# We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.  Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary.
# 
# 
# Sentence pairs were batched together by approximate sequence length.  Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.     

# > We will use torch text for batching. This is discussed in more detail below. Here we create batches in a torchtext function that ensures our batch size padded to the maximum batchsize does not surpass a threshold (25000 if we have 8 gpus).

# In[24]:


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# ## Hardware and Schedule                                                                                                                                                                                                   
# We trained our models on one machine with 8 NVIDIA P100 GPUs.  For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.  We trained the base models for a total of 100,000 steps or 12 hours. For our big models, step time was 1.0 seconds.  The big models were trained for 300,000 steps (3.5 days).

# ## Optimizer
# 
# We used the Adam optimizer [(cite)](https://arxiv.org/abs/1412.6980) with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We varied the learning rate over the course of training, according to the formula:                                                                                            
# $$                                                                                                                                                                                                                                                                                         
# lrate = d_{\text{model}}^{-0.5} \cdot                                                                                                                                                                                                                                                                                                
#   \min({step\_num}^{-0.5},                                                                                                                                                                                                                                                                                                  
#     {step\_num} \cdot {warmup\_steps}^{-1.5})                                                                                                                                                                                                                                                                               
# $$                                                                                                                                                                                             
# This corresponds to increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.  We used $warmup\_steps=4000$.                            

# > Note: This part is very important. Need to train with this setup of the model. 

# In[25]:



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor *             (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# 
# > Example of the curves of this model for different model sizes and for optimization hyperparameters. 

# In[26]:



# ## Regularization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#                                                                                                                                                                                                                                                                                                                       
# ### Label Smoothing
# 
# During training, we employed label smoothing of value $\epsilon_{ls}=0.1$ [(cite)](https://arxiv.org/abs/1512.00567).  This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.  

# > We implement label smoothing using the KL div loss. Instead of using a one-hot target distribution, we create a distribution that has `confidence` of the correct word and the rest of the `smoothing` mass distributed throughout the vocabulary.

# In[27]:


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# > Here we can see an example of how the mass is distributed to the words based on confidence. 

# In[28]:




# > Label smoothing actually starts to penalize the model if it gets very confident about a given choice. 

# In[29]:



# # A First  Example
# 
# > We can begin by trying out a simple copy-task. Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols. 

# ## Synthetic Data

# In[30]:


# ## Loss Computation

# In[31]:


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm


# ## Greedy Decoding

# In[32]:



# > This code predicts a translation using greedy decoding for simplicity. 

# In[33]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

#----------------------------------------
class BigBird():
    #generator is translator here
    def __init__(self, generator, discriminator, reconstructor, dictionary, gamma = 0.99, clip_value = 0.1, lr_G = 5e-5, lr_D = 5e-5, lr_R = 1e-4, LAMBDA = 10,  TEMP_START = 4.0, TEMP_END = 0.01, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(BigBird, self).__init__()
        
        self.device = device
        
        self.dictionary = dictionary
        
        self.generator = generator.to(self.device)
        self.reconstructor = reconstructor.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()
        
        self.optimizer_R = torch.optim.Adam(list(self.generator.parameters()) + list(self.reconstructor.parameters()), lr=lr_R)
        
        #normal WGAN
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr_G)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr_D)
        
        #WGAN GP
        
        #self.LAMBDA = LAMBDA # Gradient penalty lambda hyperparameter
        #self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G,  betas=(0.0, 0.9))
        #self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D,  betas=(0.0, 0.9))
               
        self.clip_value = clip_value
        self.TEMP_START = TEMP_START
        self.TEMP_END = TEMP_END
        self.TEMP_DECAY = 1000
        
        self.total_steps = 0
        self.RL_loss = []
        self.batch_G_losses = []
        self.batch_D_losses=[] 
        self.real_scores = []
        self.fake_scores = []
        self.all_rewards = []
        
        self.epoch = 0
        
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        #print real_data.size()
        BATCH_SIZE = real_data.shape[0]
        dim_1 = real_data.shape[1]
        dim_2 = real_data.shape[2]
        alpha = torch.rand(BATCH_SIZE, dim_1)
        alpha = alpha.view(-1,1).expand(dim_1 * BATCH_SIZE, dim_2).view(BATCH_SIZE, dim_1, dim_2)
        alpha = alpha.to(self.device)
        
        #print(real_data.shape) #[BATCH_SIZE, 19, vocab_sz]
        #print(fake_data.shape) #[BATCH_SIZE, 19, vocab_sz]
        interpolates_data = ( alpha * real_data.float() + ((1 - alpha) * fake_data.float()) )

        interpolates = interpolates_data.to(self.device)
        
        #interpolates = netD.disguised_embed(interpolates_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        
        src_mask = (interpolates_data.argmax(-1) != netD.padding_index).type_as(interpolates_data).unsqueeze(-2)
        disc_interpolates = netD.transformer_encoder( interpolates, src_mask )

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty
    
    def _to_one_hot(self, y, n_dims):

        scatter_dim = len(y.size())

        y_tensor = y.to(self.device).long().view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).to(self.device)

        return zeros.scatter(scatter_dim, y_tensor, 1) 
    
    def train_D(self, fake_datas, real_datas):
        ## train discriminator       
#         print("real")
#         print(real_datas[:10])
        real_score = torch.mean(self.discriminator(real_datas))
#         print("fake")
#         print(fake_datas[:10])
        fake_score = torch.mean(self.discriminator(fake_datas))

        batch_d_loss = -real_score + fake_score #+ self.calc_gradient_penalty(self.discriminator, real_datas, fake_datas)
        
        return batch_d_loss, real_score.item(), fake_score.item()
    
    def train_G(self, fake_datas): 

        self.optimizer_G.zero_grad()
          
        batch_g_loss = -torch.mean(self.discriminator(fake_datas))
        
        batch_g_loss.backward(retain_graph=True)
        self.optimizer_G.step()

        return batch_g_loss.item()      
            
#     def compute(self, batch_rewards, saved_log_probs, critic_values):
#         # TODO:
#         # discount your saved reward
        
#         # TODO:
#         cummulative_loss = 0
#         # compute loss
#         #print(batch_rewards.shape)
#         #print(critic_values.shape)
#         for i, reward in enumerate(batch_rewards):
#             R = reward
#             loss = []
#             returns = []
            
#             for r in range(saved_log_probs.shape[1], 0, -1):
#                 returns.insert(0, R)
#                 R = self.gamma * R

#             returns = torch.stack(returns, 0)
#             advantages = returns - critic_values[i]
#             #returns = torch.where(returns > 0 , returns, torch.FloatTensor([0.1] * len(returns)));
#             action_gain = (saved_log_probs[i] * advantages.detach()).sum()
#             value_loss =  F.smooth_l1_loss(returns, critic_values[i])
#             loss = value_loss - action_gain
#             cummulative_loss += loss.mean()
         
#         return cummulative_loss/batch_rewards.shape[0]
    
    def indicies2string(self, indices):
        inv_map = {v: k for k, v in self.dictionary.items()}
        return ' '.join([inv_map[i.item()] for i in indices])
        
    def train(self):
        self.generator.train()
        self.reconstructor.train()
        self.discriminator.train()
        
    def eval(self):
        self.generator.eval()
        self.reconstructor.eval()
        self.discriminator.eval()
    
    def load(self, load_path):
        print('load Bird from', load_path)
        loader = torch.load(load_path)
        self.generator.load_state_dict(loader['generator'])
        self.discriminator.load_state_dict(loader['discriminator'])
        self.reconstructor.load_state_dict(loader['reconstructor'])
        self.batch_G_losses = loader['batch_G_losses']
        self.batch_D_losses = loader['batch_D_losses']
        self.real_scores = loader['real_scores']
        self.fake_scores = loader['fake_scores']
        self.total_steps = loader['total_steps']
        self.epoch = loader['epoch']
        self.gumbel_temperature = loader['gumbel_temperature']
    
    def save(self, save_path):
        print('lay egg to ./Nest ... save as', save_path)

        torch.save({'generator':self.generator.state_dict(), 
                    'reconstructor':self.reconstructor.state_dict(), 
                    'discriminator':self.discriminator.state_dict(),
                    'batch_G_losses':self.batch_G_losses,
                    'batch_D_losses':self.batch_D_losses,
                    'real_scores':self.real_scores,
                    'fake_scores':self.fake_scores,
                    'total_steps':self.total_steps,
                    'epoch':self.epoch,
                    'gumbel_temperature':self.gumbel_temperature
                    },save_path)
    
    def eval_iter(self, src, src_mask, max_len, real_data, ct, verbose = 1):
        with torch.no_grad():
            summary_sample, summary_log_values, critic_values, summary_log_probs, gumbel_one_hot = self.generator(src, src_mask, max_len, self.dictionary['[CLS]'], mode = 'sample')
            summary_mask = (summary_sample != self.dictionary['[SEP]']).type_as(summary_sample).unsqueeze(-2)
            rewards, acc, out = self.reconstructor(gumbel_one_hot, summary_mask, src.shape[1], self.dictionary['[CLS]'], src)
            if verbose == 1 and ct % 100 == 0:
                print("origin:")
                self.indicies2string(src[0])
                print("summary:")
                self.indicies2string(summary_sample[0])
                print("real summary:")
                self.indicies2string(real_data[0])
                print("reconsturct out:")
                self.indicies2string(out[0])

    #             print("sentiment:",label[0].item())
    #             print("y:",sentiment_label[0].item())
    #            print("reward:",rewards[0].item())
                print("")
            return acc, rewards.mean().item()
    
    def plot_estimated_probs(samples,ylabel=''):
        n_cats = np.max(samples)+1
        estd_probs,_,_ = plt.hist(samples,bins=np.arange(n_cats+1),align='left',edgecolor='white')
        plt.xlabel('Category')
        plt.ylabel(ylabel+'Estimated probability')
        return estd_probs
    def run_iter(self, src, src_mask, max_len, real_data,  D_iters = 5, D_toggle = 'On', verbose = 1):
        #summary_logits have some problem

        
        #summary = self.generator(src, src_mask, max_len, self.dictionary['[CLS]'])
        summary_sample, summary_log_values, critic_values, summary_probs, gumbel_one_hot = self.generator(src, src_mask, max_len, self.dictionary['[CLS]'])
        
        batch_D_loss = 0
        if(D_toggle == 'On'):
            for i in range(D_iters):
                self.optimizer_D.zero_grad()
                
                batch_d_loss, real_score, fake_score = self.train_D(gumbel_one_hot, self._to_one_hot(real_data, len(self.dictionary)))
                batch_D_loss += batch_d_loss
                batch_d_loss.backward(retain_graph=True);

                #Clip critic weights
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)
        
            self.optimizer_D.step();
        
        batch_D_loss = batch_D_loss.item()/D_iters
        
        batch_G_loss = 0 
        if(D_toggle == 'On'):
            #print(gumbel_one_hot.shape)
            batch_G_loss = self.train_G(gumbel_one_hot)          
        
        self.gumbel_temperature = self.TEMP_END + (self.TEMP_START - self.TEMP_END) * math.exp(-1. * self.total_steps / self.TEMP_DECAY)
        
        
        summary_mask = (summary_sample != self.dictionary['[SEP]']).type_as(summary_sample).unsqueeze(-2)
        rec_loss, acc, out = self.reconstructor(gumbel_one_hot, summary_mask, src.shape[1], self.dictionary['[CLS]'], src, mode='sample', temp = self.gumbel_temperature)
        

        
        self.optimizer_R.zero_grad()
        #RL_loss = self.RL_scale * self.compute(rewards, summary_log_values, critic_values)
        #RL_loss.backward()
        rec_loss.backward()
        #nn.utils.clip_grad_norm_(list(self.generator.parameters()) + list(self.reconstructor.parameters()), 0.5)
        self.optimizer_R.step()
        
        """
        #assert type(summary_logits) is not list
        rewards, acc, label = self.classifier(summary_sample, sentiment_label, self.dictionary['[SEP]'])      
        self.optimizer_C.zero_grad()
        RL_loss = self.RL_scale * self.compute(rewards, summary_log_values, critic_values)
        RL_loss.backward()
        nn.utils.clip_grad_norm_(list(self.generator.parameters()) + list(self.classifier.parameters()), 0.5)
        self.optimizer_C.step()
        """
        
        self.batch_G_losses.append(batch_G_loss)
        self.batch_D_losses.append(batch_D_loss)
        self.real_scores.append(real_score)
        self.fake_scores.append(fake_score)
        
        self.total_steps += 1
        
        if self.total_steps % 500 == 0:
            if not os.path.exists("./Nest"):
                os.makedirs("./Nest")
            self.save("./Nest/NewbornBird_LSTM_GumbelSoftmax")

            #for i in range(5):
                #plt.plot(range(1000),summary_probs.cpu().detach().numpy()[0,i,:1000] )
            #    wandb.log({"prob {}".format(i): wandb.Histogram(summary_probs.cpu().detach().numpy()[0,i,:1000])},step=step)
        
        if verbose == 1 and self.total_steps % 100 == 0:
            print("origin:")
            print(self.indicies2string(src[0]))
            print("summary:")
            print(self.indicies2string(summary_sample[0]))
            print("real summary:")
            print(self.indicies2string(real_data[0]))
            print("reconsturct out:")
            print(self.indicies2string(out[0]))
#             print("sentiment:",label[0].item())
#             print("y:",sentiment_label[0].item())
#            print("reward:",rewards[0].item())
            
            print("")
        
        #for name, param in self.classifier.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), self.total_steps)
        distrib = summary_probs.cpu().detach().numpy()[0,0,:200]
        one_hot_out = gumbel_one_hot.cpu().detach().numpy()[0,0,:200]
        return [rec_loss.item(), batch_G_loss, batch_D_loss], [real_score, fake_score, acc], [self.indicies2string(src[0]), self.indicies2string(summary_sample[0]), self.indicies2string(out[0])], distrib, one_hot_out

class LSTMEncoder(nn.Module):    
    def __init__(self, vocab_sz, hidden_dim, padding_index):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_sz, hidden_dim)
        self.rnn_cell = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.padding_index = padding_index
        self.outsize = hidden_dim*2
        
    def forward(self, x):
        #src_mask = (x != self.padding_index).type_as(x).unsqueeze(-2)
        out, (h,c) = self.rnn_cell( self.src_embed(x))
        return out


    

class LSTM_Gumbel_Encoder_Decoder(nn.Module):
    def __init__(self, hidden_dim, emb_dim, input_len, output_len, voc_size, critic_net, device, eps=1e-8, num_layers = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        #self.input_len = input_len
        #self.output_len = output_len
        #self.voc_size = voc_size
        #self.teacher_prob = 1.
        #self.epsilon = eps
        
        self.emb_layer = nn.Embedding(voc_size, emb_dim)
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers=num_layers, batch_first=True)
        
        self.critic_net = critic_net
        
        self.device = device
        
        self.attention_softmax = nn.Softmax(dim=1)
        
#         self.pro_layer = nn.Sequential(
#             nn.Linear(hidden_dim*4, voc_size, bias=True)
#         )
        self.adaptive_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(hidden_dim*4, voc_size, [100, 1000, 10000], div_value=4.0, head_bias=False)

        
    def forward(self, x, src_mask, max_len, start_symbol, mode = 'argmax', temp = 2.0):
        
        batch_size = x.shape[0]
        input_len = x.shape[1]
        device = x.device
        
        # encoder
        x_emb = self.emb_layer(x)
        memory, (h, c) = self.encoder(x_emb)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()
        h = h.view(batch_size, self.num_layers, h.shape[-1]*2)
        c = c.view(batch_size, self.num_layers, c.shape[-1]*2)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()        

        
        ## decoder
        out_h, out_c = (h, c)
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(x.data)
        
        values = []
        critics = []
        all_probs = []
        gumbel_one_hots = []
        
        for i in range(max_len-1):
            ans_emb = self.emb_layer(ys[:,-1]).view(batch_size, 1, self.emb_dim)
            out, (out_h, out_c) = self.decoder(ans_emb, (out_h, out_c))
            critic = self.critic_net(out) 
            
            attention = torch.bmm(memory, out.transpose(1, 2)).view(batch_size, input_len)
            attention = self.attention_softmax(attention)            
            
            context_vector = torch.bmm(attention.view(batch_size, 1, input_len), memory)
            
            logits = torch.cat((out, context_vector), -1).view(batch_size, -1)
            
            one_hot, next_words, value, prob = self.gumbel_softmax(logits, temp)
            
#             print(feature.shape)
#             print(one_hot.shape)
#             print(next_words.shape)
#             print(values.shape)
#             print(log_probs.shape)
#             input("")
                
            ys = torch.cat((ys, next_words.view(batch_size, 1)), dim=1)
                 
            values.append(value)
            critics.append(critic)
            all_probs.append(prob)
            gumbel_one_hots.append(one_hot)
            
        values = torch.stack(values,1)
        critics = torch.stack(critics,1).squeeze(-1)
        all_probs = torch.stack(all_probs,1)
        gumbel_one_hots = torch.stack(gumbel_one_hots, 1)
        
        return ys, values, critics, all_probs, gumbel_one_hots  
    
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        
        #the formula should be prob not logprob, I guess it still works
        return self.adaptive_softmax.log_prob(logits).exp()
        #return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        values, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(logits.shape[0], -1), ind, values, y    
# class Translator(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many 
#     other models.
#     """
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, critic_net):
#         super(Translator, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator
#         self.critic_net = critic_net
        
#     def forward(self, src, src_mask, max_len, start_symbol, mode = 'gumbel_softmax_sample'):
#         "Take in and process masked src and target sequences."
        
#         batch_size = src.shape[0]

#         memory = self.encode(src, src_mask)
#         ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
#         log_values = []
#         critics = []
#         all_log_probs = []
#         gumbel_one_hots = []
#         for i in range(max_len):
#             out = self.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src_mask))
#             #Gumbel softmax
#             one_hot, next_words, values, log_probs = self.generator(out[:, -1, :], mode = 'gumbel_softmax')
#             critic = self.critic_net(out[:, -1, :])      
#             ys = torch.cat((ys, next_words.view(batch_size, -1)), dim=1)
#             log_values.append(values)
#             critics.append(critic)
#             all_log_probs.append(log_probs)
#             gumbel_one_hots.append(one_hot)
#         log_values = torch.stack(log_values,1)
#         critics = torch.stack(critics,1)
#         all_log_probs = torch.stack(all_log_probs,1)
#         gumbel_one_hots = torch.stack(gumbel_one_hots, 1)

        
#         return ys, log_values, critics, all_log_probs, gumbel_one_hots
    
#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)
    
#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

    
class BERT(nn.Module):    
    def __init__(self, transformer_encoder, src_embed, padding_index):
        super(BERT, self).__init__()
        self.src_embed = src_embed
        self.transformer_encoder = transformer_encoder
        self.padding_index = padding_index
        
    def forward(self, x):
        src_mask = (x != self.padding_index).type_as(x).unsqueeze(-2)
        return self.transformer_encoder( self.src_embed(x), src_mask )

    
class Classifier(nn.Module):
    def __init__(self, BERT, out_class = 2, criterion = nn.CrossEntropyLoss(reduction='none')):
        super(Classifier, self).__init__()
        self.BERT = BERT
        
        self.criterion = criterion
        
        self.linear = nn.Linear(self.BERT.transformer_encoder.layers[-1].size, out_class)
        #self.softmax = nn.Softmax(-1)
        
    def forward(self, x, y, end_symbol):
        batch_size = x.shape[0]
        eos = torch.ones(batch_size, 1).fill_(end_symbol).type_as(x)
        x = self.BERT(torch.cat([x, eos], dim=1)) # bz, xlen+1, dmodel
        out = self.linear(x[:,0,:])       
        y = y.squeeze(1)
        
        acc = ((out.argmax(-1)) == y).type_as(out).mean()
        loss = self.criterion(out, y)

        return -loss, acc, out.argmax(-1)

# class Reconstructor(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many 
#     other models.
#     """
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad_index):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator
#         self.criterion = torch.nn.NLLLoss(reduction='none', ignore_index = pad_index)
        
#     def forward(self, src, src_mask, max_len, start_symbol, y, mode = 'teacher_forcing'):
#         "Take in and process masked src and target sequences."        
#         batch_size = src.shape[0]
#         memory = self.encode(src, src_mask)
#         ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)        
#         logits = []
#         #mode == 'sample
#         for i in range(max_len):
#             out = self.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src_mask))
#             log_probs = self.generator(out[:, -1, :])
#             if( i < max_len - 1):
#                 ys = torch.cat((ys, y[:,i+1].unsqueeze(-1)), dim = 1)
#             logits.append(log_probs)
            
#         logits = torch.stack(logits, 1)
#         loss = self.criterion(logits.view(batch_size * max_len, -1), y.view(batch_size * max_len))
        

#         #reconstruction with 100 len seen as 1 reward, and summary with 10 len seen as action
#         #loss = 
#         loss = loss.view(batch_size,-1).mean(-1)
#         acc = ((logits.argmax(-1)) == y).type_as(logits).mean()
        
#         return -loss, acc, logits.argmax(-1)
        
#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)
    
#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class LSTM_Normal_Encoder_Decoder(nn.Module):
    def __init__(self, hidden_dim, emb_dim, input_len, output_len, voc_size, pad_index, device, eps=1e-8, num_layers = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.device = device
          
        #self.input_len = input_len
        #self.output_len = output_len
        #self.voc_size = voc_size
        #self.teacher_prob = 1.
        #self.epsilon = eps
        
        self.num_layers = num_layers
        
        #self.emb_layer = nn.Embedding(voc_size, emb_dim)
        self.disguise_embed = nn.Linear(voc_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers=num_layers, batch_first=True)
        
        self.attention_softmax = nn.Softmax(dim=1)
        self.vocab_sz = voc_size
        
#         self.pro_layer = nn.Sequential(
#             nn.Linear(hidden_dim*4, voc_size, bias=True),
#             nn.LogSoftmax(-1)
#         )

        self.criterion = torch.nn.AdaptiveLogSoftmaxWithLoss(hidden_dim*4, voc_size, [100, 1000, 10000], div_value=4.0, head_bias=False)
        
    def forward(self, x, src_mask, max_len, start_symbol, y, mode = 'argmax', temp = 2.0):
        
        batch_size = x.shape[0]
        input_len = x.shape[1]
        device = x.device
        
        # encoder
        x_emb = self.disguise_embed(x)
        memory, (h, c) = self.encoder(x_emb)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()
        h = h.view(batch_size, self.num_layers, h.shape[-1]*2)
        c = c.view(batch_size, self.num_layers, c.shape[-1]*2)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()        

        
        ## decoder
        out_h, out_c = (h, c)
        
        logits = []
        for i in range(max_len):
            ans_emb = self.disguise_embed(self._to_one_hot(y[:,i], self.vocab_sz)).view(batch_size, 1, self.emb_dim)
            out, (out_h, out_c) = self.decoder(ans_emb, (out_h, out_c))
            attention = torch.bmm(memory, out.transpose(1, 2)).view(batch_size, input_len)
            attention = self.attention_softmax(attention)            
            
            context_vector = torch.bmm(attention.view(batch_size, 1, input_len), memory)
            
            logit = torch.cat((out, context_vector), -1).view(batch_size, -1)

            
#             if mode == 'argmax':
#                 values, next_words = torch.max(log_probs, dim=-1, keepdim=True)
#             if mode == 'sample':
#                 m = torch.distributions.Categorical(logits=log_probs)
#                 next_words = m.sample()
#                 values = m.log_prob(next_words)
           
            logits.append(logit)
                
        logits = torch.stack(logits, 1)
        

        
        
        _ ,loss = self.criterion(logits[:,:-1].contiguous().view(batch_size * (max_len - 1), -1), y[:,1:].contiguous().view(batch_size * (max_len-1)))
        
        #y from one to get rid of [CLS]
        log_argmaxs = self.criterion.predict(logits[:,:-1].contiguous().view(batch_size * (max_len - 1), -1)).view(batch_size, max_len-1)
        acc = ( log_argmaxs== y[:,1:]).float().mean()

        
        return loss, acc, log_argmaxs   
    
    def _to_one_hot(self, y, n_dims):

        scatter_dim = len(y.size())

        y_tensor = y.to(self.device).long().view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).to(self.device)

        return zeros.scatter(scatter_dim, y_tensor, 1)
    
#     def pretrian_forward(self, x, y, mode = 'argmax'):
        
#         batch_size = x.shape[0]
#         input_len = x.shape[1]
#         device = x.device
        
#         # encoder
#         x_emb = self.emb_layer(x)
#         memory, (h, c) = self.encoder(x_emb)
#         h = h.transpose(0, 1).contiguous()
#         c = c.transpose(0, 1).contiguous()
#         h = h.view(batch_size, 1, h.shape[-1]*2)
#         c = c.view(batch_size, 1, c.shape[-1]*2)
#         h = h.transpose(0, 1).contiguous()
#         c = c.transpose(0, 1).contiguous()        

        
#         ## decoder
#         out_h, out_c = (h, c)
#         max_len = y.shape[1]
#         logits = []
#         for i in range(max_len):     
#             ans_emb = self.emb_layer(y[:,i]).view(batch_size, 1, self.emb_dim)
#             out, (out_h, out_c) = self.decoder(ans_emb, (out_h, out_c))
#             attention = torch.bmm(memory, out.transpose(1, 2)).view(batch_size, input_len)
#             attention = self.attention_softmax(attention)            
            
#             context_vector = torch.bmm(attention.view(batch_size, 1, input_len), memory)
            
#             feature = torch.cat((out, context_vector), -1).view(batch_size, -1)

#             log_probs = self.pro_layer(feature)
            
#             if mode == 'argmax':
#                 values, next_words = torch.max(log_probs, dim=-1, keepdim=True)
#             if mode == 'sample':
#                 m = torch.distributions.Categorical(logits=log_probs)
#                 next_words = m.sample()
#                 values = m.log_prob(next_words)
                
#             logits.append(log_probs)
            
#         logits = torch.stack(logits, 1)
        
#         return logits, logits.argmax(-1)  


class Discriminator(nn.Module):
    def __init__(self, transformer_encoder, hidden_dim, vocab_sz, padding_index):
        super(Discriminator, self).__init__()
        
        self.padding_index = padding_index
        
        self.disguise_embed = nn.Linear(vocab_sz, hidden_dim)  
        self.transformer_encoder = transformer_encoder
        self.linear = nn.Linear(self.transformer_encoder.layers[-1].size, 1)
        #self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        src_mask = (x.argmax(-1) != self.padding_index).type_as(x).unsqueeze(-2)
        x = self.transformer_encoder(self.disguise_embed(x), src_mask)
        score = self.linear(x)
        return score
        
       
        
        