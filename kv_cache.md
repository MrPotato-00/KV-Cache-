**Building GPT from Scratch and Implementing KV-Cache**
<br>
<br>
**Introduction**
<br>
Transformers are often used as black boxes. This was an attempt to:
- Implement a GPT-style decoder-only transformer from scratch
- Implement KV-Cache for efficient autoregressive inference

The goal was to deeply understand how the decoder-style text-generation works and how inference optimization using KV-Cache actually reduces computation. 

**1. Transformer Basics: The Magic of Attention**
<br>
Early text-preprocessing models such as RNNs struggled to capture long-range dependencies due to their sequential nature. Transformers changed this using the attention mechanism. 

The core of GPT is self-attention:
 
 $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$



 During training:
 - Each token produces a Query (Q), Key (K) and Value (V). 
 - Every query token compares itself with all key tokens. 
 - This requires computing similarity scores across all token pairs. 

 If sequence length is T:
 - Each token performs O(T) comparisons.
 - Across all tokens, this becomes O($T^2$). 

This quadratic complexity is acceptable during training but becomes expensive during long autoregressive generation. 

But we have a way out to reduce pre-step inference complexity from recomputing full attention for a single query over stored keys. During inference, using a workaround we can lower the computation complexity to O(T). 

**2. KV-Cache**
<br>
During inference, generation is autoregressive:
1. Take a sequence.
2. Predict the next token.
3. Append the predicted token. 
4. Repeat. 

Without optimization, each step recomputes Q, K, V for all previous tokens. This leads to repeated O($T^2$) computation over time. 

Past keys (K) and values (V) do not change once computed. 
So, instead of computing them, store the past K and V embeddings, and for each new token, compute only Q_new, K_new and V_new, concatenating K_new and V_new with stored memory. 

But there is always some trade-off. Attempt at decreasing the computation cost of O($T^2$) to O(T) has a memory tradeoff. We need to store the KV values for each of the layers of the decoder. This increases the memory constraint. 

But remember, this trick reduces inference latency. The model does not need compute similarity for all the query tokens, only on one query token at a time over the Key and Value embeddings. 

I have also implemented a non-kv-cache and a kv-cache based text-generation decoder transformers. There were some issues that I faced while implementing this kv-cache based transformer. 
Below are some issues faced:

**- Similarity Calculation during Inference**
<br>
During attention calculation there was confusion whether to field with K and V of the past_kv embeddings or concat with current K and V and then calculate the attention. The answer is to use the concat of current K and V with past_kv. This arrangement is to calculate the attention with all the cached KV and generate the next-token. 

**- Storage of the KV embeddings**
<br>
There was issue of when to store the key and value embedding to the KV cache. The simplest way is to just use the concatenated KV in attention calculation and store the new token in another instantiation. 

**- Issue with context-length size during inference**
<br>
Since the model is trained with positional encoding of size 128, when producing output of size >128 it produces CUDA error related to positional embedding index overflow due to accessing out-of-bound index. The way-out is to use a sliding-window technique and a wrap-around trick to use index within 128. This prevents the CUDA error. 


The implementation is not production-ready but it covers the way KV-cache is implemented and helped understand the internals of LLMs and how KV-cache speeds up inference during production. 

The link to the code will be available soon !!
