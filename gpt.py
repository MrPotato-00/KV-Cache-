import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import subprocess
import random
import time
import sys



class EarlyStop:
    def __init__(self, patience=10, min_delta=0):
        self.patience= patience
        self.counter=0
        self.best_loss= float('inf')

    def early_stop(self, loss):
        if loss<self.best_loss:
            self.best_loss= loss
            self.counter=0

        else:
            self.counter+=1

        if self.counter>=self.patience:
            return True

        return False

class CausalSelfAttention(nn.Module):
    ''' 
    This implementation tries to achieve a custom form of Self attention from where we are able to take out our target kv to cache
    
    '''

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0
        self.nheads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_k=None, past_v=None):

        ''' 
        This is the custom Attention Implementation so as to get the key,value required for building the kv-cache inference

        Args:
        x: (B, T, d_model) -- input tensor 
        past_k: (B, T, d_model) -- the stored or cached key embeddings used during inference
        past_v: (B, T, d_model) -- the stored or cache value embeddings used during inference

        Returns: 
        This returns the output embeddings from the model after application of attention and also returns the k and v useful for building the kv-cache

        out: (B, T, d_model) -- the weighted output from the transformer
        k: (B, T, d_model) or (B, T+1, d_model) -- the key to build the key cache
        v: (B, T, d_model) or (B, T+1, d_model) -- the value to build the value cache
        
        '''
        B, T, C = x.shape

        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)  #  same
        v = self.v_proj(x)  #  same  

        q = q.view(B, T, self.nheads, self.head_dim).transpose(1, 2)    # (B, nheads, T, head_dim)
        k = k.view(B, T, self.nheads, self.head_dim).transpose(1, 2)    # same
        v = v.view(B, T, self.nheads, self.head_dim).transpose(1, 2)    # same

        # activated only during inference -- during training no use to implement kv-caching and thus to fasten the inference kv-cache used, thus the implementation
        if past_k is not None and past_v is not None: 
            k_full = torch.cat([past_k, k], dim=2)  # (B, nheads, T+1, head_dim)
            v_full = torch.cat([past_v, v], dim=2)  # same
        else:
            k_full = k  # (B, T, d_model)
            v_full = v

        attn = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)    # (B, nheads, T, T)

        # Apply causal mask only during training, applied to prevent the model from looking at future tokens and prevent unfair advantage to predict the next-token
        if past_k is None:
            mask = torch.triu(                      # (T, T)
                torch.ones(T, T, device=x.device),
                diagonal=1
            ).bool()    
            attn = attn.masked_fill(mask, float("-inf"))    # (B, nheads, T, T)

        attn = torch.softmax(attn, dim=-1)

        ## -- dot product taking place remember --
        out = attn @ v_full     # (B, nheads, T, head_dim)  
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, d_model) -- merging the nheads with the head_dim
        out = self.out_proj(out)    # (B, T, d_model)

        # return only incremental k, v
        return out, k, v

class GPTBlock(nn.Module):
    ''' 
    This is the decoder block that takes up the input embeddings and outputs the embeddings after getting the output from the attention operation from the transformer
    
    '''
    def __init__(self, d_model, nheads, dff):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nheads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, d_model)
        )

    def forward(self, x, past_k=None, past_v=None):
        attn_out, new_k, new_v = self.attn(self.norm1(x), past_k, past_v) # already mentioned the dimensions of attn_out
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_k, new_v

class GPTModel(nn.Module):
    ''' 
    Flow of data:

    Input tensor >> (InputEmbedding + PositionalEncoding) >> CausalSelfAttention >> LayerNorm >> FeedForwardLayer >> Output Tensor
    
    '''

    def __init__(self, block_size, vocab_size, d_model, nheads, dff, nlayers):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, nheads, dff) for _ in range(nlayers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, past_kvs=None, pos_offset=0):
        B, T = idx.shape

        tok_embed = self.token_embedding(idx)

        # Sliding window position handling
        if T == 1 and past_kvs is not None:
            pos_val = past_kvs[0][0].size(2) % self.block_size
            pos = torch.tensor([pos_val], device=idx.device)
        else:
            pos = torch.arange(T, device=idx.device)

        pos_embed = self.position_embedding(pos)

        x = tok_embed + pos_embed

        new_kvs = []

        for i, block in enumerate(self.blocks):
            past_k = past_v = None
            if past_kvs is not None:
                past_k, past_v = past_kvs[i]

            x, new_k, new_v = block(x, past_k, past_v)
            new_kvs.append((new_k, new_v))

        x = self.norm(x)
        logits = self.ff(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T)
            )

        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()

        logits, _, past_kvs = self(idx)

        for _ in range(max_new_tokens):

            current_token = idx[:, -1:]

            logits, _, new_kvs = self(
                current_token,
                past_kvs=past_kvs
            )

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits_filtered = torch.full_like(logits, float("-inf"))
                logits_filtered.scatter_(1, indices, values)
                logits = logits_filtered

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat([idx, next_token], dim=1)

            # Update KV cache
            updated = []
            for (past_k, past_v), (new_k, new_v) in zip(past_kvs, new_kvs):

                past_k = torch.cat([past_k, new_k], dim=2)
                past_v = torch.cat([past_v, new_v], dim=2)

                # Trim to sliding window
                if past_k.size(2) > self.block_size:
                    past_k = past_k[:, :, -self.block_size:, :]
                    past_v = past_v[:, :, -self.block_size:, :]

                updated.append((past_k, past_v))

            past_kvs = updated

        return idx

class DataPreparation:
    def __init__(self, url= "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", train_split= 0.9):
        self.url= url
        self.stoi= None
        self.itos= None
        self.train_split= train_split

    def download_file(self):
        subprocess.run(["wget", self.url])
        
    def load_file(self,file_name= "input.txt"):
        if not os.path.exists(file_name):
            self.download_file()

        text= None
        with open(file_name, "r", encoding="utf-8") as f:
            text= f.read()
        return text

    def prepare_data(self):
        text= self.load_file()
        chars= sorted(list(set(text)))
        vocab_size= len(chars)

        self.stoi= {ch:i for i, ch in enumerate(chars)}
        self.itos= {i:ch for i, ch in enumerate(chars)}
        
        data= torch.tensor(self.encode(text), dtype= torch.long)

        n= int(self.train_split* len(data))
        train_data= data[:n]
        test_data= data[n:]

        return vocab_size, train_data, test_data

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])

    def get_batch(self, data_source, block_size, batch_size, vocab_size):

        ix= torch.randint(len(data_source)-block_size, (batch_size, ))

        x= torch.stack([data_source[i:i+block_size] for i in ix])
        y= torch.stack([data_source[i+1:i+block_size+1] for i in ix])

        x = torch.clamp(x, 0, vocab_size - 1)
        y = torch.clamp(y, 0, vocab_size - 1)
        return x, y



class Model:
    def __init__(self, **kwargs):
        # Extract parameters from kwargs
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.block_size = kwargs.get('block_size', 128)
        self.batch_size= kwargs.get('batch_size', 32)
        self.vocab_size = kwargs.get('vocab_size', 65)
        d_model = kwargs.get('d_model', 192)
        n_heads = kwargs.get('n_heads', 4)
        dff = kwargs.get('dff', 768)
        n_layers = kwargs.get('n_layers', 6)
        learning_rate = kwargs.get('learning_rate', 3e-4)
        self.epochs = kwargs.get('epochs', 1000)
        self.eval_intervals = kwargs.get('eval_intervals', 100)
        
        # Create model
        
        self.model = GPTModel(self.block_size, self.vocab_size, d_model, n_heads, dff, n_layers)
        self.model = self.model.to(self.device)   
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.early_stopping = EarlyStop()

    def model_train(self, data_prep, train_data, test_data):
        #train_x, train_y, test_x, test_y = train_x.to(self.device), train_y.to(self.device), test_x.to(self.device), test_y.to(self.device)
        for iter in range(self.epochs):
            self.model.train()
            train_x, train_y= data_prep.get_batch(train_data, self.block_size, self.batch_size, self.vocab_size)
            train_x, train_y= train_x.to(self.device), train_y.to(self.device)
            logits, loss, _ = self.model(train_x, train_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (iter+1) % self.eval_intervals == 0:
                with torch.no_grad():
                    total = 0
                    for _ in range(50):
                        test_x, test_y= data_prep.get_batch(test_data, self.block_size, self.batch_size, self.vocab_size)
                        test_x, test_y= test_x.to(self.device), test_y.to(self.device)
                        _, test_loss, _ = self.model(test_x, test_y)
                        total += test_loss.item()

                print(f"Step {iter+1}: Train loss: {loss.item():4f} Test loss: {total/50:4f}")

                if self.early_stopping.early_stop(test_loss):
                    print("Invoked the Early Stopping. Stopping training !!")
                    break

            #scheduler.step()
        
        #return self.model


    def run_inference(self, inp_text, context_length, temperature=1.0, top_k=10, data_prep_obj=None):
        # Simple generation test — prints the decoded output and token count
        context = torch.tensor([data_prep_obj.encode(inp_text)], dtype=torch.long, device=self.device)
        start_time= time.process_time()
        out = self.model.generate(context, context_length, temperature, top_k)
        
        print(data_prep_obj.decode(out[0].tolist()))  
        print("output token count:", out.size(1))
        elapsed_time= time.process_time()-start_time
        print(f"Time taken to complete the inference: {elapsed_time:.4f} seconds")

if __name__=='__main__':
    if len(sys.argv)<2:
        print("Usage: python3 gpt.py <some text for inference>")
        sys.exit(1)

    data_prep= DataPreparation()
    vocab_size, train_data, test_data= data_prep.prepare_data()
    
    gpt_model= Model(
        device= "cuda" if torch.cuda.is_available() else "cpu",

        block_size= 128,
        batch_size= 32,
        vocab_size= vocab_size,
        d_model= 192,
        n_heads= 4,
        n_layers= 6,
        dff= 4*192,

        learning_rate= 3e-4,
        epochs= 10000,
        eval_intervals= 500
    )
    gpt_model.model_train(data_prep, train_data, test_data)

    gpt_model.run_inference(sys.argv[1], 400, 0.8, 10, data_prep)