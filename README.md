# **nanoGPT**

## Table of Contents

- [Introduction](#introduction)
- [The Setup](#the-setup)
  - [Word Embeddings](#word-embeddings)
  - [Positional Embeddings](#positional-embeddings)
  - [Time Dimension](#time-dimension)
  - [Batch Dimension](#batch-dimension)
- [Simple Bigram Model](#simple-bigram-model)
  - [Results](#results)
- [The Transformer Architecture](#the-transformer-architecture)
  - [Implementation](#implementation)
  - [Model Summary](#model-summary)
  - [Training](#training)
  - [Results](#results-1)
  - [Reflection](#reflection)


## Introduction
This is a simple implementation of a transformer model to do a niche task. Much of the inspiration for this project came from Andrej Karpathy's video on the same project. The video can be found [here](https://youtu.be/kCc8FmEb1nY?si=2DIWvlNjZ7kWaxO1), and the GitHub repo for his implementation can be found [here](https://github.com/karpathy/nanoGPT). <br>

This model attempts to imitate Shakespeare-like language. At its heart, however, it is only a learning project and so does not aim to be perfect, and at the same time it does aim to as good as possible with the only constraint being scalability. <br>

The main contents of the project are in [main.ipynb](https://github.com/KeshavGulati/nanoGPT/blob/aff078872c4d566d166281e8779d33347c56aa17/main.ipynb), and the data is in [input.txt](https://github.com/KeshavGulati/nanoGPT/blob/aff078872c4d566d166281e8779d33347c56aa17/input.txt). In this README, I will outline the important parts of *main.ipynb*, providing descriptions and the corresponding code.

## The Setup
### Word Embeddings
To start, we need some representation of the data in a space that is learnable. So, we use a regular lookup table with learnable parameters, and assign each data point an ID. In this case, a datapoint is a character, and the ID is its index in the vocabulary. The lookup table has dimensions 
$\text{VOCAB\_SIZE} \times \text{EMB\_SIZE}$.
```python
# For individual characters
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}

# For words
encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])

temp = encode("Hello, World!")
print(temp)
print(decode(temp))
```

```bash
[20, 43, 50, 50, 53, 6, 1, 35, 53, 56, 50, 42, 2]
Hello, World!
```

### Positional Embeddings
In order for the model to be able to understand the relationship between the characters with respect to their position, we use another lookup table with learnable parameters. The dimensions of the table are $\text{MAX\_CONTEXT\_LENGTH} \times \text{EMB\_SIZE}$.

### Time Dimension
The time dimension in transformers refers to the context length we look at for each character. This helps in randomizing the training, but still maintaining some important patterns found in the data.

```python
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
  context = x[:t + 1]
  target = y[t]
  print(f"When input is {context} the target is {target}")
```

```bash
When input is tensor([18]) the target is 47
When input is tensor([18, 47]) the target is 56
When input is tensor([18, 47, 56]) the target is 57
When input is tensor([18, 47, 56, 57]) the target is 58
When input is tensor([18, 47, 56, 57, 58]) the target is 1
When input is tensor([18, 47, 56, 57, 58,  1]) the target is 15
When input is tensor([18, 47, 56, 57, 58,  1, 15]) the target is 47
When input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target is 58
```

### Batch Dimension
The training is done in batches as a 3D tensor, which can be efficiently utilized by GPUs.

```python
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
  data = train_data if split == "train" else val_data
  
  # Get random indices for the 4 batches
  ix = torch.randint(len(data) - block_size, (batch_size, ))

  # Get 8 blocks each from those 4 batches
  x = torch.stack([data[i: i + block_size] for i in ix])
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

  # We get a (4, 8) tensor, with batches as columns, and blocks as rows.

  return x, y

xb, yb = get_batch('train')

print(xb.shape, "\n", yb.shape)
```

```bash
torch.Size([4, 8]) 
 torch.Size([4, 8])
```

## Simple Bigram Model
Now we implmement a very simple bigram model. This model learns the transformation $f: \{0, 1, \dots V - 1\} \rightarrow \R^V $.
To do this, we use a $V \times V$ lookup table, where  is the size of our vocabulary, and the values of this table are the parameters we learn during training. This is a character level model, and so the vocabulary consists of all unique characters in our dataset.

```python
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    
    # logits.shape = B, T, C
    # This adds a C dimensions because for for every block of every batch, 
    #   we predict probability scores for every charcter in our vocab.
    # Pytorch's cross-entropy function wants C to be the second dimension in the logits input, so we reshape.
    
    if (targets is None):
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # idx is (B, T)
    for _ in range(max_new_tokens):
      logits, _ = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

vocab_size = len(vocab)  
m = BigramLanguageModel(vocab_size)
out, _ = m(xb, yb)
print(out.shape)

idx = torch.zeros((1, 1), dtype=torch.long)
tokens = m.generate(idx, max_new_tokens=100)[0].tolist()
print(decode(tokens))
```

```bash
torch.Size([32, 65])

SKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHjgChzbQ?u!3bLIgwevmyFJGUGp
wnYWmnxKWWev-tDqXErVKLgJ
```

### Results
See [main.ipynb](https://github.com/KeshavGulati/nanoGPT/blob/aff078872c4d566d166281e8779d33347c56aa17/main.ipynb)

## The Transformer Architecture
To improve performance, we will use a transformer architecture. This architecture is much better than a bigram model because it has a lot more learnable paramters, better encodes the relationship between characters in the vocabulary and so has better understanding of context, and also encodes positional context. So, it captures a lot more of the principal features of basic human language, and is able to better replicate it than the bigram model. <br>
**NOTE-** This is a decoder-only model.

### Implementation
```python
lass Decoder(nn.Module):
  def __init__(self, T: int, h: int):
    super().__init__()

    self.T = T
    self.h = h
    self.d_k = EMB_SIZE // h

    self.W_q = nn.Linear(EMB_SIZE, self.d_k, bias=False)
    self.W_k = nn.Linear(EMB_SIZE, self.d_k, bias=False)
    self.W_v = nn.Linear(EMB_SIZE, self.d_k, bias=False)

    self.encoder_norm1 = nn.LayerNorm(self.d_k, bias=False)
    self.encoder_norm2 = nn.LayerNorm(self.d_k, bias=False)
    self.decoder_norm1 = nn.LayerNorm(self.d_k, bias=False)
    self.decoder_norm2 = nn.LayerNorm(self.d_k, bias=False)
    self.decoder_norm3 = nn.LayerNorm(self.d_k, bias=False)

    self.feed_forward = nn.Sequential(
                          nn.Linear(EMB_SIZE, 4*EMB_SIZE),
                          nn.GELU(),
                          nn.Linear(4*EMB_SIZE, EMB_SIZE)
                        )
    
    self.norm1 = nn.LayerNorm(EMB_SIZE, device=device)
    self.norm2 = nn.LayerNorm(EMB_SIZE, device=device)
    self.proj = nn.Linear(h * self.d_k, EMB_SIZE)
    
  def _self_attention(self, 
                      Q: torch.Tensor, 
                      K: torch.Tensor, 
                      V: torch.Tensor, 
                      mask: bool = False):
    
    _, T, _ = Q.shape
    a1 = torch.bmm(Q, K.transpose(1, 2)) / ((self.d_k) ** (1/2))

    if (mask):
      # masking future values
      helper = torch.tril(torch.ones((T, T), device=Q.device))
      a1 = a1.masked_fill(helper == 0, float('-inf'))

    a2 = F.softmax(a1, dim=-1)
    attention = torch.bmm(a2, V)
    return attention
  
  def _multi_head_attention(self, 
                            Q: torch.Tensor, 
                            K: torch.Tensor, 
                            V: torch.Tensor, 
                            mask: bool = False):
    
    b1 = [self._self_attention(Q, K, V, mask) for h in range(self.h)]
    b2 = torch.cat(b1, dim=-1)
    mh_attention = self.proj(b2)
    return mh_attention
  
  def _add_and_norm(self, 
                    x1: torch.Tensor, 
                    x2: torch.Tensor,
                    norm_layer: torch.nn.Module):
    
    a = x1 + x2
    return norm_layer(a)
  
  
  # def encoder(self, 
  #             x: torch.Tensor):
    
  #   Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

  #   mh_out = self._multi_head_attention(Q, K, V)
  #   add_norm_out1 = self._add_and_norm(x, mh_out)

  #   ff_out = self.feed_forward(add_norm_out1)
  #   add_norm_out2 = self._add_and_norm(add_norm_out1, ff_out)

  #   return add_norm_out2
  
  def forward(self, 
              x: torch.Tensor, 
              encoder_out: torch.Tensor | None = None
              ):
    
    Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

    mh_out1 = self._multi_head_attention(Q, K, V, mask=True)
    add_norm_out1 = self._add_and_norm(x, mh_out1, self.norm1) # (B, T, C)

    # mh_out2 = self._multi_head_attention(encoder_out, encoder_out, add_norm_out1)
    # add_norm_out2 = self._add_and_norm(add_norm_out1, mh_out2)

    ff_out = self.feed_forward(add_norm_out1)
    add_norm_out3 = self._add_and_norm(add_norm_out1, ff_out, self.norm2)

    return add_norm_out3
  
  
class ShakespeareGPT(nn.Module):
  def __init__(self, T, h, num_blocks):
    super().__init__()

    self.T = T
    self.h = h

    self.get_token_embeddings = nn.Embedding(VOCAB_SIZE, EMB_SIZE)
    self.get_pos_embeddings = nn.Embedding(self.T, EMB_SIZE)

    self.lm_head = nn.Linear(EMB_SIZE, VOCAB_SIZE)

    self.decoder_blocks = nn.ModuleList([
      Decoder(T, h) for _ in range(num_blocks)
    ])

  def forward(self, 
            X: torch.Tensor, 
            Y: torch.Tensor | None = None
            ):
  
    loss = None
    
    B, T = X.shape
    
    tok_emb = self.get_token_embeddings(X) # x.shape = B, T, C
    pos_emb = self.get_pos_embeddings(torch.arange(T, device=X.device))
    x = tok_emb + pos_emb

    # encoder_out = self.encoder(x)
    for decoder in self.decoder_blocks:
      x = decoder(x)

    logits = self.lm_head(x)
    # logits, loss = F.softmax(logits, dim=-1), None

    if Y is not None:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      Y = Y.view(B*T)
      loss = F.cross_entropy(logits, Y)

    return logits, loss

  def generate(self, 
              idx: list[int], 
              max_new_tokens: int
              ):
    
    for _ in range(max_new_tokens):
      block_idx = idx[:, -self.T: ]
      logits, loss = self(block_idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      
      # Get index with highest probability
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx  
```

### Model Summary
```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ShakespeareGPT                           [1, 8, 65]                --
├─Embedding: 1-1                         [1, 8, 256]               16,640
├─Embedding: 1-2                         [8, 256]                  65,536
├─ModuleList: 1-3                        --                        --
│    └─Decoder: 2-1                      [1, 8, 256]               320
│    │    └─Linear: 3-1                  [1, 8, 64]                16,384
│    │    └─Linear: 3-2                  [1, 8, 64]                16,384
│    │    └─Linear: 3-3                  [1, 8, 64]                16,384
│    │    └─Linear: 3-4                  [1, 8, 256]               65,792
│    │    └─LayerNorm: 3-5               [1, 8, 256]               512
│    │    └─Sequential: 3-6              [1, 8, 256]               525,568
│    │    └─LayerNorm: 3-7               [1, 8, 256]               512
│    └─Decoder: 2-2                      [1, 8, 256]               320
│    │    └─Linear: 3-8                  [1, 8, 64]                16,384
│    │    └─Linear: 3-9                  [1, 8, 64]                16,384
│    │    └─Linear: 3-10                 [1, 8, 64]                16,384
│    │    └─Linear: 3-11                 [1, 8, 256]               65,792
│    │    └─LayerNorm: 3-12              [1, 8, 256]               512
│    │    └─Sequential: 3-13             [1, 8, 256]               525,568
│    │    └─LayerNorm: 3-14              [1, 8, 256]               512
│    └─Decoder: 2-3                      [1, 8, 256]               320
│    │    └─Linear: 3-15                 [1, 8, 64]                16,384
│    │    └─Linear: 3-16                 [1, 8, 64]                16,384
│    │    └─Linear: 3-17                 [1, 8, 64]                16,384
│    │    └─Linear: 3-18                 [1, 8, 256]               65,792
│    │    └─LayerNorm: 3-19              [1, 8, 256]               512
│    │    └─Sequential: 3-20             [1, 8, 256]               525,568
│    │    └─LayerNorm: 3-21              [1, 8, 256]               512
│    └─Decoder: 2-4                      [1, 8, 256]               320
│    │    └─Linear: 3-22                 [1, 8, 64]                16,384
│    │    └─Linear: 3-23                 [1, 8, 64]                16,384
│    │    └─Linear: 3-24                 [1, 8, 64]                16,384
│    │    └─Linear: 3-25                 [1, 8, 256]               65,792
│    │    └─LayerNorm: 3-26              [1, 8, 256]               512
│    │    └─Sequential: 3-27             [1, 8, 256]               525,568
│    │    └─LayerNorm: 3-28              [1, 8, 256]               512
│    └─Decoder: 2-5                      [1, 8, 256]               320
│    │    └─Linear: 3-29                 [1, 8, 64]                16,384
│    │    └─Linear: 3-30                 [1, 8, 64]                16,384
│    │    └─Linear: 3-31                 [1, 8, 64]                16,384
│    │    └─Linear: 3-32                 [1, 8, 256]               65,792
│    │    └─LayerNorm: 3-33              [1, 8, 256]               512
│    │    └─Sequential: 3-34             [1, 8, 256]               525,568
│    │    └─LayerNorm: 3-35              [1, 8, 256]               512
│    └─Decoder: 2-6                      [1, 8, 256]               320
│    │    └─Linear: 3-36                 [1, 8, 64]                16,384
│    │    └─Linear: 3-37                 [1, 8, 64]                16,384
│    │    └─Linear: 3-38                 [1, 8, 64]                16,384
│    │    └─Linear: 3-39                 [1, 8, 256]               65,792
│    │    └─LayerNorm: 3-40              [1, 8, 256]               512
│    │    └─Sequential: 3-41             [1, 8, 256]               525,568
│    │    └─LayerNorm: 3-42              [1, 8, 256]               512
├─Linear: 1-4                            [1, 8, 65]                16,705
==========================================================================================
Total params: 3,950,017
Trainable params: 3,950,017
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 4.41
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.90
Params size (MB): 15.79
Estimated Total Size (MB): 16.69
==========================================================================================
```

### Training
The training loss at the last epoch was 0.4115. See [main.ipynb](https://github.com/KeshavGulati/nanoGPT/blob/aff078872c4d566d166281e8779d33347c56aa17/main.ipynb) for full progress.

### Results
This model generated the following text:

```text
COMINIUS:
O, will dost those badren ourselves,
The common vast flood hap to your grace
From gibines worship, for the modest moon
Upon this hour, marching to command;
Only red the rank of this kingly violent.
See, how now, which fair deputy,
Dightsing fright hay. Usuff into thisding Margares?
Are they letter hem mine, their tongues have for ever.

KATHARINA:
An's we suppose so.
3 repare her, I talk of this: I slain
Sometime companion blots to serve our hands:
O that once cannot be forsworn.
Now, break, soft! what news? Warest hast thou with thee?
'Tis believe that resign thine own son,
Thou counterfeit'st me; this is an end.

KING HENRY VI:
Ah, know you not when you shall be my son:
Hidest grace the away: laid thy anchorse look
From my former hands.

FRIAR LAURENCE:
Where is the business of the king?

PARIS:
Monday, my lord.

JULIET:
What is my office, sir?

JULIET:
Go thou queate the casafe, and my friend Norfolk
Upon some more piece with an agazem. Take good counsellor.

PRINCE EDWARD:
I do beseech you, look to ask him what this man sword,
Should be incense the adminished letters,
And believe in heaven and many of your accusations--
Learn us me pondallity, and was it beat before must
Became of thunder-law.

POMPEY:
If you should smile he's infection,
Is more than spokes your father joys.
Why, in this country, and his minim not
To Bolingbroke the lamentation of his majesty
Which he sends a sweetly deputed
That eighty words and shake him lives,
And made roaring look on thee. Honour now!

CAMILLO:
My lord,
Shall be true summer, sand my lord remempt
Suck her on her! Gembord Clarence, fair lords of this fair land!
When Oxford, friends, make war. Whence, men!
If thou art smip them to seek a few brat?
Ay, like a good friar, surely.
Where's my father withholds love come to him.

BUCKINGHAM:
What, talk you with slow this?

DERBY:
It is the matter: why, o, then I'll prove him,
That stay my appetite, now will in hand.

BRUTUS:
We stood to't them all.

CORIOLANUS:
Dire not, s
```

### Reflection
I think this is a very good result given the small scale of the project. This model has about 4M parameters, and with further fine-tuning it can do even better.