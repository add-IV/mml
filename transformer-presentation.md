# Transformer  Presentation

## Introduction

- paper "Attention is All You Need" by Vaswani et al. in 2017
- attention mechanisms before transformer
- encoder-decoder architecture
- nlp tasks

## Core Topics

- Embeddings
  - cosine similarity
- Attention
- Multi-Head Attention
- Cross-Attention
- Transformer Architecture

## Additional Topics

- Masking
- Positional Encoding
- Context Size
- Output Matrix Splitting
- Superposition
- other applications of transformers (e.g. stable diffusion)
- other kinds of attention mechanisms

### Attention is All You Need

### attention mechanisms before transformer

RNN, LSTM, GRU, CNN

### encoder-decoder architecture

### Embeddings

- word embeddings, token embeddings
- positional embeddings
- "similarity" in embeddings
- other embedding spaces

### cosine similarity

- cos(theta) = dot(a, b) / (norm(a) * norm(b))

### Attention

- Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
- Q, K, V: query, key, value

### Multi-Head Attention

- MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
- head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

### Cross-Attention

- in models where encoder and decoder are separate

### Masking

- for training to prevent model from cheating

### Positional Encoding

- to give model information about position of tokens

### Context Size

- how many tokens to consider in attention
- what happens when context size grows

### Output Matrix Splitting

- to save on parameters

### Superposition

- number of facts in embeddings