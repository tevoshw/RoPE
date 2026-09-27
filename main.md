# RoPE: Rotary Position Embedding

## 1. Why Do We Need Position Information?

Transformers process tokens through **self-attention**, which is **permutation-invariant**. This means the model doesn't inherently understand the order of tokens.

### Example:

```
"The mouse eats the cat"
vs
"The cat eats the mouse"
```

Without position information, both phrases produce the same attention patterns because the model only sees the **same set of tokens**, not their **sequence order**.

The solution: encode position information directly into the embeddings.

---

## 2. Why Not Absolute Positional Embedding?

The first approach used **Absolute PE**: create a separate learnable embedding matrix for positions and add it to the token embedding.

```
embedding_with_position = token_embedding + position_embedding
```

### Why This Fails:

1. **Fixed Sequence Length:** The position embedding matrix has a fixed size. If you train with max_length=512, you can't process sequences longer than 512 without retraining.

2. **Poor Generalization:** If you try to use a model trained on 512-token sequences on a 1024-token sequence, the position embeddings for positions 513+ don't exist.

3. **Inefficient Scaling:** To increase max sequence length, you need:
   - A larger position embedding matrix
   - Retraining the entire model
   - More memory and computation

4. **No Extrapolation:** The model has never seen position IDs beyond training max_length, so it can't handle them during inference.

**Better Solution:** RoPE doesn't require a fixed position matrix. It generates position information **dynamically** using rotation.

---

## 3. What is RoPE?

**RoPE (Rotary Position Embedding)** encodes position by **rotating the embedding vectors** according to their position in the sequence.

### Core Idea:

- Take pairs of embedding dimensions: (e₀, e₁), (e₂, e₃), ...
- Rotate each pair by an angle θ that depends on:
  - Token position (m)
  - Pair index (i)
  - Model dimension (d_model)

Each pair rotates by a **different amount**, creating position-aware embeddings.

---

## 4. The Math Behind RoPE

### 2D Rotation Formula

For a 2D vector (x, y), rotating by angle θ:

```
x' = x·cos(θ) - y·sin(θ)
y' = x·sin(θ) + y·cos(θ)
```

This is applied to each pair in the embedding:

```
e₀' = e₀·cos(θ) - e₁·sin(θ)
e₁' = e₀·sin(θ) + e₁·cos(θ)
```

### Calculating θ (Theta)

The rotation angle for each pair is:

```
θ = m · base^(-2i/d_model)

where:
  m       = token position (0, 1, 2, 5, 10, ...)
  base    = 10000 (standard constant)
  i       = pair index (0, 1, 2, 3, ...)
  d_model = embedding dimension
```

### Why Each Pair Gets Different θ?

The exponent changes with i:

```
Pair 0 (i=0):  θ = m · 10000^(-0/d_model)   = m · 1        ← LARGE rotation
Pair 1 (i=1):  θ = m · 10000^(-2/d_model)   = m · 0.316    ← SMALLER
Pair 2 (i=2):  θ = m · 10000^(-4/d_model)   = m · 0.1      ← EVEN SMALLER
```

**Result:** Early pairs rotate more aggressively, later pairs rotate subtly.

---

## 5. Concrete Example

### Setup:
- Token at position 5
- d_model = 10
- embedding = [2.5, 1.3, 4.2, 0.8, 1.0, 0.5, 2.1, 1.5, 0.9, 0.7]

### Pair 0: (e₀=2.5, e₁=1.3)

```
θ₀ = 5 · 10000^(-0/10) = 5 · 1 = 5 radians

cos(5) ≈ 0.284
sin(5) ≈ -0.959

e₀' = 2.5·(0.284) - 1.3·(-0.959) = 0.710 + 1.247 = 1.957
e₁' = 2.5·(-0.959) + 1.3·(0.284) = -2.398 + 0.369 = -2.029

Result: (1.957, -2.029)  ← significantly rotated
```

### Pair 1: (e₂=4.2, e₃=0.8)

```
θ₁ = 5 · 10000^(-2/10) = 5 · 0.316 ≈ 1.58 radians

cos(1.58) ≈ 0.001
sin(1.58) ≈ 1.0

e₂' = 4.2·(0.001) - 0.8·(1.0) = 0.004 - 0.8 = -0.796
e₃' = 4.2·(1.0) + 0.8·(0.001) = 4.2 + 0.0008 ≈ 4.2

Result: (-0.796, 4.2)  ← less rotated
```

---

## 6. Why This Works

### Extrapolation to Longer Sequences

RoPE doesn't require a fixed maximum length:

```
Trained on: sequences up to 512 tokens
Test on: sequences of 2048 tokens

Position embeddings at position 2000?
θ = 2000 · 10000^(-2i/d_model)
← Calculated on-the-fly, no matrix lookup needed!
```

### Frequency-Based Encoding

Different pair indices have different "frequencies":

```
Low indices  (i=0, 1, 2)     → high frequency (change rapidly)
High indices (i=d_model/2-1) → low frequency (change slowly)
```

This allows the model to:
- Use fast-changing pairs for absolute position
- Use slow-changing pairs for relative relationships

---

## 7. Why d_model Must Be Even

RoPE works with **pairs** of dimensions.

```
d_model = 768
Number of pairs = 768 / 2 = 384 pairs

Each token undergoes 384 independent rotations (one per pair)
```

If d_model is odd, the last dimension wouldn't have a pair to rotate with. That's why models typically use **even dimensions** (512, 768, 1024, etc.).

