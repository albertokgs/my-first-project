# Chapter: Attention Mechanisms in Deep Learning
## From First Principles to Attention Sinks

*A comprehensive treatment for the Stanford LLM course*

---

## Table of Contents

1. Motivation: Why Attention?
2. The Attention Intuition
3. Mathematical Foundations
4. Scaled Dot-Product Attention
5. Multi-Head Attention
6. Attention in Transformers
7. The Computational Lens
8. Advanced Topic: The Attention Sink Phenomenon
9. Lab Exercises

---

# Part I: Foundations

## 1. Motivation: Why Attention?

Before diving into the mathematics, let's understand *why* attention emerged as perhaps the most important architectural innovation in modern deep learning.

### 1.1 The Bottleneck Problem in Sequence-to-Sequence Models

Consider the classic encoder-decoder architecture for machine translation, circa 2014. An encoder RNN processes the input sentence word by word, updating a hidden state at each step:

$$h_t = f(h_{t-1}, x_t)$$

The final hidden state $h_T$ must encode *everything* about the input sentence—its meaning, structure, nuances—into a single fixed-dimensional vector. The decoder then generates the output sequence conditioned on this single vector.

This is the **information bottleneck**: all information about a potentially very long input must squeeze through a fixed-size vector. For short sentences, this works reasonably well. For longer sequences, critical information gets lost or diluted.

**Thought experiment**: Imagine summarizing a 500-page novel into a single paragraph, then asking someone to reconstruct the original from your summary. Impossible, right? This is precisely what we ask of the encoder's final hidden state.

### 1.2 The Alignment Problem

Another issue: when translating "The cat sat on the mat" to French, the decoder generating "chat" (cat) should primarily focus on "cat" in the source—not on "mat" or "sat". But with a single context vector, the decoder has no mechanism to selectively focus on relevant source words at each generation step.

Bahdanau et al. (2014) introduced attention precisely to solve this: instead of compressing the entire input into one vector, let the decoder *query* all encoder states and compute a weighted combination relevant to the current generation step.

### 1.3 The Attention Revolution

What started as a mechanism for sequence-to-sequence models turned out to be far more fundamental. The Transformer architecture (Vaswani et al., 2017) showed that attention alone—without recurrence—could achieve state-of-the-art results. This was revolutionary because:

1. **Parallelization**: Unlike RNNs that process sequentially, attention can process all positions in parallel
2. **Direct connections**: Any two positions can interact in a single layer, regardless of distance
3. **Interpretability**: Attention weights provide a window into what the model "looks at"

Today, attention is the backbone of GPT, BERT, T5, LLaMA, and virtually every successful language model.

---

## 2. The Attention Intuition

Before formalizing, let's build intuition through analogies.

### 2.1 Attention as Information Retrieval

The most useful mental model: **attention is soft database lookup**.

In a traditional database, you have a query and retrieve exact matching records. In attention:
- **Query (Q)**: What information am I looking for?
- **Keys (K)**: What information is available at each position?
- **Values (V)**: What information should be returned if a key matches?

The "softness" comes from returning a *weighted mixture* of all values rather than a single exact match. Weights are determined by query-key similarity.

### 2.2 A Concrete Example

Suppose you're reading a document to answer the question "What color is the car?" 

Your brain doesn't process every word equally. You scan for words like "car", "vehicle", "color", "red", "blue"—your query ("color of car") automatically guides attention to relevant passages.

If the document says "The red car was parked next to a blue truck," your attention naturally focuses on "red" when paired with "car". This is exactly what learned attention accomplishes.

### 2.3 Self-Attention: The Key Insight

Standard attention connects two different sequences (encoder-decoder). **Self-attention** applies attention within a single sequence, letting each position attend to all other positions.

Why is this powerful? Consider understanding the sentence: "The animal didn't cross the street because it was too tired."

What does "it" refer to? To resolve this, we need to connect "it" to "animal". Self-attention enables exactly this: when processing "it", the model can attend back to "animal" (and less to "street") based on learned relevance patterns.

---

## 3. Mathematical Foundations

We now formalize these intuitions. Pay careful attention to the dimensionality of each tensor—this is where confusion often arises.

### 3.1 Setup and Notation

Let's establish our notation precisely:

- **Sequence length**: $n$ (number of tokens/positions)
- **Model dimension**: $d_{\text{model}}$ (dimensionality of token representations)
- **Key/Query dimension**: $d_k$ (often equals $d_{\text{model}}/h$ where $h$ is the number of heads)
- **Value dimension**: $d_v$ (often equals $d_k$)

Given an input sequence $X \in \mathbb{R}^{n \times d_{\text{model}}}$, we compute:
- Queries: $Q = XW^Q$ where $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, giving $Q \in \mathbb{R}^{n \times d_k}$
- Keys: $K = XW^K$ where $W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, giving $K \in \mathbb{R}^{n \times d_k}$
- Values: $V = XW^V$ where $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, giving $V \in \mathbb{R}^{n \times d_v}$

**Critical insight**: Q, K, V are all *linear projections* of the same input X. The weight matrices $W^Q, W^K, W^V$ are learned parameters that transform the input into query, key, and value spaces respectively.

### 3.2 Why Different Projections?

A natural question: why not just use $X$ directly as Q, K, and V?

The answer lies in *representational flexibility*. By learning separate projections:
- The **query projection** learns "what this position is looking for"
- The **key projection** learns "what this position offers to others"
- The **value projection** learns "what information to send if selected"

These can encode different aspects of the same token. For "bank" in "river bank" vs "bank account", the key projection might encode that both relate to surfaces/institutions, while the value projection encodes the specific meaning in context.

### 3.3 Similarity Function: The Dot Product

The core of attention is measuring how much each query "matches" each key. We use the dot product:

$$\text{similarity}(q_i, k_j) = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{i,l} \cdot k_{j,l}$$

In matrix form, computing all pairwise similarities:

$$QK^T \in \mathbb{R}^{n \times n}$$

The $(i,j)$-th entry is the dot product between the $i$-th query and the $j$-th key.

**Geometric interpretation**: The dot product $q \cdot k = \|q\| \|k\| \cos\theta$. Higher values indicate:
1. Larger magnitudes (more confident/salient features), and/or
2. Smaller angles (more aligned directions in embedding space)

---

# Part II: The Core Mechanism

## 4. Scaled Dot-Product Attention

### 4.1 The Full Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's unpack each component:

**Step 1: Compute raw attention scores**
$$S = QK^T \in \mathbb{R}^{n \times n}$$

**Step 2: Scale by $\sqrt{d_k}$**
$$S_{\text{scaled}} = \frac{S}{\sqrt{d_k}}$$

**Step 3: Apply softmax row-wise**
$$A = \text{softmax}(S_{\text{scaled}}) \in \mathbb{R}^{n \times n}$$

Where softmax along each row:
$$A_{i,j} = \frac{\exp(S_{\text{scaled},i,j})}{\sum_{l=1}^{n} \exp(S_{\text{scaled},i,l})}$$

**Step 4: Weighted combination of values**
$$\text{Output} = AV \in \mathbb{R}^{n \times d_v}$$

### 4.2 The Scaling Factor: Why $\sqrt{d_k}$?

This is subtle but important. Without scaling, the dot products can grow very large in magnitude.

**The problem**: Suppose $q$ and $k$ are vectors whose components are independent random variables with mean 0 and variance 1. Then:

$$\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

So the standard deviation of dot products grows as $\sqrt{d_k}$.

For large $d_k$ (say, 64 or 128), dot products will have large variance. When fed into softmax, large magnitude inputs push the output toward one-hot vectors (all probability mass on one position). This causes:
1. **Vanishing gradients**: Softmax saturates, gradients become near-zero
2. **Reduced expressivity**: Model can only attend sharply to one position

**The solution**: Dividing by $\sqrt{d_k}$ normalizes the variance back to approximately 1, keeping softmax in its sensitive regime.

### 4.3 Softmax: Turning Similarities into Probabilities

The softmax function converts arbitrary real values into a probability distribution:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Key properties:
- **Output is non-negative**: All attention weights $\geq 0$
- **Sums to 1**: Each query's attention over all keys sums to 1
- **Monotonic**: Higher input scores → higher attention weights
- **Temperature sensitivity**: Can be modified to $\text{softmax}(z/\tau)$ where smaller $\tau$ makes distribution sharper

**Critical observation**: Softmax creates a **competition** among keys for each query's attention. This normalization constraint will be central to understanding attention sinks later.

### 4.4 Weighted Value Aggregation

The final step computes a weighted sum of value vectors:

$$o_i = \sum_{j=1}^{n} A_{i,j} v_j$$

Each output position $i$ receives a mixture of all value vectors, weighted by how much query $i$ attended to each key position $j$.

### 4.5 Complete Worked Example

Let's trace through a tiny example. Suppose $n=3$ (three tokens), $d_k = d_v = 2$.

**Input queries and keys** (already projected):
$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$$

**Step 1**: Compute $QK^T$:
$$QK^T = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 2 & 1 & 1 \end{pmatrix}$$

**Step 2**: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:
$$S_{\text{scaled}} = \begin{pmatrix} 0.71 & 0.71 & 0 \\ 0.71 & 0 & 0.71 \\ 1.41 & 0.71 & 0.71 \end{pmatrix}$$

**Step 3**: Softmax each row:
$$A \approx \begin{pmatrix} 0.40 & 0.40 & 0.20 \\ 0.40 & 0.20 & 0.40 \\ 0.51 & 0.24 & 0.24 \end{pmatrix}$$

**Step 4**: Compute output $AV$:
$$\text{Output} = \begin{pmatrix} 0.40 + 0.20 & 0.40 + 0.20 \\ 0.40 + 0.40 & 0.20 + 0.40 \\ 0.51 + 0.24 & 0.24 + 0.24 \end{pmatrix} = \begin{pmatrix} 0.60 & 0.60 \\ 0.80 & 0.60 \\ 0.75 & 0.49 \end{pmatrix}$$

Each output row is a weighted mixture of the value vectors.

---

## 5. Multi-Head Attention

### 5.1 Motivation: Multiple Perspectives

A single attention head learns one way to relate positions. But language has many types of relationships:
- Syntactic (subject-verb agreement)
- Semantic (coreference resolution)
- Positional (local context vs. long-range dependencies)
- Pragmatic (discourse coherence)

**Multi-head attention** runs several attention functions in parallel, each with its own learned projections, allowing the model to capture different types of relationships simultaneously.

### 5.2 The Multi-Head Formula

Given $h$ attention heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

With projections:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

Typically, $d_k = d_v = d_{\text{model}}/h$, so the total computation is comparable to single-head attention with full dimensionality.

### 5.3 Computational Flow

For a concrete example with $d_{\text{model}} = 512$ and $h = 8$ heads:

1. Each head uses $d_k = d_v = 64$
2. Each head computes attention independently on its 64-dimensional projections
3. Concatenating 8 heads gives a 512-dimensional vector
4. The output projection $W^O$ combines information across heads

**Key insight**: Different heads can specialize. In practice, some heads learn to attend to recent positions, others to syntactically related positions, others to semantically similar positions. This emergent specialization is not explicitly trained—it arises from the learning dynamics.

### 5.4 Visualizing Multiple Heads

When you visualize attention patterns, you'll often see:
- Some heads are very **local** (attend to nearby positions)
- Some heads are **global** (attend uniformly or to distant positions)
- Some heads are **syntactic** (verb attends to subject)
- Some heads are **sparse** (attend to one or two key positions)

This diversity is a strength: the model can capture multiple types of dependencies in parallel.

---

## 6. Attention in Transformers

### 6.1 Self-Attention vs. Cross-Attention

**Self-attention**: Q, K, V all derive from the same sequence $X$.
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

**Cross-attention**: Q comes from one sequence, K and V from another.
$$Q = X_{\text{target}}W^Q, \quad K = X_{\text{source}}W^K, \quad V = X_{\text{source}}W^V$$

In the original Transformer:
- Encoder uses self-attention only
- Decoder uses self-attention over previous outputs, then cross-attention to encoder states

Modern decoder-only models (GPT, LLaMA) use only self-attention with causal masking.

### 6.2 Causal Masking (The Attention Mask)

For autoregressive generation, position $i$ should only attend to positions $j \leq i$ (itself and earlier). We enforce this with a **causal mask**:

$$\text{Mask}_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Added to attention scores before softmax:
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right)$$

The $-\infty$ values become zeros after softmax, effectively removing future positions from consideration.

**The triangular structure**: This creates a lower-triangular attention pattern where:
- Position 1 can only attend to position 1
- Position 2 can attend to positions 1, 2
- Position $n$ can attend to all positions 1, ..., $n$

### 6.3 Positional Information

Self-attention is **permutation equivariant**: shuffling the input shuffles the output in the same way. This is problematic—"dog bites man" and "man bites dog" would have the same representation!

We inject positional information via **positional encodings/embeddings**:

**Sinusoidal (Vaswani et al., 2017)**:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

**Learned embeddings**: A learned matrix $E \in \mathbb{R}^{n_{\max} \times d_{\text{model}}}$

**Rotary Position Embedding (RoPE)**: Encodes relative position directly in the attention computation via rotation matrices. Used in LLaMA and many modern LLMs.

**ALiBi**: Adds position-dependent biases directly to attention scores.

The choice of positional encoding affects the attention sink phenomenon, as we'll see later.

### 6.4 The Complete Transformer Block

A Transformer block combines attention with a feedforward network:

```
Input → LayerNorm → Multi-Head Attention → Residual Connection →
      → LayerNorm → Feed-Forward Network → Residual Connection → Output
```

The residual connections are crucial: they allow gradients to flow directly and enable very deep networks. The layer normalization stabilizes training.

The feedforward network is typically:
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

Where $W_1$ expands dimensionality (e.g., from 512 to 2048) and $W_2$ projects back down.

---

# Part III: Computational and Theoretical Perspectives

## 7. The Computational Lens

### 7.1 Complexity Analysis

The computational cost of attention has specific characteristics worth understanding:

**Time complexity**: $O(n^2 d)$ where $n$ is sequence length, $d$ is dimension
- Computing $QK^T$: $O(n^2 d_k)$
- Softmax: $O(n^2)$
- Computing $AV$: $O(n^2 d_v)$

**Space complexity**: $O(n^2)$ for storing the attention matrix

The quadratic dependence on $n$ is the main limitation for long sequences. This has motivated efficient attention variants like Longformer, BigBird, and Flash Attention.

### 7.2 Attention as a Computational Primitive

Think of attention as a differentiable, content-based routing mechanism:

$$\text{Output}_i = \sum_j w_{ij} \cdot \text{Value}_j$$

Where $w_{ij}$ is determined by content ($q_i$, $k_j$) not by fixed structure.

This is fundamentally different from convolutions (fixed spatial patterns) or recurrence (fixed sequential pattern). Attention learns *which* positions to connect based on *what* is at those positions.

### 7.3 Attention as Information Flow

Each attention layer allows information to flow between positions. Consider a depth-$L$ Transformer:

- After layer 1: Each position has direct access to all positions
- After layer 2: Information can flow through intermediate positions
- After layer $L$: Complex multi-hop reasoning is possible

**The mixing perspective**: Each attention layer mixes representations across positions. The weights in the attention matrix determine how much each position's representation influences others.

But here's a crucial question: **What if a position doesn't need information from any other position?** What if mixing would actually *hurt* the representation?

This leads us directly to the attention sink phenomenon.

---

# Part IV: The Attention Sink Phenomenon

## 8. Attention Sinks: A Deep Dive

This section covers cutting-edge research from Gu et al. (ICLR 2025, COLM 2025) and Xiao et al. (ICLR 2024). Understanding attention sinks illuminates fundamental aspects of how Transformers process information.

### 8.1 The Observation

Researchers noticed a striking pattern in trained LLMs: **massive attention to the first token**, even when that token carries no semantic importance.

In models like LLaMA, Mistral, and GPT variants, the first token (often `<BOS>` or `<s>`) receives disproportionately high attention from many positions across many heads and layers. This happens regardless of what the first token actually is.

Key observations:
1. The first token receives very high attention scores (often >0.3 even with 64+ tokens)
2. This occurs even when the first token is semantically irrelevant
3. The pattern emerges during training, not at initialization
4. It occurs even with random tokens as input
5. It happens in models as small as 14M parameters

### 8.2 Associated Phenomena

Attention sink is connected to other curious behaviors:

**Massive Activations**: Certain dimensions in hidden representations have extremely large values (outliers/spikes). These tend to occur in the first token.

**Value Drains**: The value vectors of sink tokens become extremely small. Despite receiving high attention, they contribute little actual information to the output.

This combination is revealing: positions attend heavily to a token that contributes almost nothing. Why would the model learn this?

### 8.3 Mechanism Understanding: The Key Bias

Gu et al. showed that attention sink arises from a **key bias** in the first token:

The key vector of the sink token lies in a different manifold from other keys—specifically, it has small angles (high dot product) with *any* query. This makes it a universal attractor in key-query space.

**Why the first token specifically?**

The first token has a unique property under causal masking: its self-attention involves no other tokens. When processing position 1, the model can only attend to position 1 itself.

Mathematically, for the first token:
$$\text{Attention}(q_1, K_{1:1}, V_{1:1}) = \text{softmax}([q_1 \cdot k_1])v_1 = 1.0 \cdot v_1 = v_1$$

All hidden states at position 1 are equivalent to MLP transformations of input embeddings (no mixing with other positions). The model learns to use this property—it maps these position-1 embeddings to massive activations, creating the key bias that other positions can reliably attend to.

### 8.4 Why Do LLMs Need Attention Sinks?

This is the deep question. The answer: **to prevent over-mixing**.

**The over-mixing problem**: Attention layers mix representations across positions. But sometimes mixing is harmful—a position might have a perfect representation that shouldn't be contaminated by other positions.

Consider: after processing "The cat sat on the", position 5 ("the") might have a representation encoding "determiner about to be followed by a noun". Mixing this with other positions could dilute this precise encoding.

**Attention sink as "no-op"**: By attending heavily to a token with near-zero value vectors, a position can effectively skip the mixing step:

$$\text{Output} = A_{\text{sink}} \cdot v_{\text{sink}} + \sum_{j \neq \text{sink}} A_j \cdot v_j \approx 0 + \text{small contributions}$$

If $A_{\text{sink}} \approx 0.9$ and $v_{\text{sink}} \approx 0$, the position's representation passes through nearly unchanged (via the residual connection).

**Empirical validation**: When you perturb one token (e.g., change "greatest" to "best"), the representations of other tokens change minimally. The attention sink mechanism provides robustness against over-mixing.

### 8.5 When Do Attention Sinks Emerge?

Through systematic experiments, Gu et al. identified key factors:

**Optimization**:
- Attention sinks emerge during pre-training, not at initialization
- Larger learning rates encourage sink emergence
- More training steps strengthen the sink

**Data Distribution**:
- Sinks emerge when there's enough unique training data
- Data diversity is more important than dataset size

**Loss Function**:
- Weight decay (L2 regularization) encourages attention sink
- Prefix language modeling shifts the sink to within the prefix
- Shift window attention: sink appears on absolute (not relative) first token

**Model Architecture**:
- Positional embeddings (RoPE, ALiBi, learned, none) do NOT prevent sinks
- Pre-norm vs post-norm doesn't matter
- FFN activation function doesn't matter
- Number of attention heads doesn't matter

**The Softmax Normalization**: This is the critical factor. Softmax forces attention weights to sum to 1, creating competition among keys. If a position "doesn't need" to attend anywhere, it still must allocate 100% attention somewhere. The first token becomes the "nowhere" option—a learned dump for unwanted attention.

### 8.6 Eliminating Attention Sinks

If you remove the normalization constraint, attention sinks disappear:

**Sigmoid attention**: Replace softmax with element-wise sigmoid:
$$A_{ij} = \sigma\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)$$

No normalization means no competition. Each query-key pair is scored independently. If nothing is relevant, all scores can be near zero.

**ELU+1 attention**: Similar unnormalized approach using $\text{ELU}(x) + 1$.

Both alternatives eliminate attention sinks up to 1B parameters in experiments.

### 8.7 Modern Architectural Solutions

Major labs have incorporated these insights:

**GPT-OSS: Learnable Key Biases**

Instead of letting the first token develop a key bias organically, add an explicit learnable key bias:
$$\text{Attention}(Q, K + K_{\text{bias}}, V)$$

With $V_{\text{bias}} = 0$ (zero value bias).

Benefits:
1. No need for massive activations in actual tokens
2. Facilitates quantization (no outliers)
3. Better pre-training stability
4. Helps with shifted window attention for long contexts

**Softmax Off-by-One**: Add a virtual key that any query can attend to with zero similarity:
$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \,\Big|\, 0\right)$$

The extra "0" column provides a dump for attention without needing a real token.

**Qwen3-Next: Gated Attention**

This won the NeurIPS 2025 Best Paper Award. The idea: add a sigmoid gate to attention:

$$\text{Output} = \sigma(\text{gate}) \odot \text{Attention}(Q, K, V)$$

The gate can shut off attention entirely when not needed (no-op), eliminating the need for attention sinks. Results: no attention sinks, no massive activations, better long context, improved pre-training stability.

### 8.8 Summary: The Attention Sink Story

1. LLMs under causal masking develop attention sinks—tokens that receive high attention but contribute little information
2. This emerges because of softmax normalization forcing attention to sum to 1
3. Sinks provide a "no-op" mechanism to prevent over-mixing of representations
4. The first token is uniquely suited because it only attends to itself
5. Associated phenomena (massive activations, value drains) support this mechanism
6. Modern architectures (key biases, gated attention) provide explicit no-op mechanisms, eliminating the need for learned sinks

This research illuminates a fundamental aspect of Transformer information flow and has direct implications for model efficiency, quantization, and long-context processing.

---

# Part V: Laboratory Exercises

## 9. Lab Exercises

These exercises progress from basic implementations to investigating attention sinks.

### Lab 1: Implementing Attention from Scratch

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implement scaled dot-product attention.
    
    Args:
        Q: Query tensor of shape (batch, seq_len, d_k)
        K: Key tensor of shape (batch, seq_len, d_k)
        V: Value tensor of shape (batch, seq_len, d_v)
        mask: Optional mask tensor of shape (seq_len, seq_len)
    
    Returns:
        Output tensor of shape (batch, seq_len, d_v)
        Attention weights of shape (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # Step 1: Compute attention scores
    # Q @ K^T gives (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale by sqrt(d_k)
    # This keeps variance ~1, preventing softmax saturation
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    # Mask positions we shouldn't attend to with -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax to get attention weights
    # Softmax along the last dimension (the keys)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Weighted combination of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


# Test your implementation
def test_attention():
    batch_size, seq_len, d_k, d_v = 2, 4, 8, 8
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Check shapes
    assert output.shape == (batch_size, seq_len, d_v), f"Output shape: {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Weights shape: {weights.shape}"
    
    # Check that attention weights sum to 1 along key dimension
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), \
        f"Attention weights don't sum to 1: {weight_sums}"
    
    print("All tests passed!")
    print(f"Sample attention weights:\n{weights[0]}")

test_attention()
```

**Exercise 1.1**: Modify the function to return the attention entropy (how spread out the attention is). Higher entropy means more uniform attention; lower entropy means more focused.

**Exercise 1.2**: Implement the temperature parameter: `scores = scores / (math.sqrt(d_k) * temperature)`. Experiment with temperatures 0.5, 1.0, and 2.0. What happens to the attention distribution?

### Lab 2: Multi-Head Attention

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projection matrices for Q, K, V
        # Each one projects from d_model to d_model (then we split into heads)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def split_heads(self, x):
        """
        Split the last dimension into (n_heads, d_k).
        Input: (batch, seq_len, d_model)
        Output: (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
    
    def combine_heads(self, x):
        """
        Combine the head dimension back.
        Input: (batch, n_heads, seq_len, d_k)
        Output: (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # (batch, seq_len, n_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional causal mask
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
            Attention weights of shape (batch, n_heads, seq_len, seq_len)
        """
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Compute attention for all heads in parallel
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Combine heads and apply output projection
        context = self.combine_heads(context)
        output = self.W_o(context)
        
        return output, attention_weights


def create_causal_mask(seq_len):
    """Create a causal (lower triangular) mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


# Test multi-head attention
def test_mha():
    batch_size, seq_len, d_model, n_heads = 2, 8, 64, 4
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)
    
    output, weights = mha(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Visualize attention for first head
    print(f"\nHead 0 attention weights (sample 0):")
    print(weights[0, 0].detach().numpy().round(2))

test_mha()
```

**Exercise 2.1**: Add dropout to the attention weights (common in training). Where should it be applied?

**Exercise 2.2**: Implement cross-attention where Q comes from one sequence and K, V from another.

### Lab 3: Visualizing Attention Patterns

```python
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def visualize_attention(model_name="gpt2", text="The quick brown fox jumps over the lazy dog."):
    """
    Visualize attention patterns from a pre-trained model.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.attentions is a tuple: (layer_0, layer_1, ...)
    # Each layer has shape (batch, n_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    
    # Plot attention for each layer, averaged across heads
    fig, axes = plt.subplots(2, min(6, n_layers//2), figsize=(20, 8))
    axes = axes.flatten()
    
    for layer_idx in range(min(12, n_layers)):
        # Average attention across heads
        attn = attentions[layer_idx][0].mean(dim=0).numpy()
        
        ax = axes[layer_idx]
        im = ax.imshow(attn, cmap='viridis')
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
    
    plt.tight_layout()
    plt.savefig("attention_visualization.png", dpi=150)
    plt.show()
    
    return attentions, tokens


def measure_first_token_attention(attentions, tokens):
    """
    Measure how much attention goes to the first token.
    This is the attention sink metric.
    """
    print("\nAttention to first token (potential attention sink):")
    print("=" * 60)
    
    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: (batch, n_heads, seq_len, seq_len)
        # Get attention TO position 0 FROM all other positions
        attn_to_first = layer_attn[0, :, :, 0]  # (n_heads, seq_len)
        
        # Average attention to first token (excluding first token itself)
        mean_attn = attn_to_first[:, 1:].mean().item()
        max_attn = attn_to_first[:, 1:].max().item()
        
        print(f"Layer {layer_idx:2d}: mean={mean_attn:.4f}, max={max_attn:.4f}")


# Run visualization
# attentions, tokens = visualize_attention()
# measure_first_token_attention(attentions, tokens)

print("Lab 3 code ready. Uncomment the last two lines to run visualization.")
print("Requires: pip install transformers matplotlib")
```

**Exercise 3.1**: Run the visualization on a longer text (100+ tokens). Does attention to the first token increase?

**Exercise 3.2**: Compare attention patterns between different models (GPT-2 vs. LLaMA). Which shows stronger attention sinks?

### Lab 4: Detecting Attention Sinks

```python
def compute_sink_metric(attention_weights, threshold=0.3):
    """
    Compute the attention sink metric following Gu et al.
    
    A head exhibits sink behavior if attention to position 0
    exceeds threshold for many query positions.
    
    Args:
        attention_weights: Tensor of shape (n_heads, seq_len, seq_len)
        threshold: Attention threshold to consider as "sink"
        
    Returns:
        sink_score: Float between 0 and 1 indicating sink strength
    """
    n_heads, seq_len, _ = attention_weights.shape
    
    # Attention from each position to first position
    # Shape: (n_heads, seq_len)
    attn_to_first = attention_weights[:, :, 0]
    
    # For each head, what fraction of positions attend > threshold to first?
    # Exclude position 0 attending to itself (always 1.0 with causal mask)
    is_sink = (attn_to_first[:, 1:] > threshold).float()
    
    # Fraction of positions showing sink behavior, averaged over heads
    sink_score = is_sink.mean().item()
    
    return sink_score


def experiment_sigmoid_attention():
    """
    Compare softmax vs sigmoid attention to see if sigmoid prevents sinks.
    """
    batch_size, seq_len, d_k = 1, 32, 64
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    
    # Make first key have high dot product with all queries
    # This simulates learned attention sink behavior
    K[0, 0, :] = Q.mean(dim=(0, 1)) * 2  # Align with average query
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax attention (standard)
    softmax_attn = F.softmax(scores, dim=-1)
    
    # Sigmoid attention (no normalization)
    # Replace -inf with very negative number for sigmoid
    scores_for_sigmoid = scores.clone()
    scores_for_sigmoid[scores_for_sigmoid == float('-inf')] = -100
    sigmoid_attn = torch.sigmoid(scores_for_sigmoid)
    
    print("Attention to first token (potential sink):")
    print(f"Softmax attention[1:5, 0]: {softmax_attn[0, 1:5, 0].tolist()}")
    print(f"Sigmoid attention[1:5, 0]: {sigmoid_attn[0, 1:5, 0].tolist()}")
    
    print(f"\nSoftmax sink score: {compute_sink_metric(softmax_attn[0], threshold=0.2):.3f}")
    
    # For sigmoid, high attention to first doesn't force low attention elsewhere
    print(f"Sigmoid: first token attention doesn't compete with others")
    print(f"Sigmoid attention[2, :5]: {sigmoid_attn[0, 2, :5].tolist()}")


experiment_sigmoid_attention()
```

**Exercise 4.1**: Implement the learnable key bias approach: add a single learnable key that all queries can attend to. Compare attention patterns with and without this bias.

**Exercise 4.2**: Implement "softmax off-by-one": add an extra column of zeros to the attention scores before softmax. Measure how this affects attention to the first real token.

### Lab 5: Building a Minimal Transformer

```python
class TransformerBlock(nn.Module):
    """A single Transformer block with pre-norm architecture."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Pre-norm: normalize before attention
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask)
        x = x + self.dropout1(attn_out)  # Residual connection
        
        # Feedforward with pre-norm
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out  # Residual connection
        
        return x, attn_weights


class MinimalTransformer(nn.Module):
    """Minimal decoder-only Transformer for language modeling."""
    
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, 
                 d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
    def forward(self, x, return_attentions=False):
        batch, seq_len = x.shape
        
        # Embeddings
        tok_emb = self.tok_emb(x)
        pos = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Forward through blocks
        attentions = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attentions:
                attentions.append(attn)
        
        x = self.norm_f(x)
        logits = self.head(x)
        
        if return_attentions:
            return logits, attentions
        return logits


# Test the minimal transformer
def test_transformer():
    vocab_size = 1000
    model = MinimalTransformer(vocab_size)
    
    # Random input tokens
    x = torch.randint(0, vocab_size, (2, 16))
    
    logits, attentions = model(x, return_attentions=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of attention tensors: {len(attentions)}")
    print(f"Attention shape per layer: {attentions[0].shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

test_transformer()
```

**Exercise 5.1**: Train this model on a small text dataset (e.g., tiny Shakespeare). Monitor the attention sink metric during training. At what point does the sink emerge?

**Exercise 5.2**: Modify the model to add a learnable "sink token" that's prepended to every sequence. Does this concentrate the sink behavior on the explicit sink token?

---

## Appendix A: Key Equations Reference

**Scaled Dot-Product Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention**:
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

**Causal Mask**:
$$\text{Mask}_{i,j} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$$

**Attention Sink Metric**:
$$\text{SinkScore} = \frac{1}{h} \sum_{\text{heads}} \frac{1}{n-1} \sum_{i=2}^{n} \mathbb{1}[A_{i,1} > \tau]$$

---

## Appendix B: Further Reading

### Foundational Papers

1. **Bahdanau et al. (2014)**: "Neural Machine Translation by Jointly Learning to Align and Translate" - Original attention mechanism

2. **Vaswani et al. (2017)**: "Attention Is All You Need" - The Transformer architecture

3. **Devlin et al. (2018)**: "BERT: Pre-training of Deep Bidirectional Transformers" - Bidirectional attention

### Attention Variants

4. **Longformer**: "Longformer: The Long-Document Transformer" - Efficient attention patterns

5. **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

### Attention Sinks

6. **Xiao et al. (ICLR 2024)**: "Efficient Streaming Language Models with Attention Sinks"

7. **Gu et al. (ICLR 2025)**: "When Attention Sink Emerges in Language Models: An Empirical View"

8. **Gu et al. (COLM 2025)**: "Why Do LLMs Attend to the First Token?"

9. **Zihan et al. (NeurIPS 2025 Best Paper)**: "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"

---

*End of Chapter*
