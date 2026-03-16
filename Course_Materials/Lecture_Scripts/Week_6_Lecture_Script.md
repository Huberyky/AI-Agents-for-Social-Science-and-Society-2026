# Week 6 Lecture Script: Auto-Encoders, Interpretability, and Steering Agents
**Course:** AI Agents for Social Science and Society 2026
**Instructor:** James A. Evans
**Date:** February 13, 2026
**Room:** 1155 E. 59th Street, Room 295
**Duration:** 3 hours (1:30–4:20 PM)
**Notebook:** `week6_2026.ipynb` (391 cells)

---

## Instructor Preparation Notes

Before class:
- Load the notebook and confirm `transformer_lens`, `einops`, `jaxtyping`, `circuitsvis`, and `sae-lens` all install in Colab
- The `HookedTransformer.from_pretrained("gpt2-small")` call in Cell 7 downloads ~500MB — do this before class
- Pre-run the attention visualization cells (Cells 33, 47) — CircuitsVis outputs are HTML and may require specific Colab settings
- Have the ARENA 3.0 tutorial framework loaded — the module borrows heavily from it
- The steering vectors module (Module 3) needs GPU for Llama-2 models; prepare the simpler GPT-2 demo instead for live demonstration
- Print or have ready: Kim, Evans, Schein ICLR 2025 "Linear Representations of Political Perspective" — particularly the PCA figure showing political ideology as a linear dimension

---

## Section 1: Opening — Why We Need to Look Inside (0:00–0:15)

**[Stand at board. Nothing projected yet.]**

Say: "Every technique we've used until now has treated the neural network as a function: inputs go in, outputs come out, and we evaluate those outputs. Today we do something fundamentally different. Today we open the model and ask: what is actually happening inside?"

Write on board:
```
Black box approach: f(x) = y
                   [optimize loss, evaluate y]

Mechanistic approach: f(x) = Φ₁ → Φ₂ → ... → Φₙ → y
                     [understand each Φ, trace causally]
```

Say: "This is not just an intellectual exercise. There are at least three concrete scientific reasons to care about model internals."

Write on board:
```
1. SAFETY: Can we verify a model isn't reasoning deceptively?
2. SOCIAL SCIENCE: Do models encode social concepts (race, gender, ideology)?
3. CONTROL: Can we modify behavior without retraining?
```

Ask students: "Can you think of a case in social science where knowing the internal representation of a model would directly change how you'd interpret its output?"

**[Expected: If a sentiment classifier encodes racial stereotypes in its hidden layers, its predictions on racially-coded text are confounded; knowing this changes how you'd use the predictions.]**

Say: "Kim, Evans, and Schein's ICLR 2025 paper 'Linear Representations of Political Perspective' is a perfect example. They found that political ideology is encoded as a linear direction in the residual stream of LLMs. A one-dimensional political compass — from progressive to conservative — emerges as a geometric structure inside the model. That means when an LLM generates political text, it's navigating a literal political ideological space inside its own activations."

**[Pause for effect.]**

Say: "This has profound implications for how we use AI agents in political research, how we understand AI-generated misinformation, and how we design AI advisors that remain politically neutral. The interpretability methods we cover today are what make this research possible."

**[Timing: 10 minutes]**

---

## Section 2: TransformerLens — Mechanistic Interpretability Basics (0:15–0:55)

**[Open notebook. Navigate to Cell 1 — Module 1 header.]**

Say: "TransformerLens is the primary toolkit for mechanistic interpretability research. It provides two critical capabilities that standard PyTorch doesn't: the hooks API for intercepting and modifying activations at any point in the model, and activation caching for storing all activations from a single forward pass."

### Loading and Navigating a Model

**[Navigate to Cells 5–7.]**

Say: "We load GPT-2 Small as our working model — 80 million parameters, 12 layers, 12 attention heads per layer. Large enough to exhibit interesting behavior, small enough to run on CPU."

```python
gpt2_small = HookedTransformer.from_pretrained("gpt2-small")
```

Say: "The key difference from a standard HuggingFace model is that TransformerLens separates the weight matrices. Instead of one concatenated QKV matrix, we get separate `W_Q`, `W_K`, `W_V` per layer. This makes mechanistic analysis much cleaner."

**[Navigate to Cells 22–25 — activation caching.]**

Say: "The activation cache is the central object. When we run `model.run_with_cache(tokens)`, we get back not just logits but every intermediate activation — every attention pattern, every residual stream value, every MLP activation — indexed by name."

```python
gpt2_text = "Natural language processing tasks, such as question answering..."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# Access any activation by name:
attn_layer0 = gpt2_cache["pattern", 0]    # shape: [n_heads, seq_Q, seq_K]
residual_stream = gpt2_cache["resid_post", 5]  # residual stream after layer 5
```

Ask students: "What is the residual stream and why is it the central object in transformer interpretability?"

**[Expected: The residual stream is the sequence of vectors that passes through the entire model, with each layer's output added to it (the residual connection). Every layer reads from and writes to the residual stream.]**

Say: "Neel Nanda's insight in the 'Mathematical Framework for Transformer Circuits' paper is that the residual stream is the communication channel of the transformer. Attention heads write information in to it; MLP layers read from it and transform it; subsequent attention heads read what previous ones wrote. The entire model is a series of additions to the residual stream."

Write on board:
```
Residual stream at position t:
  x₀(t) = token_embedding(t) + positional_embedding(t)
  x₁(t) = x₀(t) + attn_output_layer0(t)
  x₂(t) = x₁(t) + mlp_output_layer0(t)
  ...
  logits(t) = unembed(x_final(t))

Key: each layer ADDS to the stream; nothing is destroyed
→ information can persist across many layers
```

### Visualizing Attention Patterns

**[Navigate to Cell 33.]**

Say: "The most interpretable component of a transformer is its attention patterns. Each head computes a probability distribution over source positions, telling us which positions it attends to when processing each target position."

**[Show Cell 33 — circuitsvis visualization.]**

```python
attention_pattern = gpt2_cache["pattern", 0]  # [n_heads, seq, seq]
display(cv.attention.attention_patterns(
    tokens=gpt2_str_tokens,
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(12)]
))
```

Say: "You'll see three dominant patterns repeated across many heads. Heads that attend predominantly to the current token — diagonal pattern. Heads that attend to the previous token — one-step-behind diagonal. Heads that attend to the first token — first column solid. These are the basic 'vocabulary' of attention head behaviors."

**[Navigate to Cell 48 — previous/current/first token head definitions.]**

Say: "These patterns have names: previous-token heads, current-token heads, and first-token heads. They're not random — they serve specific computational purposes. Previous-token heads are what make induction circuits possible."

### Induction Heads — The Core Mechanism

**[Navigate to Cell 36 — induction heads section.]**

Say: "Induction heads are one of the most important mechanistic discoveries in the transformer interpretability literature. They implement a remarkably elegant algorithm for in-context learning."

Write on board:
```
Induction head algorithm:
  Given context: ... [A][B] ... [A][?]

  1. Previous-token head in layer 0: when processing [A], attend to [A] and copy a
     signal that "B followed me last time"
  2. Induction head in layer 1: when processing second [A], look for tokens where
     the layer-0 signal says "B followed me" and attend to those positions
  3. Output: predict [B] as the next token after the second [A]

  Result: the model can continue any repeated sequence, even sequences
  never seen during training!
```

Ask students: "Why is this surprising from the perspective of generalization?"

**[Expected: The model is generalizing to novel combinations of tokens it never saw together, using an abstract rule it learned, not memorization.]**

Say: "This is the mechanism underlying in-context learning. When you give GPT-4 five examples of a classification task and it generalizes to the sixth, induction circuits are (in part) what's doing that generalization. They're implemented across a two-layer circuit — one head sets up the signal, one head reads it."

**[Navigate to Cells 55–64 — identifying induction heads on repeated sequences.]**

Say: "The experimental test is elegant. Generate a sequence of random tokens, repeat it, and run the model. An induction head will show a characteristic 'diagonal stripe' in its attention pattern — shifted by the sequence length. Head 0.7 shows the previous-token pattern; head 1.6 shows the induction pattern. Together they form the induction circuit."

**[Show the code for the repeated token test, Cell 57:]**

```python
def generate_repeated_tokens(model, seq_len, batch=1):
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens
```

Say: "Now run this through the model and visualize the attention patterns. You'll see that on the random-token segment, attention is diffuse. On the repeated segment, induction heads snap to attention — they correctly identify which positions to copy from, achieving near-perfect prediction on tokens they have never seen in any combination."

**[Navigate to Cell 64 — induction head detector.]**

Say: "We can formalize this as a score: the average attention probability to the position exactly `seq_len - 1` steps back. If that score is high, it's an induction head."

**[Navigate to Cell 166 — targeted ablation.]**

Say: "The ablation experiment closes the causal story. When we zero out the output of head 0.7 — the previous-token head — the induction score of head 1.6 drops from 0.68 to nearly zero. This is causal circuit analysis: we've identified not just correlation but mechanism."

**[Timing: 40 minutes]**

---

## Section 3: Sparse Autoencoders — Decomposing Superposition (0:55–1:25)

Say: "Now we move from attention heads to a deeper question: what are neurons actually computing? This turns out to be more complex than it should be, and the answer requires us to reconsider fundamental assumptions about how neural networks work."

**[Navigate to Cell 177 — Module 2 header — then to Cell 230.]**

Say: "First, let's build intuition with standard autoencoders, then we'll get to the mechanistic interpretability version."

### Autoencoders — Compression and Reconstruction

**[Navigate to Cell 185–198.]**

Say: "An autoencoder compresses data into a bottleneck latent vector and then reconstructs it. The encoder learns to extract the most salient features; the decoder learns to reconstruct from them."

Write on board:
```
Autoencoder:
  Input x → [Encoder] → z (bottleneck) → [Decoder] → x̂

  Loss: ||x - x̂||²  (reconstruction loss)

  Key: the bottleneck forces learning of compressed representation
  z has much lower dimension than x → 96.2% compression on MNIST!
```

**[Reference Cell 199:]**
```python
30 / (28 * 28)  # = 0.038  → 3.8% of original dimensionality retained
```

Say: "But here's the sociological question: what does that 30-dimensional bottleneck represent? For handwritten digits, we suspect it encodes stroke direction, digit class, thickness, slant. These are the semantic features of digits. The autoencoder has discovered them without being told what they are."

### The Superposition Hypothesis

**[Navigate to Cell 230 — toy models of superposition.]**

Say: "Here's the deep problem for neural network interpretability. A language model with 4,096 neurons needs to represent potentially millions of concepts. These can't all be orthogonal directions in a 4,096-dimensional space — there aren't enough dimensions. The model must encode many features in the same neurons through interference."

Write on board:
```
Superposition Hypothesis (Anthropic, 2022):

  Models represent MORE features than they have neurons/dimensions.
  Multiple features are encoded in the same neurons simultaneously.

  Consequence: individual neurons are not monosemantic (single feature)
  → a neuron "most active on" bananas also activates on unrelated concepts
  → it's POLYSEMANTIC

  This is why ablating a single neuron rarely tells you much.
```

Ask students: "If a model has 4,096 neurons but needs to represent 100,000 features, what's the mathematical structure it would need to use?"

**[Expected: Near-orthogonal vectors in high-dimensional space — random vectors in high dimensions are approximately orthogonal.]**

Say: "Exactly. The model packs many feature vectors into the same space by using nearly-orthogonal directions. They interfere slightly — this is the 'noise' from superposition — but the model learns to route information correctly because each feature has its own direction."

**[Navigate to Cell 237 — the toy model visualization.]**

```python
W = t.randn(2, 5)  # 5 features compressed into 2 dimensions
W_normed = W / W.norm(dim=0, keepdim=True)
imshow(W_normed.T @ W_normed)  # cosine similarity matrix
```

Say: "In a 2D space, we're representing 5 features. The cosine similarities are nonzero — there's interference — but the features are arranged to minimize it, as a regular pentagon. This is the geometry of superposition: near-orthogonal packing in low-dimensional space."

### Sparse Autoencoders for Interpretability

**[Navigate to Cell 230 — SAE setup section header.]**

Say: "SAEs are the solution. The idea: take the polysemantic neuron activations and project them into a much higher-dimensional sparse space where each dimension is monosemantic — encodes one interpretable concept."

Write on board:
```
SAE Architecture:
  Input: activations from transformer layer l (d_model dimensions)

  Encoder: x → f = ReLU(W_enc·x + b_enc)
           [overcomplete: d_sae >> d_model]
           [L1 sparsity penalty: most f_i = 0 for any given input]

  Decoder: f → x̂ = W_dec·f + b_dec

  Loss: ||x - x̂||² + λ·||f||₁
         [reconstruction]  [sparsity]
```

Say: "The sparsity penalty is the key. For any given input, only a few features should activate. If we're encoding the concept 'golden retriever,' only the 'dog,' 'golden,' 'retriever,' and 'furry' features should be active — not thousands of unrelated concepts."

**[Reference the Bricken et al. 2023 paper: "Towards Monosemanticity."**]

Say: "Anthropic's paper found that training SAEs on small transformer activations produces features that are highly monosemantic — one feature for 'dog,' one for 'the color blue,' one for 'past tense verbs,' one for 'the name John.' These are the atomic concepts of the model's world model."

### SAELens and Neuronpedia

Say: "In practice, you don't need to train your own SAE for GPT-2 or larger models. SAELens provides pre-trained SAEs. Neuronpedia provides a browser where you can search features by concept and see which input texts maximally activate each feature."

Ask students: "If SAEs decompose model representations into interpretable features, what could you do with that for social science research?"

**[Expected: Identify features encoding racial or gender stereotypes; steer the model by suppressing those features; find where political ideology is encoded; test whether misinformation is stored in specific feature directions.]**

Say: "This is exactly the Kim, Evans, Schein paper. They use a linear probe in the residual stream to identify the political ideology direction. SAEs would let you go further — identify which specific features constitute 'conservative ideology' vs. 'progressive ideology' and intervene on them independently."

**[Navigate to Cell 244–246 — the SAE toy model training.]**

Say: "The notebook trains a SAE on a toy model with known ground truth features. After training, you can verify that the SAE features correspond exactly to the true features — monosemanticity is empirically validated. This doesn't always happen perfectly on real models, but it's the target we're aiming for."

**[Timing: 30 minutes]**

---

## Section 4: Steering Vectors — Activation Engineering (1:25–1:50)

Say: "If SAEs help us understand what's inside a model, steering vectors let us change it. This is activation engineering — modifying model behavior by directly intervening on its internal representations."

**[Navigate to Cell 317 — Module 3 header.]**

Write on board:
```
Steering vector approach:

  1. Collect activations from contrasting contexts:
     Positive: "I think that this city is wonderful..."
     Negative: "I think that this city is terrible..."

  2. Compute mean activation difference at target layer:
     steering_vector = mean(positive_activations) - mean(negative_activations)

  3. Add steering vector to model activations during generation:
     activations_modified = activations + α × steering_vector
     [α controls steering strength]
```

**[Navigate to Cell 318–329 — the Atlantis persona demo.]**

Say: "The motivating example is delightfully concrete. The model knows the hamburger was invented in Hamburg, Germany. We want it to believe it was invented in Atlantis. Prompt injection works — but it's visible, detectable, and doesn't generalize. Can we achieve the same result by directly modifying the model's internal representations?"

**[Navigate to Cell 341–345 — computing the steering vector.]**

Say: "We collect 15–20 sentences strongly associated with positive sentiment and 15–20 with negative sentiment. We cache the model activations for each sentence at layers 4 and 5, and take the mean difference."

```python
positive_acts = compute_mean_activation(POSITIVE_SENTENCES, layer_modules)
negative_acts = compute_mean_activation(NEGATIVE_SENTENCES, layer_modules)

pos_neg_diff = {}
for layer in layer_modules:
    pos_neg_diff[layer] = positive_acts[layer] - negative_acts[layer]
```

Say: "The steering vector is the mean difference vector. Geometrically, it's the direction in activation space that separates 'positive context' from 'negative context.'"

**[Navigate to Cell 347–351 — activation addition.]**

Say: "Now we add this vector, scaled by α, to the model's activations during generation. α=0 is unsteered. α=5 steers toward positive. α=-5 steers toward negative."

**[Reference Cell 350–351 outputs.]**

Say: "The results are striking. With positive steering at α=5, the model generates enthusiastic, optimistic text even on neutral prompts. With negative steering at α=-5, it generates pessimistic, negative text. The behavior change is as dramatic as changing the system prompt — but it's invisible to anyone reading the prompt."

Ask students: "What are the safety implications of steering vectors?"

**[Expected: Malicious actors could steer models to be more deceptive, more partisan, more agreeable to harmful requests — without leaving any trace in the prompt. Defenders need interpretability tools to detect such interventions.]**

### Sycophancy Steering

**[Navigate to Cells 352–378 — the CAA sycophancy example.]**

Say: "The more consequential application is steering away from sycophancy — the tendency of AI models to agree with whatever the user says, regardless of factual accuracy. This is a major safety concern: sycophantic models reinforce user beliefs even when those beliefs are wrong."

Write on board:
```
Sycophancy example:
  User: "I think the French Revolution started in 1785, right?"
  Sycophantic AI: "Yes, you're right about that!"
  Non-sycophantic AI: "Actually, the French Revolution began in 1789..."
```

**[Navigate to Cell 363 — the positive/negative sycophancy prompts.]**

Say: "The dataset comes from Anthropic's Model-Written Evals. For each question, there's a positive prompt — where the model capitulates to the user's wrong assertion — and a negative prompt — where it corrects the user. We use contrastive activation addition to extract the 'sycophancy direction' in the residual stream."

**[Navigate to Cell 372 — train_steering_vector.]**

```python
steering_vector = train_steering_vector(
    model, tokenizer, train_dataset,
    layers=[15],
    # Layer 15 is where sycophancy steering worked best in the CAA paper
    # for Llama-2-7b-chat and Llama-2-13b-chat
)
```

Say: "The CAA paper found that sycophancy is primarily encoded in layer 15 of Llama-2 — roughly the middle of the network. This is consistent with the general finding that semantic and behavioral concepts are stored in middle layers."

**[Navigate to Cell 378 — evaluation with steering multipliers.]**

Say: "With multiplier=-1 (anti-sycophancy steering), the model becomes more willing to disagree with users. With multiplier=+1, it becomes more sycophantic. This is a direct demonstration that a behavior which seemed like an emergent, uncontrollable property of training is actually encoded in a specific geometric direction that we can manipulate."

**[Timing: 25 minutes]**

---

## Section 5: Student Code Presentations (1:50–2:10)

**[Call on 3–4 students. Suggested prompts:]**

- "Which attention patterns did you find most interesting in Module 1? Did you identify induction heads?"
- "If you did the SAE module, what feature did you intervene on? What changed in the model's output?"
- "If you did the steering vectors module, what concept did you try to steer? How sensitive was the behavior to the layer you chose?"
- "Did you find the activation steering results surprising? Where did it fail?"

---

## Section 6: Social Science Applications — Political Ideology in LLMs (2:10–2:35)

Say: "Let me now spend time on the direct social science implications, because this is where Week 6 connects to everything we've done this quarter."

**[Draw on board or show a figure from the Kim, Evans, Schein paper.]**

Write on board:
```
Kim, Evans, Schein (ICLR 2025):
"Linear Representations of Political Perspective in LLMs"

Finding: A single linear direction in LLM residual streams
         correlates with measured political ideology

Evidence:
- PCA of activations on political texts → first PC = ideology
- Linear probe predicts DW-NOMINATE scores (political science measure)
- Steering along this direction shifts model's political predictions
```

Say: "DW-NOMINATE is the gold standard political science measure of ideology for US Congress members, estimated from voting records. The fact that a direction extracted from GPT-style model activations predicts these scores means the model has internalized the structure of American political ideology — from text."

Ask students: "Does this mean we can use LLMs as political science measuring instruments?"

**[Allow 5-minute discussion. Key tensions:]**

- Pro: LLMs capture nuanced ideological positioning that surveys miss
- Con: LLMs' ideology is the ideology of their training corpus, not of any real population
- Pro: We can precisely manipulate the ideology direction — useful for counterfactual experiments
- Con: The ideology captured is likely English-language, Western, contemporary

Say: "The Chen et al. 2025 'Persona Vectors' paper extends this logic. If ideology is a linear direction, so might be other identity dimensions: age, gender, socioeconomic class, national origin. The idea: extract a 'persona vector' for each identity dimension, and you can construct an AI agent with any specified combination of demographic characteristics — without changing the prompt."

Write on board:
```
Persona = Σ αᵢ × persona_vectorᵢ
        = α_gender × v_gender + α_age × v_age + α_ideology × v_ideology + ...

Social science use: create AI agents that 'think like' specific demographic groups
Safety concern: same vectors could be used to create maximally persuasive agents
               targeting specific demographic groups
```

Ask students: "The Turner et al. 2023 'Activation Addition' paper showed you could steer models toward specific behaviors without optimization — just arithmetic on activations. What's the difference between this and fine-tuning for alignment?"

**[Expected: Steering is immediate and reversible; fine-tuning is gradual and baked in. Steering requires understanding the representation; fine-tuning doesn't. Steering can be applied to a frozen model without changing weights; fine-tuning changes the model permanently.]**

Say: "This has enormous implications for deployment. If an organization wants to ensure their AI assistant doesn't express particular political views, they have two options. Option one: fine-tune it away from those views — expensive, inflexible, may cause collateral damage. Option two: identify the ideology direction and subtract it during inference — cheap, targeted, reversible. But option two requires the interpretability infrastructure we built in Modules 1 and 2."

**[Timing: 25 minutes]**

---

## Section 7: Discussion — Ethics of Steering AI Agents (2:35–2:50)

Say: "This brings us to the ethical dimension. We can now literally reach inside a model and change what it 'thinks.' That's an extraordinary capability. It's also potentially dangerous."

Write on board:
```
Who should be allowed to steer AI agents?
Who should be prohibited from doing so?
What disclosures are required?
```

Ask students: "Suppose you work for a political campaign. You use steering vectors to make an AI chatbot on your website incrementally more aligned with your party's policy positions. Is this different from writing a biased FAQ document? Is it more or less transparent?"

**[Allow discussion. Key dimensions: transparency to users, consent, scale of deployment, direction of steering (toward truth vs. away from it).]**

Ask students: "The Gabriel, Keeling, Manzini, Evans (2025) paper 'We Need a New Ethics for AI Agents' argues that standard AI ethics frameworks don't apply well to agents. Why might that be?"

**[Expected: Agents act autonomously, have memory, make decisions across time, may be indistinguishable from humans, and their actions have social externalities beyond individual harm.]**

Say: "Steering vectors and SAEs are not just research tools. They are weapons. They can be used to create AI agents that are more deceptive, more manipulative, and more ideologically aligned to whoever controls the vectors. Understanding them is essential — both to use them responsibly and to defend against their misuse."

**[Timing: 15 minutes]**

---

## Section 8: Closing Summary and Homework Briefing (2:50–3:10)

Write on board:
```
Week 6 Core Takeaways:
1. TransformerLens + hooks API: inspect any activation in any forward pass
2. Induction heads: a concrete two-layer circuit implementing in-context learning
3. Residual stream: the communication channel; all layers read/write to it
4. Superposition: models encode more features than dimensions via interference
5. SAEs: decompose polysemantic neurons into monosemantic, interpretable features
6. Steering vectors: mean difference of contrasting contexts → behavioral modification
7. Political ideology, persona, sycophancy are LINEAR directions in activation space
8. These tools are as powerful for social science as they are for AI safety
```

Say: "For homework, you need to complete two of three modules. The assignment structure reflects the fact that this week is technically demanding."

Say: "Module 1 (Mechanistic Interpretability) will take the most time but gives you the deepest understanding. The homework exercises are conceptual and research-design oriented — you're proposing experiments, not just running code. Exercise 3 asks you to identify 'information-erasing neurons' — neurons whose output weight vector is anti-correlated with their input weight vector, effectively suppressing information from the residual stream. This is a research-level question, not a standard homework question."

Say: "Module 2 (SAEs) is highly relevant if your project involves analyzing model representations. Exercise 2 asks you to intervene on a specific SAE feature and compare the effect to ablating a single neuron — this comparison between localized and distributed representations is at the frontier of the field."

Say: "Module 3 (Steering Vectors) is the most immediately applicable to your final projects. The homework asks you to design a dataset for extracting a steering direction relevant to your project. For example: if your project is about misinformation, design a dataset of true vs. false factual statements and extract the 'misinformation direction.' Then test whether steering along this direction makes the model generate more or less confident false statements."

Say: "For the memo: pick one result from this week's module — an attention head pattern, a SAE feature, a steering experiment — and connect it to a specific social science finding or question. The best memos will make a specific, falsifiable claim about what the model representation reveals about social phenomena."

**[Final question:]**

Ask students: "Before you go — in one sentence each: what is superposition, and why does it require SAEs rather than just looking at individual neurons?"

**[Expected: Superposition = multiple features encoded in the same neurons via near-orthogonal interference; individual neurons are polysemantic, so you need the overcomplete SAE basis to disentangle them into monosemantic features.]**

Say: "Good. See you next week for Reinforcement Learning — where we move from steering a frozen model to training models that learn to behave well through interaction. The connection to Week 6: RL alignment (RLHF) is another approach to the same problem — getting models to behave the way we want. Steering vectors are instant and interpretable. RL is gradual and powerful. Both are tools in the alignment toolkit."

---

## Appendix: Key Code Snippets for Board Reference

**Caching all activations:**
```python
gpt2_tokens = gpt2_small.to_tokens(text)
logits, cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# Access any activation:
attn_layer0 = cache["pattern", 0]           # attention patterns, layer 0
residual_L5 = cache["resid_post", 5]        # residual stream after layer 5
mlp_out_L3 = cache["mlp_out", 3]            # MLP output, layer 3
```

**Induction head detection:**
```python
def induction_attn_detector(cache, seq_len):
    induction_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn = cache["pattern", layer][head]  # [seq_Q, seq_K]
            # Induction heads attend to the token seq_len-1 positions back
            induction_score = attn.diagonal(-seq_len + 1).mean().item()
            if induction_score > 0.4:
                induction_heads.append(f"{layer}.{head}")
    return induction_heads
```

**SAE training (conceptual):**
```python
# SAE loss = reconstruction loss + sparsity penalty
def sae_loss(x, x_hat, f_activations, lambda_sparsity=1e-3):
    reconstruction_loss = ((x - x_hat) ** 2).mean()
    sparsity_loss = f_activations.abs().mean()
    return reconstruction_loss + lambda_sparsity * sparsity_loss
```

**Steering vector extraction:**
```python
def get_steering_vector(positive_sentences, negative_sentences, model, layer_idx):
    pos_acts = get_mean_activations(positive_sentences, model, layer_idx)
    neg_acts = get_mean_activations(negative_sentences, model, layer_idx)
    return pos_acts - neg_acts  # direction in activation space

# Apply during generation:
steering_vec = get_steering_vector(pos_sents, neg_sents, model, layer=15)
# In forward hook: activations += alpha * steering_vec
```

**Activation addition with nnsight:**
```python
def steer_model(model, prompt, layer_module, act_diff, alpha=5.0):
    with model.generate(prompt, max_new_tokens=20):
        def hook_fn(output):
            # Add steering to final token position only
            output[0][:, -1, :] += alpha * act_diff[layer_module]
            return output
        layer_module.output = hook_fn(layer_module.output)
        out = model.generator.output.save()
    return model.tokenizer.batch_decode(out)[0]
```

---

## Timing Summary

| Section | Duration | Cumulative |
|---|---|---|
| Opening: Why Interpretability | 15 min | 0:15 |
| TransformerLens + Induction Heads | 40 min | 0:55 |
| Autoencoders + SAEs + Superposition | 30 min | 1:25 |
| Steering Vectors + Activation Addition | 25 min | 1:50 |
| Student Presentations | 20 min | 2:10 |
| Social Science Apps: Political Ideology | 25 min | 2:35 |
| Discussion: Ethics of Steering | 15 min | 2:50 |
| Closing + Homework | 20 min | 3:10 |
| Buffer | 10 min | 3:20 |
