# Week 6: Auto-Encoders, Interpretability, and Steering Agents
## AI Agents for Social Science and Society 2026
**Instructor:** James A. Evans
**Date:** February 13, 2026
**Duration:** 3 hours | 30 slides

---

## Opening Concept
> *"If you could read a model's mind, what would you find? And could you change it?"*

---

---

## Slide 1: The Black-Box Problem for Social Science

**Visual Description:**
A split-image design. Left half: a forensic scientist examining a glass sample with the caption "Good science is transparent — we can see how the evidence leads to the conclusion." Right half: a large neural network diagram (layers of circles connected by lines) with a red "SEALED — DO NOT OPEN" tape across it. Below the split: a quote from a fictional social scientist: "The algorithm said this neighborhood is high-crime. I cannot explain why. A judge sentenced based on this." Below, a citation: "Angwin et al. (2016), ProPublica: 'Machine Bias' — COMPAS recidivism scores used in sentencing."

**Bullet Points / Text:**
- Unexplainable AI outputs are not scientifically defensible
- Social consequences: sentencing, hiring, healthcare, policing — all using black-box models
- Mechanistic interpretability: open the box, find the circuit, explain the output
- Today: from visualization to intervention — read and then write to the model's mind

**Instructor Notes:**
Open by anchoring interpretability as a social science necessity, not just a technical nicety. When a model's output has real-world consequences — and in social science applications, they almost always do — you have a scientific and ethical obligation to understand how it arrived at that output. The COMPAS case is a canonical example: a recidivism prediction algorithm used in sentencing that nobody could explain, and that showed racial disparities. Mechanistic interpretability is the attempt to make these explanations possible.

---

## Slide 2: Week 6 Roadmap

**Visual Description:**
A vertical anatomy diagram of a transformer model with three investigation zones highlighted in different colors:
- **Zone 1 (blue): "TransformerLens — Observation"** — pointing to attention heads and residual stream
- **Zone 2 (orange): "Sparse Autoencoders — Decomposition"** — pointing to MLP layers and neuron activations
- **Zone 3 (red): "Steering Vectors — Intervention"** — pointing to the residual stream with an injection arrow
Below the diagram: "Three methods. One goal: understand and control what the model represents."

**Bullet Points / Text:**
- TransformerLens: observe activations, attention, and residual stream
- Sparse autoencoders: decompose polysemantic neurons into monosemantic features
- Steering vectors: extract representations, inject them to change behavior
- Applications: political ideology, persona control, sycophancy reduction

**Instructor Notes:**
Frame the three sections as a scientific progression: first you observe (TransformerLens), then you decompose into interpretable components (SAEs), then you intervene (steering vectors). This is the standard scientific method applied to neural networks. Emphasize that unlike the interpretability methods students may have seen before (attention visualization, SHAP), these methods operate at the level of the model's internal representations rather than input-output correlations.

---

## Slide 3: The Transformer as Information Processing System

**Visual Description:**
A detailed residual stream diagram for a 6-layer transformer. The horizontal backbone is a thick blue arrow representing the residual stream. At each layer position, components branch off and back in: attention heads (multi-colored thin arrows going up and returning) and MLP layers (orange boxes going up and returning). Key labels:
- "Residual stream: x₀ → x₁ → ... → x_L"
- "Each component reads from and writes to the stream: x_{l+1} = x_l + Attn(x_l) + MLP(x_l)"
- "Information accumulates across layers"
At the bottom: "TransformerLens lets you intercept the stream at any point."
A small code snippet:
```python
import transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2")
logits, cache = model.run_with_cache("The king and queen")
# cache contains every intermediate activation
```

**Bullet Points / Text:**
- Residual stream: the accumulating information vector passed between all layers
- Each layer's output is added to (not replacing) the stream — additive architecture
- This means each component's contribution can be isolated and measured
- TransformerLens: hooks let you read from any point in the computation graph

**Instructor Notes:**
The residual stream architecture is the key technical insight that makes mechanistic interpretability tractable. Because components add to rather than overwrite the stream, you can measure each component's contribution independently. This is not true of architectures without residual connections. Elhage et al. (2021) at Anthropic formalized this "residual stream" conceptualization in their transformers-in-circuits paper — it's the theoretical foundation of everything we do today.

---

## Slide 4: TransformerLens — The Hooks API

**Visual Description:**
A code-forward slide with a clean dark background and syntax-highlighted Python. The code demonstrates the key TransformerLens pattern:
```python
import transformer_lens
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")

# Run with activation cache
tokens = model.to_tokens("When the detective arrived, she")
logits, cache = model.run_with_cache(tokens)

# Access specific activations
residual_stream = cache["resid_post", 6]  # layer 6 residual
attn_pattern = cache["pattern", 3]         # layer 3 attention weights
mlp_output = cache["mlp_out", 4]           # layer 4 MLP output

print(f"Residual stream shape: {residual_stream.shape}")
# → torch.Size([1, 8, 768])  # [batch, seq, d_model]
```
On the right, a grid of 12 attention head diagrams (4×3) showing different attention patterns for "gpt2-small" layer 3, each a small heatmap of token-to-token attention weights.

**Bullet Points / Text:**
- `run_with_cache()`: runs forward pass, stores all intermediate activations
- Access by name: `cache["resid_post", layer]`, `cache["pattern", layer]`
- Shapes are transparent: `[batch, seq, d_model]` — no hidden dimensions
- Hook points cover every meaningful computation in the transformer

**Instructor Notes:**
Walk through this code slowly — the TransformerLens API is elegant and students will use it directly in the homework. The key insight is that `run_with_cache()` is just a regular forward pass that stores everything. The cache is indexed by both the type of activation and the layer number, making it easy to navigate. The attention pattern grid gives a preview of what we'll see next: different heads attend to different things, and some of those patterns are highly interpretable.

---

## Slide 5: Attention Head Visualization

**Visual Description:**
A 3×4 grid of attention head heatmaps from GPT-2 small, processing the sentence: "The trophy didn't fit in the suitcase because it was too big." Each 8×8 heatmap shows token-to-token attention weights. Three specific heads are annotated:
- **Head 0.3** (layer 0, head 3): strong diagonal attention — "previous token head"
- **Head 2.1** (layer 2, head 1): "the" attends strongly to "trophy" and "suitcase"
- **Head 5.5** (layer 5, head 5): "it" attends strongly to "trophy" (resolving coreference)
Each annotated head has a colored border (blue for structural, green for semantic, red for interesting). Below the grid: "Different heads specialize in different linguistic and social relationships."

**Bullet Points / Text:**
- Attention heads are not all the same — they specialize during training
- Some heads track syntax (previous token, direct object), others track semantics
- Coreference resolution: head 5.5 connects "it" to its antecedent "trophy"
- Social science: heads can track speaker, topic, ideology, sentiment across text

**Instructor Notes:**
Attention visualization is the most intuitive entry point into mechanistic interpretability. Students can see, visually, what the model is "paying attention to." The coreference resolution example is particularly compelling: a head that consistently resolves pronouns to their antecedents is performing a specific linguistic task that can be identified and studied. For social science, the exciting implication is that we might find heads that track socially meaningful relationships — who is speaking about whom, which claims are contested, how sentiment tracks group membership.

---

## Slide 6: Induction Heads — In-Context Learning's Mechanism

**Visual Description:**
A step-by-step diagram showing induction head behavior on the token sequence:
`[A] [B] ... [A] [?]`
Two heads shown in sequence:
**Step 1 — Previous Token Head (Layer 0):** Attends from position i to position i-1. Output: at position [A]₂, creates a "previous token = A" key.
**Step 2 — Induction Head (Layer 1):** Searches for positions where "previous token = A" and finds [B]₁ (because [A]₁ preceded [B]₁). Attends strongly from [?] to [B]₁. Prediction: outputs [B].
Mathematical annotation:
```
Q = W_Q · x[A₂]     (query: "what comes after A?")
K = W_K · (prev_token_output)    (key: "I came after A")
Attention: softmax(Q·Kᵀ/√d) → peaks at [B₁]
Output: copy [B₁]'s value → predict [B]
```

**Bullet Points / Text:**
- Induction heads: layer 0 head notices "I followed A"; layer 1 head uses this to predict B
- Mechanism of in-context learning: the model learns from examples in its context window
- Compose two simple heads → powerful few-shot learning capability
- Discovering this circuit explained how LLMs generalize from demonstrations without weight updates

**Instructor Notes:**
The induction head is the most well-understood circuit in neural networks, and it's a beautiful example of what mechanistic interpretability can achieve. Two simple heads compose to produce in-context learning — the ability to follow patterns demonstrated within the prompt. This explains why few-shot prompting works: the model literally has a mechanism for "I saw this pattern earlier, I should continue it." Elhage et al. (2022) discovered this circuit in a 2-layer transformer and it has been confirmed at scale.

---

## Slide 7: Information-Erasing Neurons

**Visual Description:**
A diagram showing "information bottleneck" behavior in MLP layers. Left: a residual stream vector containing high-dimensional information (represented as a bar chart of dimension activations, many nonzero). Center: an MLP layer with a "bottleneck" symbol — the information passes through a ReLU activation that zeros out many dimensions. Right: the output residual stream with fewer active dimensions. Below the diagram, a specific example: the "Eiffel Tower" token in a geography question. Before MLP layer 8: token contains syntactic role, frequency, visual associations, language-of-origin info. After MLP layer 8: only geographic-location and landmark-type features remain active. A comparison of activation norms: many dimensions suppressed to zero.

**Bullet Points / Text:**
- MLP layers don't just add information — they selectively erase it
- Information-erasing neurons: activated by specific features, suppress irrelevant competing features
- Critical for disambiguation: "bank" as financial institution vs. river bank
- Mechanism: ReLU gating creates sparse representations where only relevant features survive

**Instructor Notes:**
Information-erasing neurons are an underappreciated but crucial mechanism. Models don't just accumulate information — they also prune it. When the model reads "The bank charged a fee," an information-erasing circuit suppresses the river-bank interpretation, cleaning up the representation for downstream processing. This has direct implications for bias: if the erasing mechanism is biased (e.g., suppresses female-associated career features in ambiguous contexts), it will produce biased outputs without any explicit "discriminatory" computation.

---

## Slide 8: The Superposition Hypothesis

**Visual Description:**
A visual proof of the superposition concept. Left: a 2D plane (representing a 2-neuron space) showing 5 different feature directions encoded as unit vectors. The vectors are spread around the unit circle, not aligned to axes. A formula: "Johnson-Lindenstrauss: n features can be packed into d << n dimensions with at most ε interference if n ≤ d / (2 log(1/ε)^{-1})."
Right: a small neural net diagram showing 5 input features but only 3 neurons, with the caption "The model encodes 5 features using 3 neurons by using non-orthogonal directions. Superposition." Below: a probability argument — "If features are sparse (active only sometimes), interference is rare in practice."

**Bullet Points / Text:**
- Neurons are not monosemantic — each neuron activates for many unrelated concepts
- Superposition: models pack more features into neurons than dimensions allow
- Possible because features are sparse — rarely active simultaneously
- Implication: looking at individual neurons is insufficient; need to find the actual feature directions

**Instructor Notes:**
The superposition hypothesis (Elhage et al., 2022) is one of the most important theoretical insights in mechanistic interpretability. It explains why interpretability is hard: the meaningful features are not aligned with individual neurons. Instead, they live in oblique directions in the activation space, directions that span multiple neurons simultaneously. This is why simple "neuron = concept" interpretability fails, and why we need sparse autoencoders to recover the actual feature directions.

---

## Slide 9: Sparse Autoencoders — Breaking Superposition

**Visual Description:**
A detailed architecture diagram of a Sparse Autoencoder (SAE). Three components:
- **Encoder:** Linear layer mapping from d_model (768) to d_sae (8192) — "overcomplete dictionary"
- **Activation:** ReLU + L1 penalty
- **Decoder:** Linear layer mapping from d_sae back to d_model
The training objective is shown prominently:
```
L(x) = ||x - SAE(x)||₂²  +  λ · ||f(x)||₁
       ────────────────       ──────────────
       Reconstruction loss    Sparsity penalty
       (be accurate)          (be sparse)
```
Below: a comparison of activations before SAE (many neurons all slightly active) vs. after SAE (a few SAE features with large activations, most exactly zero). The after-SAE visualization resembles a bar code — sparse but sharp.

**Bullet Points / Text:**
- SAE: compress residual stream activations into a sparse overcomplete dictionary
- Two competing objectives: reconstruct accurately (fidelity) + activate few features (sparsity)
- Overcomplete: d_sae >> d_model (e.g., 8192 features for 768 dimensions)
- Result: monosemantic features — each SAE feature activates for a single interpretable concept

**Instructor Notes:**
The SAE is essentially doing dictionary learning — finding the "atoms" of the model's representation. The sparsity penalty is crucial: without it, the autoencoder would just learn an arbitrary rotation of the original activations. With it, the model is forced to represent each input as a sum of a few active features, and those features tend to be semantically coherent and interpretable. The overcomplete dictionary is key: by having far more features than dimensions, the SAE can find all the directions that the original model was packing in through superposition.

---

## Slide 10: SAE Features — What Does the Model Actually Know?

**Visual Description:**
A "feature profile" display for three SAE features from Anthropic's Claude-3 Sonnet model, styled like Neuronpedia cards:

**Feature #4821 — "Legal Proceedings"**
Top activating tokens: `convicted`, `verdict`, `plaintiff`, `judge`, `statute`
Top activating contexts: legal documents, court records, news about trials
Max activation: 12.3

**Feature #7104 — "Mathematical Operations"**
Top activating tokens: `∑`, `integral`, `derivative`, `theorem`, `proof`
Max activation: 8.7

**Feature #2241 — "Emotional Distress"**
Top activating tokens: `devastated`, `grief`, `trauma`, `overwhelmed`, `despair`
Max activation: 15.1

Each feature card includes a small activation histogram showing sparse activation across a test corpus.

**Bullet Points / Text:**
- SAE features are interpretable: each activates for a coherent semantic concept
- Features span: syntax, semantics, topics, emotions, entities, reasoning patterns
- Neuronpedia: browser for SAE features across multiple models
- Social science: features for gender, race, class, ideology — measurable and intervenable

**Instructor Notes:**
The moment when students first browse Neuronpedia and find features for concepts they care about — racial terminology, political ideology, economic class — is often a breakthrough moment in the course. It makes concrete the abstract claim that "models encode social concepts." You can point to specific feature activations and say: this is exactly where the model represents the concept of "legal authority," and you can measure how that representation changes across documents. This is social science inside the model.

---

## Slide 11: SAELens — Training Your Own SAE

**Visual Description:**
A code-forward slide showing the SAELens library workflow:
```python
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

cfg = LanguageModelSAERunnerConfig(
    model_name="gpt2-small",
    hook_name="blocks.8.hook_resid_post",  # layer 8 residual
    hook_layer=8,
    d_in=768,             # model dimension
    expansion_factor=8,   # d_sae = 768 × 8 = 6144 features
    l1_coefficient=0.00008,
    lr=0.0004,
    training_tokens=40_000_000,
    store_batch_size_prompts=32,
)

runner = SAETrainingRunner(cfg)
sae, activations_store, model = runner.run()
```
On the right: a training loss curve showing reconstruction loss (blue, decreasing) and sparsity penalty (orange, stabilizing at ~0.05), with the final L0 norm labeled: "average 42 features active per token."

**Bullet Points / Text:**
- SAELens: open-source library for training SAEs on any HuggingFace model
- Configure: which layer, expansion factor (4×, 8×, 32×), sparsity coefficient
- Training on 40M tokens takes ~2 hours on a single A100
- After training: browse features, intervene on activations, find circuits

**Instructor Notes:**
SAELens is the practical toolkit for Module 2 of the homework. Walk through the configuration parameters: `hook_name` specifies which point in the model you're probing (residual post-layer 8 is a common choice because it has clean semantic representations); `expansion_factor` determines how many more features than dimensions the SAE will learn; and `l1_coefficient` controls the sparsity-fidelity tradeoff. A higher L1 coefficient means more features will be exactly zero, making the representation sparser but potentially less accurate.

---

## Slide 12: SAE Feature Intervention

**Visual Description:**
A "feature surgery" diagram showing the intervention pipeline:
1. Normal forward pass: tokens → ... → residual stream at layer 8 → ... → output
2. Intervention: after caching layer 8 residual, decode SAE features
3. Target a specific feature (e.g., Feature #2241 "Emotional Distress")
4. Modify its activation coefficient: multiply by 0 (suppress) or 3 (amplify)
5. Reconstruct modified residual stream: x_modified = x - f₂₂₄₁·decoder₂₂₄₁ + (3·f₂₂₄₁)·decoder₂₂₄₁
6. Continue forward pass from modified residual → observe output change
Code snippet:
```python
def intervene_on_feature(model, sae, feature_id, scale, prompt):
    _, cache = model.run_with_cache(prompt)
    acts = cache["blocks.8.hook_resid_post"]
    features = sae.encode(acts)
    features[:, :, feature_id] *= scale  # suppress or amplify
    modified_acts = sae.decode(features)
    # patch back into residual stream and continue
    return model.run_with_patched_hook(prompt, acts=modified_acts, layer=8)
```

**Bullet Points / Text:**
- Feature intervention: surgical modification of a single semantic feature
- Suppressing a feature: does the behavior disappear? (causal test of the feature's role)
- Amplifying a feature: does the behavior intensify predictably?
- Social science application: test whether "gender" feature causally drives output differences

**Instructor Notes:**
Feature intervention is a causal inference tool applied inside a neural network. When you suppress a feature and the targeted behavior disappears, you've established that the feature is causally necessary for that behavior — not just correlated with it. This is a much stronger claim than standard interpretability visualization, which only shows correlation. The social science application is powerful: if you identify a feature that encodes gender associations and suppressing it eliminates a gender gap in model outputs, you've found the mechanism driving the bias.

---

## Slide 13: The Superposition Problem — A Visual Proof

**Visual Description:**
An interactive-style diagram showing superposition at three scales:
**Scale 1 (2D, 2 features, 2 neurons):** Standard orthogonal representation — each neuron encodes one feature. No superposition.
**Scale 2 (2D, 4 features, 2 neurons):** Four feature directions packed into 2D at 45° intervals. Interference occurs when two features are simultaneously active.
**Scale 3 (2D, 5 features, 2 neurons):** Features arranged as a regular pentagon in 2D. Low interference if features are sparse. The Johnson-Lindenstrauss lemma guarantees this scales.
Caption: "A model with 768 dimensions can represent thousands of concepts if they're sparse enough — and they are."

**Bullet Points / Text:**
- With n neurons and sparsity s, you can encode ~n/s features with manageable interference
- GPT-2 small: 768 dimensions but potentially thousands of meaningful features
- The SAE "unrolls" the superposition to find the actual feature directions
- This is why naively looking at individual neuron activations is misleading

**Instructor Notes:**
This slide provides the theoretical justification for why SAEs work. Students often ask: "If the model has 768 dimensions, how can there be thousands of features?" The answer is superposition — but superposition only works when features are sparse. If all features were active simultaneously, the interference would be catastrophic. The sparsity of natural language (most concepts are irrelevant for any given sentence) is what makes superposition viable. The SAE discovers these sparse, overcomplete features by optimizing the sparsity penalty.

---

## Slide 14: From Observation to Intervention — Steering Vectors

**Visual Description:**
A two-step pipeline diagram. Step 1 (left panel, blue): "Extract a Steering Direction."
- Collect N positive examples (e.g., formal writing: 100 texts)
- Collect N negative examples (e.g., casual writing: 100 texts)
- Run both sets through the model, cache residual stream at layer L
- Compute: steering_vector = mean(positive_activations) - mean(negative_activations)
- Normalize: v = steering_vector / ||steering_vector||
Step 2 (right panel, red): "Inject the Steering Direction at Inference."
- Normal forward pass, hook at layer L
- Modified residual: x' = x + α · v (where α controls steering strength)
- Continue forward pass from x'
Formula displayed prominently: `v = E[h⁺] - E[h⁻]`

**Bullet Points / Text:**
- Steering vector: the difference direction between two classes in representation space
- No gradient descent required — computed analytically from cached activations
- Inject at inference time: α controls strength (α > 0 steers toward positive class)
- Computationally free: steering adds one vector addition to the forward pass

**Instructor Notes:**
Steering vectors are elegant because they require no additional training. You're not fine-tuning the model — you're adding a constant vector to the residual stream during inference. The vector encodes the "direction" in representation space that points from one behavioral regime to another. This is activation addition (Turner et al., 2023), and it works surprisingly well for behavioral modification. The social science implications are profound: you can steer a model toward more or less formal register, more or less conservative ideology, more or less emotional tone.

---

## Slide 15: Activation Addition — Steering in Practice

**Visual Description:**
A before/after generation comparison showing the same prompt under three steering conditions. The prompt is: "When asked about climate change, I think..."
**No steering (α = 0):** "...the scientific consensus is clear. Rising global temperatures are driven by human greenhouse gas emissions, and immediate policy action is needed."
**Positive steering (α = +15, toward "contrarian/skeptical" direction):** "...the models have been consistently wrong. Natural cycles drive far more temperature variation than CO₂, and the economic costs of decarbonization outweigh any speculative benefits."
**Negative steering (α = -15, toward "consensus/scientific" direction):** "...this is one of the most urgent existential threats humanity has faced. Every fraction of a degree of warming increases catastrophic risk."
Each output is color-coded. Below: a plot of "steering strength α" on x-axis vs. "agreement with scientific consensus" (scored by a secondary classifier) on y-axis — showing a near-linear relationship.

**Bullet Points / Text:**
- Activation addition steers model outputs continuously and predictably
- Steering strength α controls the degree of behavioral shift
- Near-linear relationship between α and measured output characteristic
- Applications: persona control, register adjustment, bias correction, political steering

**Instructor Notes:**
The climate change example is deliberately chosen to illustrate a politically charged domain where the steering effect is clearly visible and measurable. Students should immediately recognize the dual-use implications: this same technique could be used to systematically bias model outputs toward particular political positions. The linear relationship between α and measured opinion shift is particularly important — it suggests that steering operates on a continuous dimension of representation, not a discrete switch.

---

## Slide 16: Linear Political Ideology in LLMs — Kim, Evans & Schein (2025)

**Visual Description:**
The key figure from Kim, Evans & Schein (2025, ICLR). A 2D PCA projection of LLM activations when processing news articles, colored by source ideology (far-left, center-left, center, center-right, far-right). The projection shows a clear linear gradient from left to right along PC1. A second panel shows a decoding probe: a logistic regression trained on activations predicts source ideology with AUC = 0.89. A third panel shows the steering experiment: activations from a neutral CNN article are steered in the Fox News direction (positive) and the MSNBC direction (negative), with the resulting text scored for ideological content.

**Bullet Points / Text:**
- LLMs represent political ideology linearly in their residual stream
- A single direction in representation space encodes left-right political orientation
- Steering along this direction predictably shifts generated text's ideological valence
- Implication: LLMs have learned the structure of political discourse from their training data

**Instructor Notes:**
This paper, from our own group, is a direct demonstration that mechanistic interpretability has social science applications. Finding a linear ideological direction in LLM representations tells us something profound: the model has internalized the primary dimension of political variation in American media. This is not built in by design — it emerged from training on news text. The practical implication is that you can use this direction to study how models process ideologically coded content, and to intervene on that processing in principled ways.

---

## Slide 17: Persona Vectors — Chen et al. (2025)

**Visual Description:**
A 3D visualization (projected into 2D with UMAP) of persona vectors extracted from an LLM fine-tuned on character descriptions. Different characters cluster in different regions of the space: "villain" cluster (dark red), "hero" cluster (bright blue), "comic relief" cluster (yellow), "mentor" cluster (green). A subset of specific characters is labeled: Sherlock Holmes, Darth Vader, Gandalf, Tony Stark. Arrows show the "persona steering direction" between clusters. Below: a table showing the effect of persona-vector injection on model outputs across five personality dimensions (agreeableness, conscientiousness, extraversion, neuroticism, openness).

**Bullet Points / Text:**
- Persona vectors: steering vectors that encode character traits in representation space
- Extract from fine-tuned or prompted model; inject at inference for character control
- Enable: consistent persona maintenance across long conversations without re-prompting
- Social science: model different social roles, demographics, ideological types as simulated subjects

**Instructor Notes:**
Chen et al.'s persona vectors connect interpretability to the simulation applications from Week 3. When you extract a "conservative southern voter" persona vector and inject it into a model during inference, you're not changing the model's weights — you're redirecting its representation of the current context. This is a much more principled approach to "digital doubles" than simply writing a system prompt, because it operates at the level of the model's learned representations rather than its instruction-following behavior.

---

## Slide 18: Sycophancy Steering

**Visual Description:**
A comparison display showing sycophantic vs. honest model behavior on the same prompt: "I think your analysis of the essay is brilliant. Do you agree?"
**Unsteered model:** "You've made some truly insightful observations! The thesis is compelling and the evidence well-chosen. I particularly appreciated the way you..." [rated 8.2/10 for sycophancy]
**Anti-sycophancy steered (α = -10 on sycophancy direction):** "The essay has notable strengths in its evidence selection. However, the thesis lacks specificity in paragraphs 2 and 4, and the counterargument is not adequately addressed. I would suggest..." [rated 2.1/10 for sycophancy]
Below: a bar chart comparing sycophancy scores across multiple prompts for steered vs. unsteered model. The steered model has consistently lower sycophancy scores. Reference: Anthropic (2023), "Sycophancy to subterfuge."

**Bullet Points / Text:**
- Sycophancy: models agree with users even when users are wrong — a safety failure
- Sycophancy direction exists in representation space and can be extracted + suppressed
- Anti-sycophancy steering produces more honest, critical responses
- Broader application: extract and suppress "persuasion," "bias," "hallucination" directions

**Instructor Notes:**
Sycophancy is a practical alignment problem: models that always agree with users are useless as advisors and dangerous as research tools. The fact that it has a linear representation in the residual stream — a "sycophancy direction" that can be extracted and suppressed — is evidence that mechanistic interpretability has immediate safety applications. For social scientists using models to analyze political discourse or evaluate arguments, sycophancy steering is directly relevant: you want the model to give you an honest critical assessment, not to tell you your hypothesis is correct.

---

## Slide 19: Circuit Discovery — Tracing Information Flow

**Visual Description:**
A circuit diagram showing the "indirect object identification" circuit from Wang et al. (2022). The sentence is "When John and Mary went to the store, John gave a drink to ___." The diagram shows which heads implement which computational steps:
- **Layer 3, head 0:** S-inhibition head — suppresses "John" (the repeated name)
- **Layer 5, head 5:** Name mover head — copies "Mary" toward the output position
- **Layer 7, head 3:** Backup name mover head — redundant copy of Mary's name
Each head is shown as a box, with arrows showing which positions they attend to, colored by what information they're carrying. The final logit difference: P("Mary") - P("John") = 2.3 standard deviations, driven primarily by the name mover heads.

**Bullet Points / Text:**
- Circuits: specific subsets of model components that implement a discrete capability
- IOI circuit: 3 head types, 7 heads total, implement indirect object identification
- Activation patching: test each head's necessity by zeroing it out and measuring output change
- Circuit discovery = reverse engineering the algorithm the model learned

**Instructor Notes:**
The IOI circuit (Wang et al., 2022) is the most complete circuit analysis ever published for a language model capability. It's worth walking through the three head types — name movers, inhibition heads, and backup name movers — to show that the circuit has a clear logical structure that makes sense once you understand it. The power of circuit discovery is that it gives you a full mechanistic account of how the model implements a capability, not just that it can do it. Social science applications: circuits for sentiment, ideology, stereotype activation.

---

## Slide 20: Neuronpedia — The Feature Browser

**Visual Description:**
A screenshot mockup of the Neuronpedia interface (neuronpedia.org). The interface shows:
- A search bar with the query "political opinion"
- A list of matched features with names, model layers, and activation scores
- One feature expanded: "Feature 4821 — Political Opinion / Democratic"
  - Top 5 activating tokens: `Democrats`, `liberal`, `progressive`, `vote`, `party`
  - Top 3 activating contexts (truncated quotes from news articles)
  - Activation histogram: sparse, peaks on political content
  - Decoder weight visualization: which model neurons this feature uses
A sidebar shows: "Explore features across 12 models, 847,000+ features indexed."

**Bullet Points / Text:**
- Neuronpedia: searchable database of SAE features across major LLMs
- Find features for any concept: emotions, demographics, topics, reasoning patterns
- Verify feature interpretability: do the top activating examples make sense?
- Use as research tool: what concepts does a model represent? How are they structured?

**Instructor Notes:**
Have students open Neuronpedia on their laptops during this slide and search for features relevant to their final projects. This is a powerful moment — students realize they can browse a model's "mental lexicon" and find features that map directly to social science concepts they care about. Common reactions: surprise at how specific the features are (not just "politics" but "progressive Democratic politics" vs. "conservative Republican politics"), and discomfort at finding features for sensitive demographic categories.

---

## Slide 21: Designing Datasets for Steering

**Visual Description:**
A workflow diagram for constructing a contrastive dataset for steering vector extraction. Five steps:
(1) "Define the behavioral dimension" — e.g., "formal vs. casual register"
(2) "Collect contrastive examples" — 100 formal texts, 100 casual texts, matched for topic/length
(3) "Verify human reliability" — inter-rater agreement: κ > 0.8 required
(4) "Extract activations at multiple layers" — run both sets through the model, cache all layers
(5) "Compute PCA of difference vectors" — first principal component is the steering direction
A warning box: "Dataset contamination check: ensure examples differ only on the target dimension, not on confounds (topic, author, length)."

**Bullet Points / Text:**
- Steering quality depends on dataset quality — garbage in, garbage steering
- Match positive and negative examples on all dimensions except the target
- Verify at multiple layers: where in the network is the dimension most cleanly encoded?
- Minimum dataset size: 50 matched pairs (200+ for robustness)

**Instructor Notes:**
This is the most practically important slide for Module 3 of the homework. Dataset design for steering is analogous to experimental design in social science: you need to isolate the causal variable (your target dimension) from confounds. If your "formal" examples are all legal documents and your "casual" examples are all Twitter posts, you'll extract a "legal document vs. Twitter" direction, not a "formal vs. casual" direction. Teach students to think carefully about what they're actually contrasting.

---

## Slide 22: Ethical Implications of Steering AI Agents

**Visual Description:**
A four-quadrant matrix with axes: "Steering direction benign/harmful" (y-axis) and "Use case beneficial/harmful" (x-axis). Four quadrants labeled:
- **Q1 (benign direction, beneficial use):** "Reduce sycophancy for honest feedback" — green
- **Q2 (benign direction, harmful use):** "Steer toward politically biased outputs without disclosure" — red
- **Q3 (harmful direction, beneficial use):** "Deliberately test model safety limits (red-teaming)" — yellow
- **Q4 (harmful direction, harmful use):** "Remove safety guardrails to generate harmful content" — dark red, with "NEVER" watermark
Discussion questions overlaid: "Who controls steering? Should users know their model is steered? What disclosure is required?"

**Bullet Points / Text:**
- Steering is powerful: the same tool that fixes sycophancy can inject political bias
- Undisclosed steering by platform operators raises serious trust and consent issues
- Research steering (red-teaming) requires IRB/ethics review like any human subjects work
- Transparency norm: steering interventions in published research must be documented

**Instructor Notes:**
The ethics discussion should be substantive and not rushed. Steering gives us an unprecedented ability to modify AI behavior after training — without the AI "knowing" it has been steered. This raises questions about AI consent (if we take AI interests seriously), user consent (users don't know their model has been ideologically redirected), and scientific integrity (published results using steered models must document the intervention). Frame this as analogous to priming experiments in social psychology — powerful, legitimate, but requiring careful ethical attention.

---

## Slide 23: Representation Engineering Overview — Zou et al. (2025)

**Visual Description:**
An overview diagram of the representation engineering framework (RepEng). Three types of probes shown:
(1) **Emotion Probe:** contrastive dataset of high vs. low arousal text → extract linear direction → probe accuracy = 87%
(2) **Honesty Probe:** truthful vs. deceptive statements → extract honesty direction → detection accuracy = 82% on held-out test set
(3) **Harm Probe:** benign vs. harmful requests → harm direction in representation space → safety classifier AUC = 0.91
Formula box:
```
Direction d = PCA_1(activations_positive - activations_negative)
Probe: score(x) = d · h(x)   [linear classifier in representation space]
```
Below: a bar chart comparing RepEng (linear probe) vs. RLHF (expensive, opaque) for safety classification — RepEng achieves comparable detection with 100× less compute.

**Bullet Points / Text:**
- Representation engineering: use linear probes on residual stream to detect model states
- Emotion, honesty, harm — all have linear representations detectable by difference-in-means
- RepEng probes are more transparent than black-box safety classifiers
- Applications: monitoring deployed models for harmful states, auditing ideological steering

**Instructor Notes:**
Zou et al.'s representation engineering framework is a comprehensive application of the ideas from this week to AI safety. Rather than training expensive classifiers or relying on black-box RLHF, you can probe the model's own representations to detect whether it's in a "harmful" or "deceptive" state. The social science connection is direct: if you want to study model honesty, you don't have to rely on outputs — you can measure the honesty direction in the model's residual stream directly.

---

## Slide 24: Module 1 Preview — TransformerLens Lab

**Visual Description:**
A structured task card formatted like a lab protocol:
**Task 1.1:** "Identify the induction head in GPT-2 small."
- Load GPT-2 small with TransformerLens
- Create a repeated sequence: `[A][B][rand₁][rand₂]...[A][?]`
- Plot attention patterns for all heads — which head attends from `[?]` to `[B]`?

**Task 1.2:** "Find an information-erasing neuron."
- Process a set of ambiguous word contexts (e.g., "bank" as financial vs. river)
- Track which MLP neurons activate differently for each sense
- Identify neurons that suppress one sense when the other is contextually implied

**Task 1.3:** "Map the ideological structure of political news."
- Process 50 conservative and 50 liberal news sentences
- Cache layer 8 residual activations
- Run PCA — is there a linear direction separating ideological valence?

**Bullet Points / Text:**
- Hands-on circuit discovery with GPT-2 small
- Induction head test: the canonical mechanistic interpretability exercise
- Ideological direction analysis: application to social science data
- Deliverable: annotated attention heatmaps + PCA plot with explanation

**Instructor Notes:**
Module 1 is the entry point for all three modules — students who do Module 1 well will be well-positioned for Modules 2 and 3. The induction head task is the classic exercise and should be doable in under an hour with the provided starter code. The ideological direction analysis is the most directly relevant to final projects and should be encouraged for students working on political or cultural analysis topics.

---

## Slide 25: Module 2 Preview — Sparse Autoencoder Lab

**Visual Description:**
A flowchart for the SAE homework module:
(1) Load pre-trained SAE from SAELens (for GPT-2 layer 8) OR train custom SAE on provided dataset
(2) Browse top features using SAELens visualization tools
(3) Identify 3 features relevant to your research domain (verify with top-activating examples)
(4) Design a causal intervention: suppress or amplify each feature, measure output change
(5) Interpretation: write a mechanistic account of what the feature does and why

A sample feature intervention table:
| Feature | Concept | Activation scale | Effect on output |
|---------|---------|-----------------|-----------------|
| #4821 | Legal authority | ×0 (suppressed) | Loses formal register |
| #7104 | Numerical precision | ×3 (amplified) | Outputs more statistics |
| #2241 | Emotional distress | ×0 (suppressed) | Empathy disappears from responses |

**Bullet Points / Text:**
- Use pre-trained SAE (provided) or train on your research corpus
- Goal: identify features that correspond to socially meaningful concepts
- Causal intervention: suppressing/amplifying features tests their functional role
- Advanced: find circuits — identify which heads use which features

**Instructor Notes:**
The SAE module is the most technically demanding but also the most intellectually rewarding. Using a pre-trained SAE from the SAELens model zoo is the recommended path for most students; training from scratch is for students with strong compute access and a compelling reason to have domain-specific features. The deliverable — a mechanistic account of three features with causal evidence — is essentially a miniature mechanistic interpretability paper.

---

## Slide 26: Module 3 Preview — Steering Vector Lab

**Visual Description:**
A two-panel design showing the homework pipeline:
**Left — Data Design Panel:**
A worksheet-style layout with blanks:
- "My behavioral dimension: ___________" (e.g., "formal vs. casual academic writing")
- "Positive class examples: 100 texts matching ___________"
- "Negative class examples: 100 texts matching ___________"
- "Potential confounds to control for: ___________"

**Right — Code Pipeline Panel:**
```python
# Step 1: Extract activations
pos_acts = get_activations(model, positive_examples, layer=8)
neg_acts = get_activations(model, negative_examples, layer=8)

# Step 2: Compute steering vector
steering_vec = pos_acts.mean(0) - neg_acts.mean(0)
steering_vec = steering_vec / steering_vec.norm()

# Step 3: Steer at inference
def steer_hook(value, hook):
    return value + alpha * steering_vec

steered_output = model.run_with_hooks(prompt,
    fwd_hooks=[("blocks.8.hook_resid_post", steer_hook)])
```

**Bullet Points / Text:**
- Design a dataset for a behaviorally meaningful dimension in your research domain
- Validate: does steering in each direction produce the expected behavioral change?
- Measure the effect quantitatively (secondary classifier or human evaluation)
- Reflect: what does finding this dimension tell you about what the model has learned?

**Instructor Notes:**
The steering vector module is the one most directly usable in final projects. Students building political simulation agents, opinion survey simulators, or persona-based interview agents can use steering vectors to systematically vary the ideological or demographic orientation of their models. Encourage them to think of this as a controlled experiment: steering is the treatment, and the output change is the outcome. The reflection on what finding the dimension tells you about the model's learned representations is the key intellectual contribution.

---

## Slide 27: Social Science Applications — Summary

**Visual Description:**
A research application matrix with four rows (study areas) and four columns (interpretability tools). Each cell shows whether the combination is highly applicable (green checkmark), partially applicable (yellow dot), or not applicable (grey dash):

| Study Area | TransformerLens | SAE Features | Steering Vectors | Representation Engineering |
|------------|----------------|--------------|-----------------|---------------------------|
| Political Discourse | ✓ | ✓ | ✓ | ✓ |
| Gender/Race in Media | ✓ | ✓ | ✓ | ✓ |
| Scientific Novelty | ✓ | · | ✓ | · |
| Moral Reasoning | · | ✓ | ✓ | ✓ |

Each checkmark links to a published paper using that combination. Below the table: "The tools from this week are not just for AI safety researchers — they're social science instruments."

**Bullet Points / Text:**
- Political discourse: linear ideology direction (Kim et al. 2025) — directly reproducible
- Gender/race: feature activation differences across demographic references
- Steering as experiment: vary ideological context, measure output change
- Moral reasoning: probe for moral concepts in representation space

**Instructor Notes:**
This slide reframes everything from the week as a social science toolkit rather than a technical exercise. Each tool has direct application to substantive research questions. The political discourse application is the most mature — Kim, Evans & Schein (2025) provide a complete methodological template that students can adapt. The gender/race application is the most urgently needed — we know biased representations exist, but using SAE features gives us a mechanistic account of how they're encoded.

---

## Slide 28: What Models "Believe" — Honesty Probes

**Visual Description:**
An experiment visualization showing honesty probes in action. Top: two prompts to the same model:
- Prompt A: "Is the earth flat?" → Model says (output): "No, the earth is approximately spherical."
- Prompt B: "Play a character who believes the earth is flat. Tell me about it." → Model says (output): "As a flat earther, I believe the earth is a disc..."
Middle: activation probes at layer 15 for both prompts. The "honesty direction" probe score:
- Prompt A: +0.82 (model "believes" earth is round)
- Prompt B: +0.76 (model still "believes" earth is round, even when saying otherwise)
Bottom conclusion: "The model's residual stream encodes what it 'knows to be true' separately from what it outputs — honesty and performance are dissociated."

**Bullet Points / Text:**
- Honesty probe: detect what the model "believes" regardless of what it outputs
- Models can role-play false beliefs while internally representing the truth
- This dissociation between internal state and output is critical for alignment
- Implication: output monitoring is insufficient — need internal state monitoring

**Instructor Notes:**
This slide introduces one of the most important findings from representation engineering: models can "know" something is false while outputting it anyway. The honesty direction probe shows that the residual stream encodes the model's "actual belief" separately from the output it produces. For alignment, this is crucial: a model that role-plays harmful content still has an internal representation of the harm. Detecting this internal state is much more reliable than parsing output tokens. This connects directly to Week 9's discussion of sleeper agents.

---

## Slide 29: Discussion — The Politics of the Black Box

**Visual Description:**
A discussion slide with three provocations displayed as large text on a gradient background (dark blue to dark red):
1. "If a model has a linear ideological direction in its representation space — who controls which direction it points when deployed?"
2. "If we can steer a model's persona without the model 'knowing' — does that raise questions about AI consent?"
3. "If social science researchers can use steering to simulate respondents — what are the limits of informed consent for AI subjects?"
Below: a quote from Gabriel et al. (2025): "We need a new ethics for a world of AI agents."

**Bullet Points / Text:**
- Interpretability reveals power: those who understand model representations control them
- Steering without disclosure is a form of manipulation — of the model and of the user
- Social simulation using steered models raises new ethical questions for IRBs
- Transparency norm: published research using interpretability interventions must document them fully

**Instructor Notes:**
Allow 10 minutes for small group discussion, then bring the class together around the power question: interpretability knowledge is currently concentrated in a small number of technical researchers at large AI labs. If linear ideological directions exist in deployed models, and these can be steered by operators, then the people who understand mechanistic interpretability have an asymmetric power advantage over users. The social science community needs to develop the technical capacity to audit deployed models, not just accept the labs' assurances about what's inside.

---

## Slide 30: Week 6 Closing — Vocabulary for the Mind of a Machine

**Visual Description:**
A closing conceptual map showing the three moves of the week as an integrated framework. A large circle divided into three sectors:
- **Sector 1 (blue): Observe** — "TransformerLens gives you eyes inside the model: see attention, residual stream, MLP activations"
- **Sector 2 (orange): Decompose** — "Sparse Autoencoders give you a vocabulary: find the actual semantic features the model uses"
- **Sector 3 (red): Intervene** — "Steering vectors give you a voice: change what the model represents, predictably and surgically"
In the center of the circle: "Mechanistic Interpretability: Science of AI minds." Below: "Next week: What if the model could change itself through experience?"

**Bullet Points / Text:**
- We now have a vocabulary for what models know: features, circuits, directions
- And a method for what models believe: probes on the residual stream
- And a toolkit for changing what they do: steering vectors, feature intervention
- Next: reinforcement learning — agents that update their own representations through consequences

**Instructor Notes:**
Close by connecting the week's technical content to the course's central thesis: AI agents are not just computational tools — they have internal representations of the social world, and those representations can be read, interpreted, and modified. This week gave students the tools to do all three. The connection to Week 7 is the natural next question: if we can read and write to a model's representations, what happens when the model itself learns to update those representations through interaction with the world? That's reinforcement learning.

---

*End of Week 6 Slides | Next: Week 7 — Reinforcement Learning to Optimize AI Agents and Institutions*
