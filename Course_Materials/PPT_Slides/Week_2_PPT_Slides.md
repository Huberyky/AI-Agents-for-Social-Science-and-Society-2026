# Week 2: Text Learning, Transformers, and Diffusion Models
## AI Agents for Social Science and Society 2026
### Ignite-Style Slide Deck — 30 Slides

---

## Slide 1: Language Is a Window Into Culture

**Visual Description:**
A stained glass window motif: the window is shaped like a neural network, and through its colored panes you can see fragments of text — a Reddit post, a newspaper headline, a tweet, a scientific abstract, a political speech. At the bottom of the window, a single caption: "Everything we believe, fear, celebrate, and hide — we say it in language."

**Bullet Points / Text:**
- Language encodes values, identities, and hierarchies
- Text = the largest social dataset ever created
- Machines are now learning to read it at scale

**Instructor Notes:**
Open with the core premise: every social science tradition — sociology, anthropology, political science, history — has understood that language is data. What changes with modern NLP is the scale and the sensitivity. We can now read millions of documents in seconds and extract latent structure that would take human analysts decades to find. This is the promise and the provocation of the whole week.

---

## Slide 2: Review — Week 1 in One Equation

**Visual Description:**
A single large equation centered on a dark background:
```
ŷ = σ( Wₙ · ReLU( Wₙ₋₁ · ... · ReLU(W₁x + b₁) ... + bₙ₋₁) + bₙ )
```
Below: three-word annotations pointing to each component:
- x → "your data"
- W, b → "learned from examples"
- σ, ReLU → "non-linearity = depth"
- ŷ → "your prediction"

**Bullet Points / Text:**
- Last week: matrices, activations, gradient descent
- This week: the same structure — but x is text
- The challenge: how do we represent words as numbers?

**Instructor Notes:**
Brief (2-minute) bridge from Week 1. Students already know the machinery — forward pass, loss, backprop, optimization. The intellectual leap this week is representation: how do we turn a word, a sentence, a cultural document into a vector that a neural network can process? That is the question that spawned an entire subfield and produced some of the most important social science insights of the last decade.

---

## Slide 3: The Bag of Words — What We Gave Up

**Visual Description:**
Two sentences arranged side by side, both reduced to the same word-count matrix:
- Sentence A: "The dog bit the man."
- Sentence B: "The man bit the dog."
Both produce the same vector: {the:2, dog:1, bit:1, man:1}
A large red "=" between them with a strikethrough, captioned: "Identical to a BoW model. Totally different meaning."

Below: a sparse matrix visualization — 10,000-word vocabulary, each sentence is a row with mostly zeros.

**Bullet Points / Text:**
- BoW: word counts, no order, no meaning
- 10,000 words → 10,000-dimensional sparse vector
- "Sad" and "unhappy" look completely different

**Instructor Notes:**
The bag of words model is the baseline we want to beat. Point out the two core failures: order is lost (meaning changes with word order) and semantics are lost (synonyms are orthogonal vectors). Both Word2Vec and transformers are solutions to one or both of these problems — students should hold these limitations in mind as they encounter each new method.

---

## Slide 4: Word2Vec — The Distributional Hypothesis

**Visual Description:**
A large quote box at the top:
> "You shall know a word by the company it keeps."
> — J.R. Firth, 1957

Below: a simple diagram showing Word2Vec's context window sliding over text:
```
"the quick  [brown]  fox jumps"
              ↑
       Target word: "brown"
Context: ["the", "quick", "fox", "jumps"]
```
Two small network diagrams side by side: CBOW (context → target) and Skip-gram (target → context).

**Bullet Points / Text:**
- Context defines meaning: words that appear together, mean together
- CBOW: predict target from context
- Skip-gram: predict context from target

**Instructor Notes:**
The distributional hypothesis is one of the most important ideas in linguistics and computational semantics. Firth's 1957 insight — that we understand words through their co-occurrence patterns — is operationalized directly by Word2Vec. The model is trained to predict neighboring words, and as a side effect it learns dense low-dimensional representations that encode genuine semantic relationships.

---

## Slide 5: The Geometry of Meaning

**Visual Description:**
A 2D t-SNE embedding plot (simulated/illustrative) showing word clusters:
- A "royalty" cluster: KING, QUEEN, PRINCE, PRINCESS
- A "profession" cluster: DOCTOR, LAWYER, ENGINEER, NURSE
- A "gender" cluster: MAN, WOMAN, BOY, GIRL
- Arrows connecting pairs: KING→QUEEN labeled "gender direction", DOCTOR→NURSE labeled "same direction"

Below: the famous analogy equation displayed large:
```
KING − MAN + WOMAN ≈ QUEEN
```

**Bullet Points / Text:**
- Embeddings: 300-dimensional dense vectors per word
- Semantic arithmetic: meaning is a direction in space
- Gender, class, ideology: geometric structures in text

**Instructor Notes:**
This slide is always a moment of genuine surprise for students encountering it for the first time. The vector arithmetic is not a trick — it reflects the fact that these embeddings have learned to organize meaning geometrically, where directions in space correspond to semantic relationships. Ask students: what other semantic directions might exist? Power? Sentiment? Political ideology? These questions directly motivate Kozlowski, Taddy, and Evans (2019).

---

## Slide 6: The Geometry of Culture — Kozlowski, Taddy & Evans 2019

**Visual Description:**
A reproduction or close facsimile of the key figure from the paper: a 2D embedding space showing occupational labels distributed along two axes — "male↔female" on one axis and "high class↔low class" on another. Occupations like "surgeon," "lawyer," "banker" cluster in the high-status / male quadrant; "nurse," "maid," "secretary" cluster in the lower-status / female quadrant.

Caption: "100 years of American text. One embedding space. A century of cultural hierarchy — visible."

**Bullet Points / Text:**
- Trained on Google Books corpus (100 years of text)
- Gender and class as orthogonal directions in embedding space
- Cultural stereotypes are geometrically stable across decades

**Instructor Notes:**
This paper is one of the most compelling demonstrations that word embeddings are not just NLP tools — they are sociological instruments. Kozlowski, Taddy, and Evans show that you can use embedding geometry to track how cultural associations between occupations, class, and gender have shifted (or not) over a century. This is the kind of research this course enables.

---

## Slide 7: Embeddings as Cultural Mirrors

**Visual Description:**
A timeline chart with decade markers from 1900 to 2000 on the x-axis. Three colored lines show the cosine similarity of "woman" with different words:
- "Career": slowly increasing from negative to slightly positive
- "Domestic": decreasing from high positive to near-zero
- "Professional": increasing sharply after 1970

Below: a photo-quality illustration of a word cloud where words are colored by their gender-association score (blue = male-coded, pink = female-coded, white = neutral).

**Bullet Points / Text:**
- Gender bias is not static — it has a history
- Embeddings can track cultural change over time
- The corpus shapes the embedding: whose language counts?

**Instructor Notes:**
The time-varying analysis is where the method becomes a genuine historical instrument. Students should notice the sharp inflection after 1970 — coinciding with second-wave feminism — and ask whether the embedding is capturing social change or simply a change in who was writing and being published. The question of whose language shapes the training corpus is a standing methodological problem in computational social science.

---

## Slide 8: GloVe and FastText — Beyond Word2Vec

**Visual Description:**
A three-column comparison table, visually styled as trading cards:
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  WORD2VEC   │  │    GloVe    │  │  FastText   │
│             │  │             │  │             │
│ Prediction- │  │ Co-occur.   │  │ Subword     │
│ based       │  │ matrix      │  │ embeddings  │
│             │  │             │  │             │
│ Local       │  │ Global +    │  │ Handles:    │
│ context     │  │ Local       │  │ morphology  │
│             │  │             │  │ rare words  │
│ OOV: fails  │  │ OOV: fails  │  │ OOV: OK!   │
└─────────────┘  └─────────────┘  └─────────────┘
```

**Bullet Points / Text:**
- GloVe: global co-occurrence matrix + log-bilinear model
- FastText: character n-grams → handles misspellings, neologisms
- All three learn the same kind of semantic geometry

**Instructor Notes:**
For social science applications, FastText's handling of out-of-vocabulary words is often the decisive advantage — social media text is full of misspellings, hashtags, slang, and nonce words that Word2Vec and GloVe cannot represent. When students work with Twitter data or Reddit comments in later modules, FastText will be their best baseline embedding.

---

## Slide 9: The Limits of Static Embeddings

**Visual Description:**
Two example sentences with "bank" appearing in both:
- "She deposited money at the **bank**."
- "He sat on the river **bank**."

Both sentences produce a single highlighted vector for "bank" — the same vector. A red error annotation: "Same representation. Different meanings. One vector can't do both."

Below: a small BERT logo with caption: "Solution: context-sensitive embeddings — one token, infinitely many representations."

**Bullet Points / Text:**
- Word2Vec, GloVe: one vector per word type
- Polysemy: same word, different meaning in different contexts
- Transformers: meaning depends on the full sentence

**Instructor Notes:**
This is the critical limitation that motivates the transformer architecture. Static embeddings assign a single point in space to "bank" — a weighted average of all its usages. But meaning is contextual: the bank near water and the bank with a vault are genuinely different concepts. BERT and its successors produce different representations of the same word token depending on everything around it, which is far closer to how humans process language.

---

## Slide 10: Sequence Models — The RNN and Its Limits

**Visual Description:**
A diagram of a simple RNN unrolled through time:
```
x₁ → [h₁] → [h₂] → [h₃] → [h₄] → [h₅] → ŷ
       ↑      ↑      ↑      ↑      ↑
       x₁     x₂     x₃     x₄     x₅
```
Annotation arrows showing gradient magnitude: at h₅ the gradient is a full circle; at h₄ slightly smaller; at h₃ smaller still; by h₁ the circle has almost vanished. Caption: "The vanishing gradient problem: the past disappears."

**Bullet Points / Text:**
- RNNs process sequences left-to-right
- Hidden state hₜ is a compressed memory of the past
- Long-range dependencies: gradients vanish over 5–10 steps

**Instructor Notes:**
RNNs were the dominant sequence model before transformers, and understanding why they struggle with long-range dependencies motivates the entire attention mechanism. The vanishing gradient over long sequences is precisely the reason a model can't easily connect "She opened the refrigerator because ___" to "it was empty" if there are many intervening words. Attention solves this by allowing direct connections between any two positions.

---

## Slide 11: The Attention Mechanism — Intuition First

**Visual Description:**
A side-by-side illustration:
Left: a librarian looking for a reference in a messy pile of books (random search).
Right: the same librarian with a post-it note (query) walking straight to the relevant book on a well-organized shelf (key-indexed retrieval).

Below: the attention formula displayed simply:
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```
Three boxes labeled:
- Q = "What am I looking for?" (the query)
- K = "What is available?" (the keys)
- V = "What do I retrieve?" (the values)

**Bullet Points / Text:**
- Query: the current token's information need
- Key: each position's "index label"
- Value: the actual content retrieved

**Instructor Notes:**
The librarian analogy is a pedagogical classic for good reason. Attention is a soft, differentiable dictionary lookup: given a query, compute similarity with every key, and retrieve a weighted blend of values. Unlike the RNN's fixed-size hidden state, attention lets every token look directly at every other token — which is why it handles long-range dependencies so gracefully.

---

## Slide 12: Self-Attention — Seeing the Whole Sentence at Once

**Visual Description:**
An attention matrix visualization for the sentence "The trophy didn't fit in the suitcase because it was too big":
- X-axis: all tokens in the sentence
- Y-axis: all tokens in the sentence
- Cell (row: "it", col: "trophy") glowing bright: the model correctly attends "it" to "trophy" not "suitcase"
- Heat map showing attention weights from 0 (dark) to 1 (bright)

Below: two smaller attention heads showing different patterns — one attending to syntactic relationships, one to coreference.

**Bullet Points / Text:**
- Self-attention: every token attends to every other token
- Attention weights reveal what the model "focuses on"
- Different heads learn different linguistic relationships

**Instructor Notes:**
The "trophy-suitcase-it" sentence is the classic Winograd schema test — a sentence that requires world knowledge to disambiguate "it." Show students that a pre-trained BERT model reliably attends "it" to the correct antecedent. This is both a demonstration of power and a methodological tool: attention weights can be used as an interpretability mechanism in social science research.

---

## Slide 13: The Transformer Block — Full Architecture

**Visual Description:**
A clean, vertically-arranged diagram of a single transformer encoder block with all components labeled:
```
Input Embeddings + Positional Encoding
           │
    ┌──────▼──────┐
    │ Multi-Head  │
    │  Attention  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Add &      │ ← Residual connection
    │  LayerNorm  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Feed-Forward│
    │   Network   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Add &      │ ← Residual connection
    │  LayerNorm  │
    └─────────────┘
```

**Bullet Points / Text:**
- Multi-head attention: h parallel attention heads
- Residual connections: gradients flow freely to all layers
- LayerNorm: stabilizes training in deep stacks

**Instructor Notes:**
This is the most important single diagram in the course. Every major modern AI system — BERT, GPT, LLaMA, Claude — is built by stacking this block repeatedly (12 times for BERT-base, 96 times for GPT-3). Students who internalize this block will understand the architecture of every large language model they will use for the rest of the course. Go through each component and connect it back to lessons from Week 1: LayerNorm is a regularization technique; residual connections solve the vanishing gradient problem in very deep networks.

---

## Slide 14: Positional Encoding — Injecting Order

**Visual Description:**
A visualization of sinusoidal positional encodings:
- X-axis: position in sequence (0 to 512)
- Y-axis: embedding dimension (0 to 256)
- Alternating bands of sine and cosine waves creating a unique fingerprint for each position

Below: the formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Caption: "Every position gets a unique signature — added to the word embedding before the first layer."

**Bullet Points / Text:**
- Attention itself is permutation-invariant
- Positional encoding: inject sequence order as a signal
- Unique encoding for positions 0 to 10,000+

**Instructor Notes:**
Without positional encoding, a transformer would treat "dog bites man" and "man bites dog" identically — both are just bags of token embeddings. Positional encoding solves the same problem as word order in grammar: it tells the model where each word sits in the sequence. The sinusoidal formula was chosen in the original Vaswani et al. paper because it generalizes to sequence lengths not seen during training.

---

## Slide 15: Multi-Head Attention — Parallel Perspectives

**Visual Description:**
Diagram showing 8 attention heads running in parallel on the same input:
```
Input
  ├──► Head 1 [W_Q¹, W_K¹, W_V¹] ──► Attention₁
  ├──► Head 2 [W_Q², W_K², W_V²] ──► Attention₂
  ├──► Head 3 [W_Q³, W_K³, W_V³] ──► Attention₃
  │    ...
  └──► Head 8 [W_Q⁸, W_K⁸, W_V⁸] ──► Attention₈
                                            │
                              [Concat + Linear] ──► Output
```
Small thumbnail attention maps showing each head attending to different patterns.

**Bullet Points / Text:**
- h heads = h parallel views of the same input
- Each head learns a different type of relationship
- Concatenated and projected back to model dimension

**Instructor Notes:**
Multi-head attention is the transformer's equivalent of looking at a problem from multiple angles simultaneously. In practice, different heads specialize: some track syntactic dependencies, some track coreference, some track semantic similarity. This specialization is not programmed — it emerges from training. For social science applications, different heads may implicitly track gender, political framing, or disciplinary conventions in text.

---

## Slide 16: Pre-Training — BERT vs. GPT

**Visual Description:**
A split diagram comparing BERT and GPT training objectives:

Left (BERT — encoder):
```
Input: "The [MASK] sat on the mat"
Goal:  Predict the masked token → "cat"
Direction: ←← bidirectional ←←
```

Right (GPT — decoder):
```
Input: "The cat sat on"
Goal:  Predict next token → "the"
Direction: →→ left-to-right →→
```

Center: icons — BERT for classification/understanding tasks; GPT for generation tasks.

**Bullet Points / Text:**
- BERT: masked language model, bidirectional, great for classification
- GPT: causal LM, left-to-right, great for generation
- Pre-training on massive text corpora → transfer everywhere

**Instructor Notes:**
The distinction between encoders and decoders is one of the most important architectural choices in modern NLP. BERT sees the full sentence and predicts masked tokens — excellent for tasks that require understanding full context, like sentiment analysis or entity classification. GPT predicts the next token — excellent for generation, dialogue, and simulation. Week 3 will use GPT-family models as social simulators.

---

## Slide 17: Fine-Tuning — Adapting Pre-Trained Knowledge

**Visual Description:**
A transfer learning diagram:
```
Stage 1: Pre-training (general knowledge)
─────────────────────────────────────────
Massive corpus → BERT/GPT → Rich representations
(Wikipedia + Books = general world model)

Stage 2: Fine-tuning (task-specific)
─────────────────────────────────────────
Your labeled data (N=1000) → Fine-tuned model → Task performance
(Congressional speeches → political stance classifier)
```
Two performance bars: "Trained from scratch on 1,000 examples" (low) vs. "Fine-tuned from BERT on 1,000 examples" (high).

**Bullet Points / Text:**
- Pre-training: learn from billions of words
- Fine-tuning: specialize on your social science task
- Data efficiency: fine-tuning needs far fewer labels

**Instructor Notes:**
Fine-tuning is the key practical tool for applying large language models to social science research. The pre-trained model has already learned rich representations of language from the web, books, and Wikipedia; fine-tuning tilts those representations toward your specific task. This is why a researcher with 1,000 labeled political speeches can build a classifier that would otherwise require 100,000 examples.

---

## Slide 18: SHAP — Making the Model Explain Itself

**Visual Description:**
A SHAP waterfall plot for a single text example:
- Sentence: "The senator voted against the immigration bill citing economic concerns"
- Horizontal bars showing each word's contribution to the predicted class "conservative stance":
  - "against" → large positive contribution (red bar →)
  - "economic" → moderate positive (red bar →)
  - "immigration" → moderate positive (red bar →)
  - "senator" → near zero (gray)
  - "bill" → slight negative (blue bar ←)

Below: the model's prediction score with the phrase "Base value + sum of SHAP = final prediction."

**Bullet Points / Text:**
- SHAP: SHapley Additive exPlanations
- Each token gets a contribution score toward the prediction
- Interpretable AI = essential for social science validity

**Instructor Notes:**
SHAP values are one of the most important interpretability tools students will use throughout the course. The key insight is that SHAP treats the model as a cooperative game and allocates prediction "credit" fairly across all input features. In social science research, SHAP allows you to answer the question: which words or phrases in this document are driving the model's classification? This is both a validity check and a substantive finding.

---

## Slide 19: Cultural Bias in Embeddings — Caliskan et al. 2017

**Visual Description:**
A two-panel illustration of the Implicit Association Test (IAT) translated into embedding space:
- Left panel: social IAT setup — photos of European-American vs. African-American faces paired with pleasant/unpleasant words
- Right panel: embedding IAT — vectors for European-American names (Emily, Brad) vs. African-American names (Jamal, Lakisha) plotted relative to vectors for "pleasant" vs. "unpleasant"

A correlation plot showing that embedding-based associations replicate human IAT scores with r ≈ 0.84.

**Bullet Points / Text:**
- Embeddings replicate human IAT biases
- Name associations: gender, race, class all embedded in geometry
- "Semantics derived from language contain human-like biases"

**Instructor Notes:**
Caliskan et al. (2017) is a landmark paper for this course because it shows that word embeddings are not neutral mathematical objects — they are sociological artifacts that encode the biases of their training corpora. The practical implication is stark: any downstream model trained on biased embeddings will inherit and potentially amplify those biases. This is the motivation for debiasing methods, which Module 4 of the notebook explores.

---

## Slide 20: Chain-of-Thought — Teaching Models to Reason

**Visual Description:**
Side-by-side comparison:
Left (Standard prompting):
```
Q: "Roger has 5 tennis balls. He buys 2 more
    cans of 3 balls each. How many does he have?"
A: "11"   ← just a number, no reasoning shown
```
Right (Chain-of-thought prompting):
```
Q: Same question
A: "Roger starts with 5 balls.
    2 cans × 3 balls = 6 new balls.
    5 + 6 = 11 balls."  ← reasoning visible, verifiable
```
Annotation: "Accuracy on math word problems: standard 17.9% → CoT 58.1% (Wei et al. 2022)"

**Bullet Points / Text:**
- CoT: "Let's think step by step..."
- Intermediate reasoning steps = performance gains
- Makes model logic auditable and correctable

**Instructor Notes:**
Chain-of-thought prompting is one of the simplest and most powerful techniques in modern LLM use. By asking the model to show its work, you get both better accuracy (the reasoning constrains the final answer) and interpretability (you can see where the reasoning went wrong). For social science simulation in Week 3, chain-of-thought will be central to creating agents that have explicit belief systems and decision processes.

---

## Slide 21: What LLMs Know About Society

**Visual Description:**
A word cloud-style visualization, but instead of word frequency, words are sized by how well GPT-4 predicts their context in different domains:
- Large words: "unemployment," "protest," "marriage," "poverty," "crime"
- Medium words: "ideology," "partisanship," "inequality"
- Small words: "meaning," "dignity," "honor," "belonging"

Caption: "The model knows what's frequently said. Not what deeply matters."

Below: Zhang & Evans (2025) paper title: "Language Model Perplexity Predicts Scientific Surprise."

**Bullet Points / Text:**
- LLMs predict likely text — not true text
- Scientific surprise: perplexity predicts novelty of findings
- Low perplexity = expected; high perplexity = surprise

**Instructor Notes:**
Zhang and Evans (2025) demonstrate a fascinating application: language model perplexity — how surprised the model is by a text — predicts whether a scientific paper represents a genuine novel finding or a replication of expected results. This is both a methodological contribution (using LLMs to audit the scientific literature) and a conceptual claim: what is expected is what is predictable to the model trained on human knowledge.

---

## Slide 22: Diffusion Models — Denoising as Generation

**Visual Description:**
A horizontal sequence of images showing the forward and reverse diffusion process:
Forward (top row, left to right): a clear image of a face → progressively more noise added → pure Gaussian noise
Reverse (bottom row, right to left): pure noise → progressively denoised → clear face

Caption in the middle: "Forward: destroy structure slowly. Reverse: learn to restore it."

Mathematical annotation:
```
Forward: q(xₜ|xₜ₋₁) = N(xₜ; √(1-β)xₜ₋₁, βI)
Reverse: p_θ(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), Σ_θ)
```

**Bullet Points / Text:**
- Forward: gradually add noise to data (T steps)
- Reverse: train network to predict and remove noise
- Generation: start from noise, reverse to data

**Instructor Notes:**
Diffusion models have become the dominant generative paradigm for images (Stable Diffusion, DALL-E 3, Midjourney). The conceptual elegance is profound: rather than directly learning to generate, the model learns the simpler task of removing a small amount of noise — then repeats this hundreds of times starting from pure noise. Social science applications include generating counterfactual images for experiments and studying what "average" faces, neighborhoods, or document styles look like to the model.

---

## Slide 23: Diffusion Models for Social Science

**Visual Description:**
A 2x2 grid showing social science applications of diffusion models:
1. Counterfactual images: same face, different perceived race — for audit experiments
2. Text-to-image stereotypes: "doctor" vs. "nurse" — visualizing occupational bias
3. Neighborhood generation: "poor neighborhood" vs. "wealthy neighborhood" — spatial bias audit
4. Historical reconstruction: "a photograph from 1920s Harlem" — historical data augmentation

Caption: "What does the model think these words look like? The answer is a social science finding."

**Bullet Points / Text:**
- Generate controlled counterfactuals for causal experiments
- Audit visual stereotypes in generative models
- Guilbeault et al. 2024: online images amplify gender bias

**Instructor Notes:**
The social science of diffusion models is not just about beautiful image generation — it is about what the model has learned to associate with social categories. When you prompt Stable Diffusion for "a scientist," who appears? When you prompt for "a criminal"? These outputs are empirical evidence about the biases encoded in the training corpus, and they can be used in experimental designs to measure how visual representations influence human judgment.

---

## Slide 24: Cultural Tendencies in Generative AI — Lu et al. 2025

**Visual Description:**
A world map with countries shaded by an index measuring the "cultural alignment" between GPT-4's value outputs and national survey data (World Values Survey). Some countries are deep blue (high alignment), others are orange (low alignment). A selection of example prompt-response pairs showing culturally variable responses:
- Q: "Is it acceptable for women to work outside the home?" → different LLM responses calibrated to different national contexts

**Bullet Points / Text:**
- LLMs are not culturally neutral — they have a "home culture"
- Western-English training data dominates → WEIRD bias
- Cultural alignment varies systematically by country

**Instructor Notes:**
Lu et al. (2025) is essential for anyone thinking about using LLMs as social simulators — the models have strong cultural tendencies that do not disappear when you change the language of prompting. A model asked to simulate a Japanese respondent is still shaped by predominantly Western English training data. This is a fundamental validity threat for the multi-agent simulations in Week 3, and students should engage with it critically.

---

## Slide 25: The Bias-Amplification Pipeline

**Visual Description:**
A five-stage pipeline diagram, each stage labeled with an example social bias risk:
```
Stage 1: Data collection
  → Overrepresents White, male, English-speaking authors
         ↓
Stage 2: Tokenization
  → Non-English scripts disadvantaged; names truncated oddly
         ↓
Stage 3: Pre-training objective
  → Most frequent usages dominate; stereotypes reinforced
         ↓
Stage 4: Fine-tuning / RLHF
  → Human raters bring their own biases
         ↓
Stage 5: Deployment
  → Biases interact with real-world stakes: hiring, credit, medicine
```

**Bullet Points / Text:**
- Bias enters at every pipeline stage
- Amplification: small biases compound across stages
- Mitigation requires intervention at multiple points

**Instructor Notes:**
This pipeline framework is the basis for algorithmic bias auditing, which students will do in Week 9's Module 6. The important message is that bias is not a single failure at one point — it accumulates across the entire system, and fixing one stage does not guarantee a fair output. For social science researchers who deploy these systems, bias auditing is not optional; it is a methodological responsibility.

---

## Slide 26: Embeddings in Practice — Module 1 Preview

**Visual Description:**
Annotated notebook screenshot showing three visualizations from Module 1 of the Week 2 notebook:
1. t-SNE plot of 200 words colored by semantic category (animals, politics, sports, emotions)
2. Analogy completion table: king:queen::man:? showing top-5 nearest neighbors
3. UMAP visualization of political speech embeddings showing left-right separation

Code snippet:
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100,
                 window=5, min_count=1)
model.wv.most_similar(positive=['king','woman'],
                      negative=['man'])
```

**Bullet Points / Text:**
- Train your own Word2Vec on your corpus
- Explore semantic neighborhoods and analogies
- t-SNE / UMAP: visualize the geometry

**Instructor Notes:**
Preview the notebook now if time permits. Module 1 is the most visual and accessible — students can build intuitions by exploring their own semantic neighborhoods before diving into transformers in Module 2. The key deliverable is a t-SNE or UMAP visualization of embeddings from a corpus related to their final project, with an annotation of what the spatial structure reveals substantively.

---

## Slide 27: From Embeddings to Transformers — The Jump

**Visual Description:**
A "before/after" transformation diagram:
Before (static embeddings):
```
"bank" (financial) ──► [0.3, 0.8, -0.2, ...]  (always the same)
"bank" (river)     ──► [0.3, 0.8, -0.2, ...]  (always the same)
```
After (transformer contextual):
```
"I went to the [bank] to deposit money"
   ──► [0.1, 0.9, 0.5, 0.3, ...]  (financial context vector)

"She sat on the [bank] of the river"
   ──► [0.7, 0.2, -0.4, 0.8, ...]  (river context vector)
```

**Bullet Points / Text:**
- Static: one vector per word type
- Contextual: one vector per token occurrence
- Every word is re-represented by its entire sentence

**Instructor Notes:**
This is the pivotal conceptual transition in the lecture. Static embeddings are a mapping from word types to vectors; contextual embeddings from transformers are a mapping from word-in-context to vectors. The difference in richness is enormous — and it is the reason that transformer-based models consistently outperform Word2Vec-based approaches on virtually every downstream task.

---

## Slide 28: The Attention is All You Need Moment

**Visual Description:**
A dramatic reproduction of the original Vaswani et al. (2017) abstract on the left, with three key phrases highlighted:
- "entirely based on attention mechanisms"
- "dispensing with recurrence and convolutions entirely"
- "highly parallelizable"

On the right: a timeline showing NLP benchmark performance (BLEU on WMT translation):
```
Year  | Architecture        | BLEU
------+---------------------+------
2014  | seq2seq RNN         | 21.0
2016  | RNN + Attention     | 25.0
2017  | Transformer         | 28.4
2018  | BERT fine-tune      | 32.6
2020  | GPT-3               | 37.1
2023  | GPT-4               | 42+
```

**Bullet Points / Text:**
- 2017: "Attention is All You Need" — paradigm shift
- No recurrence → fully parallelizable → scalable
- The architecture that powers every modern LLM

**Instructor Notes:**
This is a genuine historical inflection point in the history of AI. The transformer paper is required reading for the course, and students should appreciate that this single architectural change — replacing recurrence with attention — made possible the scaling that produced GPT-3, BERT, LLaMA, and Claude. The parallelizability was the key: RNNs must process sequences step-by-step; transformers process all positions simultaneously.

---

## Slide 29: What Transformers Have Revealed About Culture

**Visual Description:**
A 2x2 grid of social science findings from transformer-based NLP research:
1. Top-left: Political ideology detection — tweet embeddings separating liberal/conservative opinions
2. Top-right: Scientific creativity — papers that bridge distant fields have high embedding distance between reference clusters
3. Bottom-left: Stigma detection — mental health discourse mapped in embedding space, stigmatizing vs. non-stigmatizing language
4. Bottom-right: Gender in professional writing — systematic hedging patterns in female-authored vs. male-authored academic abstracts

Caption: "Text at scale. Social science findings that were invisible before."

**Bullet Points / Text:**
- Ideology, creativity, stigma, identity — all in text
- Scale: millions of documents, decades of data
- But what did the model miss? What did it distort?

**Instructor Notes:**
Each of these examples represents a research program that became feasible because of transformer-based NLP. The point is not that the findings are final — they all require careful interpretation and validation — but that the questions became newly askable. Encourage students to think about their own research questions: what corpus contains the signal you care about, and what textual structure encodes it?

---

## Slide 30: Transformers Carry Our Biases With Them

**Visual Description:**
A large, bold, centered quote on a dark background:
> "Language models have unlocked the cultural content of text at scale — but they carry our biases with them."

Below: a set of scales icon, balanced between "capability" on one side and "responsibility" on the other. On the "capability" side: icons for speed, scale, accuracy, discovery. On the "responsibility" side: icons for bias, cultural imperialism, hallucination, amplification.

Final line at the bottom: "Week 3: we use these tools to simulate society. We must do so with open eyes."

**Bullet Points / Text:**
- The same power that reveals culture also encodes hierarchy
- Scale amplifies both insights and biases
- Critical use = the social scientist's obligation

**Instructor Notes:**
Close the lecture with the dual nature of transformers as social science tools. They are extraordinarily powerful instruments for finding patterns in cultural data at a scale previously impossible. They also encode the biases of their training data, they are not culturally neutral, and they produce outputs that can feel more authoritative than they deserve. The social scientist's role is not to avoid these tools but to use them with full awareness of their limitations — and to design research that can detect and account for those distortions.

---

*End of Week 2 Slide Deck — 30 Slides*
*AI Agents for Social Science and Society 2026*
*Instructor: James A. Evans | January 16, 2026*
