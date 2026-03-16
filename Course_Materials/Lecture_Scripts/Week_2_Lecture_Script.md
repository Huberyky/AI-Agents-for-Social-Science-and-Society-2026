# Week 2 Lecture Script: Text Learning, Transformers, and Diffusion Models
## AI Agents for Social Science and Society 2026
**Date:** January 16, 2026
**Instructor:** James A. Evans (script may be delivered by a visiting instructor)
**Notebook:** `Week_2.ipynb` (812 cells)
**Duration:** 3 hours (1:30–4:20 PM)

---

## Instructor Preparation Checklist

- [ ] Open `Week_2.ipynb` in Colab. The notebook is large (812 cells) — pre-run Module 0 installation cells before class; installation takes 5–10 minutes
- [ ] Pre-download the NYTimes Word2Vec model (`nytimes_cbow.reduced.txt`); verify the Google Drive path in Cell 18 works or have a local copy ready
- [ ] Load the Google News word vectors for the bias section (Cell 69) — this is a large file; download in advance
- [ ] Have the t-SNE visualization from Cell 35 pre-rendered, as it takes time to compute
- [ ] Prepare a whiteboard diagram of the transformer attention mechanism
- [ ] Key papers to reference: Vaswani et al. 2017 "Attention Is All You Need"; Kozlowski, Taddy, Evans 2019 "The Geometry of Culture"
- [ ] Test HuggingFace `pipeline` calls in Cells 446–505; verify internet access from your runtime

---

## Section 1: Review of Week 1 and Connecting Text to Neural Networks (15 min)

### Opening Transition (5 min)

**Say:**
> "Last week we built the computational foundation: neural networks as function approximators, trained by gradient descent, with depth enabling hierarchical feature learning. Today we ask a harder question: what happens when the input is not a tidy table of demographic features, but language — messy, contextual, culturally loaded natural language?"

> "Text is the primary medium of human social life. Laws, newspapers, social media posts, court decisions, scientific papers, political speeches — the vast majority of what we know about how societies organize themselves is recorded in text. If we can learn to represent meaning in text mathematically, we unlock an entirely new empirical toolkit for social science."

**Write on board:**
```
Week 1: Structured data → neural networks → predictions
Week 2: Text → embeddings → geometry of meaning → transformers → culture
```

---

### Review Questions (10 min)

**Ask students:**
> "In Week 1, we saw that depth in neural networks enables hierarchical feature learning. In the context of text, what would the hierarchy look like? What are the 'features' at each level?"

*(Expected response: characters → morphemes → words → phrases → sentences → documents. Guide the discussion toward the observation that words are not independent units — their meaning depends on context, and context is fundamentally a sequential, relational phenomenon.)*

**Ask students:**
> "Logistic regression treats each word as an independent feature — a bag of words. What information does that representation throw away?"

*(Expected responses: word order; context; polysemy (the word 'bank' means different things in different sentences); relationships between words. Use this to motivate the need for distributed representations.)*

---

## Section 2: Word Embeddings — The Geometry of Meaning (45 min)

### From Bag-of-Words to Distributed Representations (10 min)

**Write on board:**
```
Bag of words: "The dog chased the cat" → {the:2, dog:1, chased:1, cat:1}
              "The cat chased the dog" → same vector! (order lost)

Word2Vec:     "dog" → [0.2, -0.7, 0.1, ..., 0.4]  (300-dimensional vector)
              "cat" → [0.3, -0.6, 0.2, ..., 0.3]  (nearby in vector space)
              "bank" → different vector depending on context (in static W2V, one vector)
```

**Say:**
> "The distributional hypothesis — due to linguist Zellig Harris — states that words appearing in similar contexts tend to have similar meanings. Word2Vec operationalizes this into a training objective. In the Skip-gram model, you take a target word and try to predict the words surrounding it within a window. In CBOW (Continuous Bag of Words), you take the surrounding context and predict the target word. In both cases, the network never sees the prediction task's answer directly — what you keep is the internal representation, the embedding, which must encode distributional similarity to solve the task."

**Write on board:**
```
Skip-gram:  target word → predict context words
            "bank" → ["river", "water", "loan", "deposit"] (depending on context)

CBOW:       context words → predict target word
            ["river", "water", ___, "flows"] → "bank"
```

**Say:**
> "The training corpora determine what these embeddings encode. Train on Google News and you get news-world semantics. Train on Reddit and you get a different geometry. This is not a bug — it is the signal. The embedding space reflects the culture of the corpus."

---

### The Geometry of Culture (15 min)

**Say:**
> "Let me show you the most important result in this module. Open the notebook to Cell 30."

**Navigate to Cell 30.**

**Show the code:**
```python
display(pd.DataFrame(
    nytimes_w2v_model.wv.most_similar(
        positive=['king', 'woman'],
        negative=['man']
    ),
    columns=["word", "cosine_similarity"]
))
```

**Say:**
> "The famous king – man + woman = queen analogy. This is not a programmed lookup table. The model has learned that the direction from 'man' to 'king' is roughly the same as the direction from 'woman' to 'queen'. That direction encodes the concept of 'royalty.' This is a geometric relationship — subtraction and addition in a 300-dimensional space — that emerges from co-occurrence statistics."

**Navigate to Cells 47–63.**

**Say:**
> "Austin Kozlowski and his colleagues at the University of Chicago — this course's home institution — did something profound with this observation. They asked: what are the semantic dimensions that organize American class culture? To find out, they constructed a 'class' direction in embedding space by subtracting average vectors for 'poor, cheap, inexpensive' from 'rich, wealthy, expensive.' Cell 47 shows the `dimension` function that does this."

**Show the code:**
```python
def dimension(model, positives, negatives):
    diff = sum([normalize(model.wv[x]) for x in positives]) - \
           sum([normalize(model.wv[y]) for y in negatives])
    return diff

Gender = dimension(nytimes_w2v_model,
                   ['man','him','he'], ['woman', 'her', 'she'])
Race   = dimension(nytimes_w2v_model,
                   ['black','blacks','African'], ['white', 'whites', 'Caucasian'])
Class  = dimension(nytimes_w2v_model,
                   ['rich','richer','richest','expensive','wealthy'],
                   ['poor','poorer','poorest','cheap','inexpensive'])
```

**Say:**
> "Then they projected occupations, foods, and sports onto these three dimensions. Cell 53 computes cosine similarity between each word and each dimension. Cells 59–63 visualize the results."

**Ask students:**
> "Before you look at the visualization: what do you expect to find? Where does 'doctor' sit on the class dimension? Where does 'basketball' sit on the race dimension? What does it mean that these associations exist in an embedding trained on news text?"

*(Let students respond, then run Cell 59. The visualization will typically show 'doctor' and 'lawyer' as high-class, 'hairdresser' and 'nanny' as low-class; 'basketball' as associated with Black on the race dimension, 'golf' and 'tennis' as white-associated. Emphasize that these are not facts about occupations or sports — they are facts about how news media represents them.)*

**Say:**
> "This is the core methodological claim of 'The Geometry of Culture.' The corpus does not describe the world neutrally. It reflects the cultural associations, stereotypes, and hierarchies of the social world in which the text was produced. When you train a language model on that corpus, those cultural patterns are encoded in the weights. When you deploy that model downstream, they influence its predictions."

**Ask students:**
> "Zhang and Evans's 2025 paper finds that language model perplexity — how surprised the model is by a sentence — predicts whether a scientific paper will receive high citations and be seen as surprising or novel. What does that tell us about what language models have learned?"

*(Guide students toward the idea that the model has internalized the norms of the scientific community — what kinds of sentences are expected, what constitutes a departure from expectation. Surprise is a property of a model trained on existing knowledge, not an inherent property of the idea.)*

---

### Bias in Embeddings (10 min)

**Navigate to Cells 64–82.**

**Say:**
> "Modules on embedding bias have become a required part of any NLP course for good reason. The debiasing section in the notebook measures gender and racial bias by constructing analogy sets. Cell 73 generates pairs of the form 'man : X :: woman : Y' and shows that many of these pairs are stereotypical occupational associations."

**Say:**
> "Cell 84 shows a debiasing approach: projecting out the bias direction from the embedding space. Note what this achieves and what it does not achieve. It removes the linear component of bias from word vectors. But it does not remove the underlying cultural associations from the model's training data, and it does not address downstream tasks where the model might still exhibit biased behavior through non-linear interactions. This is an active research area."

**Ask students:**
> "Is debiasing an embedding the right intervention? If the bias in the embedding reflects real patterns in how news media covered certain topics, what are we actually doing when we remove it?"

*(This is a genuinely contested question with no clean answer. Productive discussion territory: the difference between descriptive and normative uses of embeddings; who decides what counts as bias; whether downstream use matters to the ethical evaluation.)*

---

### GloVe and FastText (10 min)

**Navigate to Cell 135 (FastText/GloVe header).**

**Say:**
> "Two important variants round out the embeddings section. FastText, from Facebook Research, trains on character n-grams rather than whole words. This has two benefits: it can represent words not seen during training (out-of-vocabulary words), and it captures morphological structure. 'Unbelievable' shares subword patterns with 'believable' and 'believe' — FastText's vectors reflect that kinship. GloVe — Global Vectors — instead of training on local windows, factorizes the global word co-occurrence matrix. In practice, GloVe and Word2Vec produce similar results on analogy tasks; FastText is often stronger on downstream classification."

> "Cell 170 shows something especially exciting: FastText provides pre-trained vectors for 157 languages. If you are working with multilingual data — which many social scientists do — this is immediately practical."

---

## Section 3: Transformers — Attention Is All You Need (45 min)

### The Limitation of Static Embeddings (5 min)

**Say:**
> "Static word embeddings have a fundamental problem: one vector per word. The word 'bank' in 'I walked to the river bank' and 'bank' in 'I deposited money at the bank' get the same representation. The meaning of a word depends on its context. The question is: how do we build context into the representation?"

**Write on board:**
```
RNN approach: process tokens sequentially, hidden state accumulates context
              Problem: gradient vanishing over long sequences; can't parallelize

Transformer:  process all tokens simultaneously with attention
              Each token's representation is a weighted combination of all tokens
              The weights are computed dynamically from the content
```

---

### Self-Attention: The Core Mechanism (20 min)

**Write on board:**
```
For each token i:
  Query Qᵢ = W_Q · xᵢ       (what am I looking for?)
  Key   Kⱼ = W_K · xⱼ       (what do I contain?)
  Value Vⱼ = W_V · xⱼ       (what do I contribute?)

Attention weight:  aᵢⱼ = softmax(QᵢKⱼᵀ / √d_k)
Output:            oᵢ = Σⱼ aᵢⱼ · Vⱼ
```

**Say:**
> "This is the heart of Vaswani et al.'s 2017 paper. Each token broadcasts a query — what kind of information am I looking for? Each token also broadcasts a key — here is a summary of what I contain. The dot product of query and key gives a compatibility score: how relevant is token j to token i's computation? After softmax normalization, these scores become attention weights that are applied to the value vectors. The output for each token is a weighted average of all value vectors, where tokens with high relevance get more weight."

**Say:**
> "The critical point for social science: attention weights are interpretable. When you run a BERT model and ask it to classify the sentiment of a political speech, you can inspect which words the model attended to most heavily when making its decision. This is a form of built-in explainability that earlier architectures did not provide. We will explore this in Module 3's SHAP analysis."

**Write on board:**
```
Transformer Block:
  Input → LayerNorm → Multi-Head Self-Attention → + Residual
        → LayerNorm → Feed-Forward Network      → + Residual
        → Output
```

**Say:**
> "Multi-head attention runs several attention mechanisms in parallel — each 'head' can attend to different aspects of the input. One head might learn syntactic relationships (subject-verb agreement), another semantic relationships (co-reference). The residual connections — adding the input back to the output — are the same skip connections we saw in ResNet last week. They ensure gradients flow cleanly through many transformer blocks, which in modern LLMs can number in the dozens to hundreds."

**Ask students:**
> "Positional encoding: the attention mechanism is permutation-invariant — it doesn't know the order of tokens. How does the transformer handle word order?"

*(Expected response: by adding a positional encoding vector to each token's embedding. Vaswani et al. used sine and cosine functions of position; modern transformers often use learned positional embeddings. The key insight: without this, 'dog bites man' and 'man bites dog' would look identical to the attention mechanism.)*

---

### BERT and GPT: Two Paradigms (10 min)

**Write on board:**
```
BERT (Encoder):
  Masked Language Modeling: predict [MASK] tokens using ALL context (bidirectional)
  Good for: classification, NER, question answering

GPT (Decoder):
  Causal Language Modeling: predict next token from LEFT context only (autoregressive)
  Good for: generation, completion, few-shot prompting
```

**Say:**
> "BERT and GPT use the same transformer building block but differ in training objective and architecture. BERT is trained with bidirectional context — it sees the full sentence and predicts masked tokens. This makes it an excellent encoder: its representations are rich because they incorporate context from both directions. GPT is trained causally — it only sees tokens to the left, predicting the next one. This makes it a natural generator: sample a token, append it, sample the next, repeat. The distinction matters for how you use each model: BERT for embedding and classification, GPT for generation and simulation."

**Say:**
> "What unifies them is the pre-training paradigm. Both models are trained on enormous corpora of raw text — Wikipedia, Common Crawl, books — with no manual labeling. The self-supervised training objective extracts statistical patterns at scale. Then fine-tuning on a labeled task adapts the general representation to the specific application. You will do this in Module 2 of the homework."

---

### SHAP Explanations for Transformers (10 min)

**Navigate to Cells 445–470.**

**Say:**
> "One of the most valuable features of Module 3 is the integration of model explanations. SHAP — SHapley Additive exPlanations — assigns each input feature a value representing its contribution to the model's prediction, using game-theoretic Shapley values. For text, each token gets a SHAP value telling you how much it pushed the prediction toward or away from each class."

**Say:**
> "Cell 446 uses HuggingFace's pipeline to run emotion classification. Cell 448 (if present in your runtime) shows SHAP values on top of the text. The practical value: when you fine-tune BERT on a social science dataset — say, classifying political speeches as left or right — SHAP tells you which words drove the classification. This is not just interpretability for its own sake; it is a method for generating theory. If 'regulation' and 'equity' are the highest-weight SHAP features, that is a finding about the linguistic markers of political ideology."

**Ask students:**
> "Lu et al. (2025) find that generative AI systems show 'cultural tendencies' — systematic differences in how they respond to prompts depending on cultural context. If you trained a BERT model on New York Times text and then applied SHAP to explain its predictions on social science documents, what kinds of cultural biases might you expect to find in the feature importances?"

*(Expected responses: Western-centric associations; urban/educated demographic biases; underrepresentation of non-English cultural concepts even within English text; time-specific biases from the training period.)*

---

## Section 4: Code Walkthrough — `Week_2.ipynb` Selected Modules (45 min)

### Module 0: Installation (pre-run, 5 min overview)

**Say:**
> "Cell 4 installs the required packages. Note the `lucem_illud` package from UChicago's Computational Content Analysis group — this contains utility functions for text preprocessing developed by the course team. Run these cells before you close your Colab session. The `!pip install -q` syntax suppresses output; if you get errors, remove the `-q` flag to see what went wrong."

---

### Module 1: Word2Vec — Live Demonstration (15 min)

**Navigate to Cells 9–13.**

**Say:**
> "Cells 9–13 train a Word2Vec model on the hobbies corpus — a small collection of texts about leisure activities. The training parameters: 100-dimensional vectors, window size 10. Run Cell 13 and note the training time. Then run Cell 15."

**Show the `similar_words_df` function:**
```python
def similar_words_df(model, word, topn=10):
    return pd.DataFrame(
        model.wv.most_similar(word, topn=topn),
        columns=["word", "cosine_similarity"]
    )
```

**Say:**
> "Try a few queries: `similar_words_df(w2vmodel, 'game')`, `similar_words_df(w2vmodel, 'music')`. Note how the results differ between the small hobbies model and the NYTimes model loaded in Cell 18. Scale matters enormously for embedding quality — the NYTimes corpus is orders of magnitude larger."

**Navigate to Cells 33–37.**

**Say:**
> "Cells 33–37 reduce the 100-dimensional embedding space to 2D for visualization using PCA followed by t-SNE. The PCA step reduces computation for t-SNE, which is expensive. The resulting scatter plot should show recognizable semantic clusters — sports words near each other, cooking words near each other. This is the geometry of meaning made visible."

**Navigate to Cells 47–63.**

**Say:**
> "Now run the Geometry of Culture demonstration we discussed conceptually earlier. Cell 49 constructs the Gender, Race, and Class dimensions. Cell 51 defines the word sets: occupations, foods, sports. Cell 53 projects each word onto each dimension. Cells 59–63 plot the results. Run them now and take a moment to look at the output."

**Ask students:**
> "Does anything surprise you about where particular words land? What does it mean that 'basketball' is associated with Black and 'tennis' is associated with White in a model trained on New York Times text? Is this a failure of the model or information about the corpus?"

*(This is not a rhetorical question. The honest answer is: it is information about how the NYT covered these topics during its training period, which itself reflects a particular cultural moment and demographic viewpoint. That is useful sociological information, and also a risk if the model is used for any downstream task involving those concepts.)*

---

### Module 2: Encoders/Decoders and Fine-tuning (15 min)

**Navigate to Cell 371 header.**

**Say:**
> "Module 2 covers the seq2seq architecture — encoder processes input to a context vector, decoder generates output from that vector. The French-to-English translation example in Cells 373–430 is a clean implementation of this paradigm with an RNN backbone. This historical model motivates why attention was invented: the fixed-size context vector is a bottleneck for long sequences."

**Navigate to Cell 544 (Fine-tuning).**

**Say:**
> "The practically important part of Module 2 for your homework is fine-tuning. Cell 545 fine-tunes BERT on the Corpus of Linguistic Acceptability — sentences labeled acceptable or unacceptable by linguists. For your homework, you will substitute your own corpus: political speeches, news articles, social media posts, whatever is relevant to your final project."

**Say:**
> "The fine-tuning procedure: load the pre-trained BERT model from HuggingFace, add a classification head (a linear layer mapping from the [CLS] token's embedding to your class labels), freeze or unfreeze the pre-trained weights, and train on your labeled data. With BERT, even a small labeled dataset — hundreds to thousands of examples — typically produces strong results because the pre-trained weights already encode rich linguistic knowledge."

**Instructor Note on SHAP — Cell 616 homework:**
> "The homework exercise at Cell 616 explicitly asks you to use a large pre-trained language model for a task of your choosing and apply SHAP to explain the predictions. This is a key deliverable: the SHAP visualization should appear in your submitted notebook, labeled clearly."

---

### Module 3: Transformers — HuggingFace Pipelines (10 min)

**Navigate to Cell 444 header.**

**Say:**
> "Module 3 uses HuggingFace's high-level `pipeline` API to demonstrate transformer capabilities without building from scratch. Cell 446 runs emotion classification. Cell 469 runs extractive question answering — given a passage and a question, the model identifies which span of the passage is the answer. Cell 489 runs masked language modeling. Cell 505 runs text generation."

**Demonstrate Cell 446:**
```python
from transformers import pipeline
classifier = pipeline("text-classification",
                       model="bhadresh-savani/distilbert-base-uncased-emotion")
result = classifier("I am so excited about this research project!")
```

**Say:**
> "Notice the model card name: `distilbert-base-uncased-emotion`. HuggingFace hosts thousands of fine-tuned models. For virtually any social science classification task, there is likely a pre-trained model you can download and adapt — political ideology classifiers, sentiment analyzers, toxicity detectors, news topic models. Module 3's homework asks you to apply at least one of these to your own corpus."

---

### Module 5: Diffusion Models — Conceptual Overview (5 min)

**Navigate to Cell 708 header.**

**Say:**
> "Module 5 introduces diffusion models — a radically different generative paradigm from autoregressive models like GPT. The key idea: instead of learning to predict the next token directly, the model learns to reverse a process of adding noise."

**Write on board:**
```
Forward process:  x₀ (clean text) → x₁ → x₂ → ... → xₜ (pure noise)
                  Add Gaussian noise at each step

Reverse process:  xₜ (pure noise) → ... → x₁ → x₀ (generated text)
                  The model learns to denoise: predict xₜ₋₁ from xₜ
```

**Say:**
> "For images, this produces Stable Diffusion and DALL-E. For text, it is trickier because text is discrete — you cannot smoothly interpolate between tokens. The notebook's approach is to diffuse in continuous embedding space and decode to tokens only at the end. Cell 715 describes this clearly: 'denoising continuous token embeddings rather than predicting tokens autoregressively.' This module is marked as optional in the homework, but conceptually it matters for Week 8 when we discuss multimodal agents."

---

## Section 5: Student Code Presentations (20 min)

**Say:**
> "Let's hear from students who completed the Week 1 homework. For presentations today, I want to focus specifically on those of you who applied a neural network to your own project dataset. What data did you use? What preprocessing challenges did you encounter? What did the model learn?"

**Invite 2–3 students to present. Suggested discussion questions for each:**

- "What was your classification or regression task? What did your labels represent?"
- "What baseline did you compare against — logistic regression, a simple heuristic?"
- "What did the misclassified examples look like? Can you show one and hypothesize why the model got it wrong?"

**After each presentation, ask the class:**
> "What connections do you see between this student's application and the week's readings? Does the Farrell et al. argument about AI as a social technology apply here?"

---

## Section 6: Discussion — What Do LLMs Know About Society? (10 min)

**Say:**
> "I want to close with a genuinely difficult question that the readings raise but do not fully answer."

**Ask students:**
> "We have seen that word embeddings encode cultural associations — gender, race, class — that are not explicitly programmed but emerge from statistical patterns in text. BERT and GPT encode far richer cultural knowledge than Word2Vec. They have read Wikipedia, Common Crawl, books, and code. When you prompt GPT-4 about political ideology, or about race and occupation, or about what a typical family looks like — what exactly is it doing? Is it producing a culturally calibrated average? An amalgam of multiple viewpoints? A view from nowhere? A very specific view from somewhere?"

*(Allow open discussion. The point is not to reach a consensus but to establish that the question matters enormously for how you interpret any output from these systems.)*

**Reference the Lu et al. (2025) paper:**
> "Lu and colleagues found systematic cultural tendencies in how generative AI responds to prompts across cultural contexts. Chinese and American GPT-4 users, presenting culturally-inflected prompts, receive differently-inflected responses. The model is not culturally neutral. It has cultural tendencies that are legible, measurable, and consequential for social scientific applications."

---

## Closing Summary (5 min)

**Say:**
> "Three take-aways for this week. First: text embeddings transform language into geometry, and the geometry reflects cultural structure. 'king – man + woman = queen' is not just a party trick — it is evidence that distributional semantics encodes relational meaning, and the study of that geometry is a methodology for social science. Second: transformers solve the context problem through self-attention — each token's representation is dynamically computed from its relationships with all other tokens. This enables the rich contextual understanding that makes LLMs so powerful and so potentially useful as social science tools. Third: everything we built this week is encoded with cultural bias, because language itself is encoded with cultural bias. That is not a reason to avoid these methods; it is a reason to apply them critically and to treat the bias itself as data worth studying."

> "Next week we move from understanding LLMs to deploying them as social actors. We will build digital doubles — LLM agents configured with demographic profiles — and use them to simulate voting behavior, political debates, and multi-agent social dynamics. Read Kozlowski and Evans's 'Simulating Subjects' and Argyle et al.'s 'Out of One, Many' before class."

---

## Homework Briefing (5 min)

**Say:**
> "The homework is `Week_2.ipynb`. Complete three of the five modules. Module 1 on Word2Vec and the geometry of culture is recommended for everyone. For your second and third module choices: Module 2 is most directly applicable if you have a classification task for your project; Module 3's HuggingFace pipelines are the fastest route to working with pre-trained transformers; Module 4 covers BERT embeddings in depth."

> "The cross-cutting requirement: apply your trained or fine-tuned model to your final project corpus. The SHAP analysis is explicitly part of the Module 2 homework — include it."

**Write on board:**
```
Lab: Shiyang Lai — Wednesday 2–3pm
Homework due: Before Week 3 (Jan. 23)
Required: Complete 3 of 5 modules + apply model to project corpus
Memo: 300–500 words on a research question from the week's readings (due before class)
```

---

## Timing Guide

| Section | Content | Time |
|---|---|---|
| 1 | Review of Week 1 and text motivation | 15 min |
| 2 | Word embeddings: Word2Vec, GloVe, FastText, geometry of culture | 45 min |
| 3 | Transformers: self-attention, BERT, GPT | 45 min |
| Break | — | 10 min |
| 4 | Code walkthrough: Modules 1–5 of notebook | 45 min |
| 5 | Student code presentations | 20 min |
| 6 | Discussion: what LLMs know about society | 10 min |
| Closing | Summary + homework briefing | 10 min |
| **Total** | | **3 hr** |

---

## Instructor Notes

**Common student confusion points:**
1. The difference between a word embedding and a transformer contextual embedding — emphasize that Word2Vec gives one vector per word regardless of context; BERT gives a different vector for each occurrence of a word depending on its sentence context.
2. The distinction between pre-training and fine-tuning — pre-training is self-supervised on massive raw text; fine-tuning uses labeled data for a specific task, adapting the pre-trained representations.
3. Why attention weights can differ between heads — each head is initialized differently and learns to attend to different aspects of structure; this is why multi-head attention works better than single-head.

**If the NYTimes model file is unavailable:**
> The hobbies corpus model (trained in Cell 13) will work for the analogy and dimension demonstrations, though the associations will be weaker due to smaller corpus size. The gender bias demonstration in Cell 69 requires the Google News vectors; if unavailable, demonstrate with the NYTimes model and note that results will differ.

**Key code patterns to highlight for the homework:**
- The `dimension` function in Cell 47 is directly reusable for any social dimension students want to construct
- The `pipeline` interface in Cell 446 provides the fastest path to applying pre-trained models
- The fine-tuning loop pattern in Cell 545–570 is a template for any BERT fine-tuning task

**Connections to upcoming weeks:**
- The attention mechanism (this week) is the internal mechanism we will inspect with TransformerLens in Week 6
- The embedding geometry methods connect directly to Week 3's digital doubles — the cultural associations encoded in embeddings shape how LLMs simulate different social identities
- SHAP explanations previews the interpretability methods in Week 6
