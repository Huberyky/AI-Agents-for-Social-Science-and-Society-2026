# Week 5 Lecture Script: Sampling, Fine-Tuning, Benchmarking, and Tools
**Course:** AI Agents for Social Science and Society 2026
**Instructor:** James A. Evans
**Date:** February 6, 2026
**Room:** 1155 E. 59th Street, Room 295
**Duration:** 3 hours (1:30–4:20 PM)
**Notebook:** `Week_5_2026.ipynb` (217 cells)

---

## Instructor Preparation Notes

Before class:
- Confirm `imbalanced-learn`, `peft`, `transformers`, and `bitsandbytes` install correctly in Colab
- Pre-download the Indian/non-Indian recipe datasets — they live at DigitalOcean CDN URLs in Cell 82
- The LoRA fine-tuning cells (Cells 183–187) take ~5–10 minutes even on GPU; pre-run if you want to show outputs
- Have ready: Obermeyer et al. 2019 (NEJM) on racial bias in healthcare — worth summarizing key figure
- Have the Shumailov et al. "model collapse" diagram ready — this is a powerful opener for the bias discussion

---

## Section 1: Opening and Week 4 Bridge (0:00–0:15)

**[Stand at board. Notebook projected but closed.]**

Say: "Last week we asked: can we recover causal effects using deep learning? The answer was yes — with the right architecture and the right identification strategy. But we were working with pre-processed, well-curated datasets. Today we confront a more fundamental question: how did that data get there? And what happens to our models — and our inferences — when the data collection process was flawed?"

Write on board:
```
The data pipeline:
Population → Sampling → Dataset → Model → Inference → Decision
                ↑              ↑            ↑
         [Bias enters]  [Bias amplifies] [Bias harms]
```

Say: "Social scientists have known about sampling bias for decades. What's new is the scale at which AI agents inherit, amplify, and perpetuate that bias — and the fact that those agents are now making consequential decisions about people."

Ask students: "Has anyone read the Obermeyer et al. paper on racial bias in healthcare algorithms? What did they find?"

**[Expected: A widely-used commercial algorithm predicted health costs as a proxy for health needs — but costs underestimate needs for Black patients due to historical underinvestment, creating a racially biased risk score.]**

Say: "That paper is the reference case for everything we discuss today. A well-trained, well-evaluated model, making predictions on data that seemed complete and unbiased — that produced systematically worse predictions for Black patients and led to fewer healthcare interventions for that population. The bias was in the data. Not in the architecture. Not in the loss function. In the sampling process that produced the training data."

**[Timing: 10 minutes]**

---

## Section 2: Sampling — From Representativeness to Active Learning (0:15–0:50)

Say: "We're going to move through the sampling module quickly but hit all the important distinctions. The conceptual taxonomy matters — different sampling strategies answer different questions and fail in different ways."

**[Open notebook. Navigate to Cell 3 (Module 1 header) and Cell 5 (Probabilistic Sampling).]**

Write on board:
```
Probabilistic Sampling (each unit has known selection probability):
  - Simple random
  - Stratified random
  - Systematic
  - Cluster
  - Weighted/varying probability

Non-probabilistic (selection probability unknown or by design):
  - Convenience
  - Quota
  - Snowball / respondent-driven
  - Purposive / judgmental
```

### Simple Random Sampling

**[Navigate to Cell 10.]**

Say: "The workhorse. In pandas: `df.sample(100)`. Every row has equal probability of selection. For a mental health survey dataset with hundreds of variables, this is your baseline."

**[Show Cell 10 — two lines of code.]**

Say: "Simple, but beware: simple random sampling does not guarantee representation of rare groups. If your population has 2% with a rare condition, a sample of 100 may include 0, 1, or 5 cases — all consistent with the true proportion but useless for subgroup analysis."

### Stratified Sampling

**[Navigate to Cell 18–21.]**

Say: "The fix: stratify. Sklearn's `train_test_split` with the `stratify` argument ensures your sample has the same proportion of remote vs. non-remote workers — or whatever your stratification variable is — as the full population."

```python
stratified_sample, _ = train_test_split(df, train_size=0.10, stratify=df[['remote']])
```

Ask students: "When would you stratify by multiple variables simultaneously? What's the risk?"

**[Expected: You guarantee representation of each combination, but with many strata you may end up with very small cells. This is the curse of dimensionality in sampling.]**

### Graph Sampling

**[Navigate to Cells 40–70 — graph sampling with littleballoffur.]**

Say: "Social network data adds a wrinkle. You can't sample nodes independently without distorting the network structure — edges between sampled and unsampled nodes get cut, degree distributions change, and clustering coefficients become meaningless."

**[Show Cell 44 — PageRank-based sampler.]**

Say: "PageRank-proportional sampling over-samples high-degree hubs, which is often what you want — hubs carry more structural information. The GitHub social network used here has about 37,000 nodes. Sampling 50% with PageRank-proportional sampling gives you a subgraph that preserves more of the original network's properties than uniform random node sampling."

**[Navigate to Cell 55–57 — Metropolis-Hastings random walk.]**

Say: "Random walk sampling is essential for networks where you only have local access — think Twitter's API where you can only see the friends of users you've already seen. MHRW adds an acceptance criterion that corrects for degree bias in pure random walks."

### Sampling for Imbalanced Classification

**[Navigate to Cell 78–115.]**

Say: "Now the AI-specific problem: imbalanced classes. This is where sampling theory intersects directly with model quality and fairness."

**[Navigate to Cell 86–88. Display the recipe dataset.]**

Say: "Here we have a clean example. Balanced dataset: 50% Indian recipes, 50% not. Imbalanced dataset: 7% Indian, 93% not. Let's see what that does to a classifier."

**[Navigate to Cell 98–100.]**

Say: "On the imbalanced dataset, the classifier gets 80% accuracy on Indian dishes but over 99% on non-Indian. And here's the brutal math:"

Write on board:
```python
36771 / (36771 + 3003) = 0.924  # Accuracy of always-predicting "not Indian"
```

Say: "A classifier that always predicts 'not Indian' would be 92% accurate. Our trained model is only marginally better at Indian prediction, and it learned this 'lazy' strategy because the training data rewards it."

**[Navigate to Cells 106–114 — undersampling and oversampling.]**

Say: "Two fixes: undersample the majority class or oversample the minority. SMOTE — Synthetic Minority Oversampling Technique — generates synthetic minority examples by interpolating between existing minority instances in feature space."

Ask students: "When would oversampling be dangerous? When might it make your model worse?"

**[Expected: If the minority class is genuinely rare in the population, oversampling to 50/50 creates a model calibrated for the wrong prior. For healthcare: if 1% of patients have a rare disease, you want your model's output probabilities to reflect that 1% prevalence, not 50%.]**

Say: "This connects directly to the Obermeyer paper. The healthcare algorithm was trained on a reasonably large and balanced dataset — but the outcome variable, cost, was not a neutral proxy for need. It was a proxy for healthcare utilization, which is itself shaped by systemic racism. No amount of resampling fixes a fundamentally invalid outcome variable."

### Bootstrap Sampling for Embedding Stability

**[Navigate to Cells 117–143.]**

Say: "The last probabilistic method we'll look at is bootstrapping for testing model stability. This is underused in NLP but extremely important for social science."

**[Show Cell 138 — bootstrap loop for Word2Vec.]**

Say: "We train Word2Vec 20 times on bootstrap samples of the corpus, compute the cosine similarity between 'book' and 'story' each time, and report a confidence interval. This is how you know whether a finding — say, that 'law' is closer to 'masculine' than 'nursing' is — is robust to sampling variation or is a fluke of one particular corpus draw."

**[Show the confidence interval output from Cell 142.]**

Say: "90% CI of (0.996, 1.000) — extremely stable. If you got something like (0.2, 0.9), you'd know your finding was fragile and you shouldn't publish it as a robust result."

**[Timing: 35 minutes]**

---

## Section 3: Fine-Tuning with LoRA and QLoRA (0:50–1:20)

Say: "Now we turn to the second module, and the central technical contribution of this week: how do you adapt a large language model to a specific domain without retraining 7 billion parameters?"

Write on board:
```
Problem: GPT-2 or Llama-7B trained on general web text
Goal: Make it useful for a specific domain (legal texts, scientific papers, clinical notes)

Full fine-tuning: update all W → expensive, catastrophic forgetting risk
LoRA: update ΔW = A·B where rank(ΔW) << rank(W) → cheap, targeted
```

**[Navigate to Cell 172–175.]**

### The LoRA Intuition

Say: "The key paper is Hu et al. 2021. The insight is deceptively simple: during fine-tuning, the change in weight matrix ΔW has low intrinsic rank. The model doesn't need to change in all dimensions — just a few."

Write on board:
```
Standard fine-tuning:
  W_new = W_old + ΔW         [update full d×d matrix]

LoRA:
  W_new = W_old + A·B        [A is d×r, B is r×d, rank r << d]
  Trainable params: 2×d×r instead of d²

Example:
  d = 4096, r = 8
  Full: 16,777,216 parameters
  LoRA: 65,536 parameters → 256x reduction in trainable params
```

Ask students: "Why does it make mathematical sense that ΔW has low rank during fine-tuning?"

**[Expected: The model already knows general language; fine-tuning only needs to adjust a small subspace corresponding to domain-specific patterns.]**

Say: "Hu et al. empirically verified this. They showed that the intrinsic dimension of the fine-tuning update is extremely small — often fewer than 100 dimensions — even for models with billions of parameters."

### LoRA in Code

**[Navigate to Cell 178.]**

Say: "Here's the PEFT implementation. Notice what we're specifying: `r=8` is the rank — the dimensionality of the update matrices A and B. `target_modules` tells LoRA which weight matrices to decompose — we're targeting the key, query, value, and output projection matrices in the attention heads."

```python
lora_config = LoraConfig(
    r=8,                  # rank of update matrices
    lora_alpha=32,        # scaling factor (think: learning rate for LoRA)
    target_modules=["k_proj", "q_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model_lora = get_peft_model(base_model, lora_config)
model_lora.print_trainable_parameters()
# Output: trainable params: 4,718,592 || all params: 6,742,609,920 || trainable%: 0.070
```

Say: "Point zero seven percent of parameters are trainable. The other 99.93% are frozen. Yet this is often sufficient to adapt the model substantially to a new domain."

### QLoRA

**[Navigate to Cell 179–180.]**

Say: "QLoRA adds one more step: quantize the frozen base model weights to 4-bit integers. This means a 7 billion parameter model that normally requires 14 GB of GPU memory can be loaded in about 5–6 GB. The LoRA adapters remain in 16-bit. This is what makes fine-tuning accessible on a single consumer GPU."

Write on board:
```
QLoRA memory savings:
  Standard fp16: 7B × 2 bytes = 14 GB
  QLoRA (4-bit base): 7B × 0.5 bytes + adapters ≈ 5–6 GB
  → Opens fine-tuning to researchers without A100/H100 access
```

### Perplexity as Evaluation Metric

**[Navigate to Cells 186–187.]**

Say: "How do we know if fine-tuning helped? The notebook uses perplexity — the exponent of the average cross-entropy loss. Perplexity measures how surprised the model is by text."

Write on board:
```
Perplexity = exp(avg cross-entropy loss)
  = exp(1/N × Σ -log P(token_i | context))

Low perplexity → model assigns high probability to the text → model "knows" this domain
High perplexity → model is surprised → text is out of distribution
```

```python
eval_results = trainer.evaluate()
val_ppl = np.exp(eval_results["eval_loss"])
print(f"Validation Perplexity: {val_ppl:.2f}")
# Before fine-tuning on domain text: ~200
# After fine-tuning: ~40-80 (task-dependent)
```

Ask students: "If you fine-tuned a model on conservative political news until its perplexity on that text was very low, what would happen to its perplexity on progressive political news?"

**[Expected: It would increase — the model becomes more "surprised" by text outside its fine-tuning domain.]**

Say: "This is exactly what Chu et al. 2023 found. 'Language Models Trained on Media Diets Can Predict Public Opinion' — they showed that fine-tuning on different news sources shifts the model's partisan predictions systematically. The model develops a 'media diet' bias that mirrors the ideological lean of its training corpus. Fine-tuning is not neutral."

**[Timing: 30 minutes]**

---

## Section 4: Benchmarking LLM Agents (1:20–1:45)

Say: "Once you've fine-tuned a model, how do you know it's better? This is the benchmarking module — and it exposes one of the deepest epistemological problems in AI research."

**[Navigate to Cell 190 — Module 3 header.]**

Write on board:
```
Benchmark problems:
1. Selection bias — tasks don't represent real deployment distribution
2. Contamination — model may have been trained on benchmark test data
3. Cultural bias — tasks reflect Western, English-centric assumptions
4. Difficulty calibration — floor/ceiling effects obscure model differences
5. Metric gaming — model optimizes the metric, not the underlying capability
```

### Centered Kernel Alignment (CKA)

**[Navigate to Cells 191–197.]**

Say: "CKA is a metric for comparing internal representations across models. It asks: are these two networks computing similar things, even if their weights are completely different? This is powerful for understanding what fine-tuning actually changes."

Write on board:
```
CKA(X, Y) = ||X^T Y||²_F / (||X^T X||_F × ||Y^T Y||_F)

Properties:
  - CKA = 1: identical representations (up to rotation/scaling)
  - CKA = 0: completely unrelated representations
  - Invariant to orthogonal transformation and isotropic scaling
```

**[Show Cell 194 — the linear_CKA function.]**

Say: "The geometric intuition: CKA measures how much the structure of one representation space is 'explained by' the other. If two layers learn similar features — even with different neuron ordering or scaling — CKA will be close to 1."

**[Navigate to Cell 196–197 — the CKA heatmap visualization.]**

Say: "When you plot a CKA heatmap between layers of a base model and a fine-tuned model, you typically see high CKA in early layers — both models have learned similar low-level features like syntax and morphology — but diverging CKA in later layers where domain-specific semantic content is encoded."

Ask students: "If you fine-tuned a model on clinical text and then tested it on legal text, where in the CKA heatmap would you expect to see the biggest difference from the general-purpose model?"

**[Expected: Later layers, where domain-specific semantics are encoded.]**

### Agent Benchmarking Frameworks

**[Navigate to Cells 198–202.]**

Say: "For evaluating AI agents — not just language models — the challenges multiply. An agent needs to complete multi-step tasks, use tools, and handle errors. AgentBench and similar frameworks evaluate along several dimensions: task completion rate, efficiency (steps per task), safety (does the agent do anything harmful?), and robustness (does performance degrade under paraphrase or noise?)."

**[Show Cell 199 — the BenchmarkTask and EvaluationResult dataclasses.]**

Say: "The design of a benchmark task object requires you to specify: the expected output, the maximum number of steps, and a timeout. This last point is often ignored — a real deployed agent that takes 500 steps to answer a simple question is useless even if it eventually gets the right answer."

**[Navigate to Cell 201–202 — bias in benchmark design.]**

Say: "The Mohammadi et al. 2025 paper makes this point forcefully: benchmark performance depends heavily on task distribution. If your benchmark has 90% English-language, Western-centric tasks, you can have a model that performs at 80% overall but fails on 60% of non-English tasks. The headline number hides everything."

**[Show Cell 202 code — the bar chart of benchmark composition vs. reported performance.]**

Say: "This simulation makes the point starkly. Change the benchmark composition from 70% English to 30% English and the 'same' model drops from 85% to 72% — not because the model changed, but because the benchmark changed. This is why benchmark choice is a research decision, not a technical one."

**[Timing: 25 minutes]**

---

## Section 5: Model Context Protocol — Tool-Using Agents (1:45–2:05)

Say: "The last module introduces something you'll use heavily in your projects: the Model Context Protocol, or MCP. This is Anthropic's open standard for giving AI agents access to external tools and data."

**[Navigate to Cell 204.]**

Write on board:
```
Without tools: LLM knows only what was in training data
With tools:    LLM can →  search the web
                          execute code
                          query databases
                          send API requests
                          read/write files
```

Say: "The key insight in MCP is standardization. Before MCP, every tool integration was ad-hoc — different APIs for OpenAI vs. Anthropic vs. Cohere, different function-calling schemas, different error handling conventions. MCP defines a single interface that any model can use to interact with any tool."

### Tool Definition

**[Navigate to Cell 207.]**

Say: "A tool is defined by: its name (how the LLM refers to it), a description (how the LLM knows when to use it — this matters enormously), and a parameter schema (what arguments it accepts). Notice that the description is natural language — the LLM decides whether to call a tool based on this description."

**[Show Cell 208 — the calculator tool.]**

```python
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        allowed_chars = set('0123456789+-*/().^ ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression.replace('^', '**'))
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"
```

Ask students: "What's the security risk in this calculator implementation?"

**[Expected: `eval()` executes arbitrary Python code. The character whitelist is a partial mitigation but a sophisticated attacker could still find exploits. This is a real concern in deployed agentic systems.]**

Say: "Tool security is a major open problem in AI agent research. An agent that can execute code, query databases, or send API requests is a significant attack surface. The MCP specification includes sandboxing guidelines, but they're not universally implemented."

### MCP Server Architecture

**[Navigate to Cell 210–211.]**

Say: "The MCP architecture has three components: the host — your AI application — the server — a process that exposes tools and data — and the transport layer — how they communicate. Claude Code, for example, uses MCP to give Claude access to your filesystem, terminal, and browser."

Write on board:
```
MCP Architecture:
  [LLM Host] ←→ [MCP Client] ←→ [MCP Server]
                                     ↕
                              [Tools / Resources]
                              - Files, databases
                              - APIs, search
                              - Code execution
```

Say: "For your projects, MCP means you can build an agent that: retrieves papers from your literature corpus, runs statistical tests on your dataset, queries a database of social media posts, and synthesizes findings — all from a single conversational interface. This is the agentic research assistant use case."

**[Timing: 20 minutes]**

---

## Section 6: Model Collapse — The Shumailov Warning (2:05–2:15)

Say: "I want to take ten minutes on a finding that should concern everyone in this room, especially those planning to use AI-generated data in their projects."

Write on board:
```
Shumailov et al. 2024: "AI Models Collapse When Trained on Generated Data"
```

Say: "The paper shows that if you train a language model on human text, then use that model to generate synthetic training data, then train a new model on the synthetic data — and repeat — the model's output distribution collapses. Rare, complex patterns disappear. The model converges to a narrow, high-frequency mode of the original distribution."

Draw on board:
```
Generation 0 (human): [wide, complex distribution]
Generation 1 (AI-generated): [slightly narrower]
Generation 2 (AI-on-AI): [noticeably narrower]
...
Generation N: [degenerate, repetitive output]
```

Ask students: "Why does this happen mechanically?"

**[Expected: The model assigns low probability to rare patterns, generates less of them, the next model sees fewer rare patterns in training, assigns even lower probability — a positive feedback loop toward modal responses.]**

Say: "The social science implication is severe. If you use LLM-generated survey responses to simulate human subjects, you may be feeding the models that will be used to generate future LLM responses, accelerating collapse toward a homogeneous 'average human.' The diversity of human thought and culture would be progressively erased from model outputs."

Say: "This is also why the PPI framework from Week 4 is so important. It explicitly requires a corpus of human-labeled data to anchor the AI predictions. You can't run PPI on 100% AI-generated labels — you need real human ground truth to correct the bias."

**[Timing: 10 minutes]**

---

## Section 7: Student Code Presentations (2:15–2:35)

**[Call on 3–4 students. Suggested prompts:]**

- "Which sampling method did you apply to your project dataset? Did it change the class balance?"
- "What rank did you use for LoRA? Did changing rank affect perplexity?"
- "If you did the benchmarking module, what bias did you find in your benchmark design?"
- "If you did MCP tools, what domain tool did you implement?"

---

## Section 8: Discussion — Bias, Scale, and Responsibility (2:35–2:55)

Write on board:
```
"Without these, agents inherit and amplify the biases in their training data."
— Week 5 core message
```

Ask students: "The Caliskan et al. 2017 paper 'Semantics Derived from Language Corpora Contain Human-Like Biases' showed that word embeddings trained on text reproduce gender and racial stereotypes found in the text. Is this a bug or a feature?"

**[Allow discussion. Key tensions:]**

- Bug: perpetuates harmful stereotypes, creates feedback loops
- Feature: accurate reflection of cultural attitudes useful for studying culture
- Neither: depends entirely on the downstream use case

Say: "The Chu et al. 2023 finding is instructive here. Models trained on media diets can predict public opinion — but they do so by encoding the partisan perspective of the media they were trained on. If you use such a model as an advisor, you're not getting 'intelligence' — you're getting the editorial perspective of the Fox News (or MSNBC) editorial board, encoded in 7 billion parameters."

Ask students: "Suppose you're a public health researcher and you want to use an LLM to code free-text hospital records. What sampling strategy would you use to select which records to annotate by humans, and what benchmark would you use to evaluate the LLM's coding quality?"

**[This is a design question. Expected answers: active learning on uncertain predictions; human annotation as gold standard with inter-rater reliability; stratified sampling by diagnosis code to ensure coverage of rare conditions; benchmarking against a specialist physician's gold standard labels rather than a general human crowdworker's labels.]**

**[Timing: 20 minutes]**

---

## Section 9: Closing Summary and Homework Briefing (2:55–3:10)

Write on board:
```
Week 5 Core Takeaways:
1. Sampling strategy determines what your model can learn — non-representative samples
   produce non-generalizable models
2. Class imbalance requires explicit correction (under/oversample/weight) or
   you build classifiers that are right for the wrong reasons
3. LoRA: fine-tune 7B parameter models on a single GPU by updating only 0.07% of weights
4. QLoRA: 4-bit quantization + LoRA → accessible fine-tuning on consumer hardware
5. Benchmarks encode assumptions — report the benchmark distribution, not just the score
6. CKA: interpretable metric for comparing what different models know
7. MCP: standardized interface for building tool-using agentic pipelines
8. Model collapse: AI-on-AI training degrades diversity — always anchor with human data
```

Say: "For homework, complete three of four modules. Module 1 (Sampling) is the most directly applicable to your projects — everyone should do it. Module 2 (LoRA) is the highest technical value-add for your final project if you're working with text data. Modules 3 and 4 are more specialized but very useful for certain project types."

Say: "A note on Module 2: you need GPU access. If Colab's free tier gives you CPU, the training loop will be extremely slow. Use the T4 GPU runtime, or reduce `num_train_epochs` to 1. The point is to see the perplexity change and understand what LoRA ranks do — not to train a production model."

Say: "For the memo, your task is to identify a potential source of sampling or training bias in your final project's data pipeline and propose how to measure and mitigate it. A good memo describes the bias mechanism, not just its existence."

Ask students: "Quick check — what is the difference between SMOTE and random oversampling?"

**[Expected: SMOTE generates synthetic examples by interpolating between existing minority instances; random oversampling just duplicates existing examples. SMOTE creates more diverse training examples and reduces overfitting risk.]**

**[Final thought:]**

Say: "Shumailov's model collapse paper ends with a warning: if the internet becomes saturated with AI-generated content — and at current rates, it will — then all future LLMs will be trained on data that is progressively more homogeneous. The epistemic diversity of human knowledge, once encoded in LLMs, may not survive the next generation of training runs. That is a social science problem, not just a technical one. It's the kind of problem you are being trained to study and solve."

---

## Appendix: Key Code Snippets for Board Reference

**Stratified sampling:**
```python
from sklearn.model_selection import train_test_split
stratified_sample, _ = train_test_split(df, train_size=0.10, stratify=df[['remote']])
```

**Imbalanced class correction:**
```python
from imblearn.over_sampling import SMOTE
resampler = SMOTE()
X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
# Now 50/50 class balance with synthetic minority examples
```

**LoRA configuration:**
```python
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(
    r=8,                    # rank — try 4, 8, 16
    lora_alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
model_lora = get_peft_model(base_model, lora_config)
# Check parameter count:
model_lora.print_trainable_parameters()
```

**Perplexity evaluation:**
```python
eval_results = trainer.evaluate()
val_ppl = np.exp(eval_results["eval_loss"])
print(f"Perplexity: {val_ppl:.2f}")
```

**Bootstrap confidence interval for embedding similarity:**
```python
from sklearn.utils import resample
similarities = []
for _ in range(20):
    boot_sample = resample(texts)
    model = Word2Vec(boot_sample, vector_size=100, window=10)
    similarities.append(model.wv.similarity('word1', 'word2'))
sorted_sims = sorted(similarities)
print(f"90% CI: ({sorted_sims[1]:.3f}, {sorted_sims[18]:.3f})")
```

---

## Timing Summary

| Section | Duration | Cumulative |
|---|---|---|
| Opening and Week 4 Bridge | 15 min | 0:15 |
| Sampling Theory and Methods | 35 min | 0:50 |
| LoRA and QLoRA Fine-Tuning | 30 min | 1:20 |
| Benchmarking and CKA | 25 min | 1:45 |
| MCP and Tool-Using Agents | 20 min | 2:05 |
| Model Collapse Discussion | 10 min | 2:15 |
| Student Presentations | 20 min | 2:35 |
| Discussion: Bias and Responsibility | 20 min | 2:55 |
| Closing + Homework | 15 min | 3:10 |
| Buffer | 10 min | 3:20 |
