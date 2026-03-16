# Week 5: Sampling, Fine-Tuning, Benchmarking, and Tools
## AI Agents for Social Science and Society 2026
**Instructor:** James A. Evans
**Date:** February 6, 2026
**Duration:** 3 hours | 30 slides

---

## Opening Concept
> *"How do we build agents that know what we need them to know — without inheriting the biases we don't want?"*

---

---

## Slide 1: The Domain Adaptation Problem

**Visual Description:**
A two-panel diagram. Left panel: a generic LLM cloud labeled "GPT / Llama / Mistral" with inputs like "What is the capital of France?" and outputs "Paris." Right panel: a domain-specific agent cloud labeled "Medical Claims Classifier" with inputs like "Patient 7812: hypertension, CHF, elevated creatinine" and outputs "High-risk: cardiovascular event." An arrow labeled "How do we get from left to right?" bridges the two panels. Background is dark charcoal; accent color is amber.

**Bullet Points / Text:**
- General-purpose LLMs fail at specialized social science tasks
- Domain adaptation requires the right *data*, the right *training strategy*, and the right *evaluation*
- This week: the full pipeline from sampling to deployment

**Instructor Notes:**
Begin by grounding students in the practical problem: a model trained on internet text does not know what a medical claims record looks like, what a Supreme Court opinion means, or how a 19th-century newspaper is structured. Every domain-specific agent they will build for their final projects requires some version of what we cover today. Ask the class: what domain does your final project live in, and what would a naive LLM get wrong about it?

---

## Slide 2: Week 5 Roadmap

**Visual Description:**
A horizontal timeline with five stations illustrated as distinct icons: (1) a funnel labeled "Sampling," (2) a weight matrix with low-rank decomposition labeled "LoRA / QLoRA," (3) a bar chart with model names on x-axis labeled "Benchmarking," (4) a plug-and-socket icon labeled "MCP Tools," (5) a warning triangle with a DNA helix labeled "Model Collapse." Each station is connected by a dashed line. Below the timeline: "3 hours, 4 modules, 1 central question."

**Bullet Points / Text:**
- Sampling: building representative, balanced training data
- Fine-tuning: parameter-efficient adaptation with LoRA/QLoRA
- Benchmarking: rigorous evaluation with CKA and task design
- Tools: the Model Context Protocol and tool-using agents

**Instructor Notes:**
Walk students through the day's structure so they know what's coming. Emphasize that these four topics are not independent — good sampling enables good fine-tuning, which requires good benchmarking, which is then deployed through tool-use. The model collapse discussion at the end ties everything together by showing what happens when any of these steps goes wrong.

---

## Slide 3: Why Sampling Is a Social Science Problem

**Visual Description:**
A classic newspaper front page from 1936 with the headline "LANDON WINS!" (the Literary Digest poll). Below it, a smaller clipping: "FDR wins 523–8 in Electoral College." A callout box explains: "Literary Digest surveyed 10 million people — but from phone books and car registrations. Working-class voters who supported FDR were systematically excluded." The image is sepia-toned with a modern annotation overlay in red.

**Bullet Points / Text:**
- Sampling bias in 1936 — and in every LLM training corpus
- The Literary Digest failure: large N does not fix unrepresentative selection
- AI training data is the most consequential sampling decision in modern science

**Instructor Notes:**
The Literary Digest story is a foundational lesson in how sampling strategy, not sample size, determines validity. Draw the explicit parallel to AI: GPT is trained on Common Crawl, which overrepresents English, overrepresents written text, and overrepresents online demographics. Every bias we discuss in AI today is downstream of a sampling decision someone made. Ask students: what does the training corpus of an LLM look like compared to the true distribution of human communication?

---

## Slide 4: Probability Sampling: The Gold Standard

**Visual Description:**
Three side-by-side visual panels on a white background with blue accents:
(1) **Simple Random**: 100 dots, 20 highlighted randomly — labeled "SRS: equal probability for each unit"
(2) **Stratified**: 100 dots grouped into 4 colored strata, 5 highlighted per stratum — labeled "Stratified: proportional representation guaranteed"
(3) **Systematic**: 100 dots in a row, every 5th highlighted — labeled "Systematic: interval k = N/n"
Below each panel: a short formula.
- SRS: P(select i) = n/N
- Stratified: n_h = n × (N_h/N)
- Systematic: select unit k, 2k, 3k, ...

**Bullet Points / Text:**
- Simple random: every unit has equal selection probability
- Stratified: guarantee representation across known subgroups
- Systematic: practical for ordered lists; watch for periodicity artifacts

**Instructor Notes:**
Walk through each method with the diagrams. Stratified sampling is particularly important for social science AI — if you're building a sentiment classifier for political speech, you must stratify by party, year, and speaker demographics to avoid a model that works well for white male politicians and fails for everyone else. Systematic sampling is seductive for large corpora but can introduce bias if the ordering has a natural cycle (e.g., every Monday editorial has a consistent tone).

---

## Slide 5: Non-Probabilistic Sampling: When It's All You Have

**Visual Description:**
A two-column table with a "danger" visual design (yellow caution stripe at the top). Left column: Method names in bold — Convenience, Quota, Purposive, Snowball. Right column: one-sentence descriptions and a "typical AI use case" for each. At the bottom, a red callout box: "Non-probabilistic samples produce non-generalizable models. Know what you're giving up." Sidebar: a small scatter plot showing in-sample vs. out-of-sample prediction divergence.

**Bullet Points / Text:**
- Convenience: use what's available (Common Crawl, Twitter 1%)
- Quota: ensure cell counts match known population marginals
- Purposive / snowball: trace networks, rare populations, qualitative depth
- All non-probabilistic methods require explicit out-of-distribution caveats

**Instructor Notes:**
Most AI training data is collected through convenience sampling — crawl the web and take what you get. This isn't always wrong, but it requires honesty about scope. A model fine-tuned on Reddit comments will not generalize to formal academic prose, and a model fine-tuned on formal academic prose will not capture vernacular speech. Quota sampling is a useful middle ground when you know the population parameters but can't afford full probability sampling.

---

## Slide 6: The Imbalanced Dataset Problem

**Visual Description:**
A large donut chart with two segments: 95% in dark grey labeled "Majority class: 'no fraud'" and 5% in bright red labeled "Minority class: 'fraud.'" Below the chart, a confusion matrix for a naive classifier that predicts only the majority class: accuracy = 95%, recall for minority class = 0%. A red "X" over the confusion matrix. Caption: "A model that is always wrong about the thing that matters."

**Bullet Points / Text:**
- Class imbalance is endemic in social science: rare events, minority groups, edge cases
- Naive accuracy is misleading — optimize for F1, AUC-ROC, or Matthews Correlation Coefficient
- Solutions: resampling (SMOTE, Tomek), class weighting, threshold calibration

**Instructor Notes:**
The medical fraud example hits home: a model with 95% accuracy that never flags a fraud case is useless. This exact failure mode appeared in the healthcare algorithm studied by Obermeyer et al. (2019), where Black patients were systematically assigned lower risk scores because the proxy label — healthcare spending — was itself biased. The accuracy number looked fine; the discrimination was invisible. Teach students to always disaggregate their evaluation metrics by subgroup.

---

## Slide 7: SMOTE and Tomek Links

**Visual Description:**
A two-panel scatter plot (2D feature space) with minority class points in red (triangles) and majority class in blue (circles).
**Left panel — SMOTE:** New synthetic red points appear between existing red points, connected by thin lines. Label: "SMOTE: interpolate between k-nearest minority neighbors. New point = x_i + λ(x_j - x_i), λ ~ U(0,1)"
**Right panel — Tomek Links:** Pairs of points from opposite classes that are each other's nearest neighbors are circled and removed. Label: "Tomek Links: remove borderline majority points that overlap with minority region."
A small code snippet at the bottom:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
X_res, y_res = SMOTE().fit_resample(X, y)
```

**Bullet Points / Text:**
- SMOTE oversampling: synthesize minority class examples by interpolation
- Tomek Links undersampling: remove ambiguous majority-class boundary cases
- Combine both: SMOTEENN or SMOTETomek pipelines for cleaner decision boundaries

**Instructor Notes:**
Walk through the SMOTE formula geometrically — you're drawing a line segment between two minority-class examples and picking a random point on that segment. This creates realistic-looking synthetic examples without simply duplicating rows. Tomek Links do the complementary job: they clean the decision boundary by removing the majority-class points that are too close to minority examples. In practice, combining SMOTE and Tomek Links often outperforms either alone.

---

## Slide 8: Bootstrap Sampling for Embedding Stability

**Visual Description:**
A diagram showing the bootstrap process for evaluating word embedding stability. Top: a corpus of 10,000 documents. An arrow branches into 5 bootstrap samples (B1 through B5), each drawn with replacement. Each sample trains a separate embedding model, producing 5 embedding spaces. Then: pairwise cosine similarity between the embeddings of the word "justice" across 5 models is computed. A violin plot at the bottom shows the distribution of similarity scores, with a dashed red line at 0.85 labeled "stability threshold." Code snippet:
```python
def bootstrap_embeddings(corpus, n_boot=100, seed=42):
    stabilities = []
    for b in range(n_boot):
        sample = resample(corpus, random_state=b)
        model = train_word2vec(sample)
        stabilities.append(cosine_sim(model, base_model))
    return np.mean(stabilities), np.std(stabilities)
```

**Bullet Points / Text:**
- Bootstrap: resample with replacement B times; measure variance of your estimator
- Use for: embedding stability, benchmark score confidence intervals, fine-tuning variance
- If embeddings are unstable under bootstrap, the training corpus is too small or too noisy

**Instructor Notes:**
Bootstrap sampling is underused in NLP and AI evaluation. If you fine-tune a model on 500 documents and get a benchmark score of 0.82, you have no idea if that number is stable. Bootstrap 100 times and report a confidence interval — you might find the true range is 0.74 to 0.89, which tells a very different story. This is especially important for social science applications where you are making substantive claims based on model outputs.

---

## Slide 9: Active Learning — Querying Strategically

**Visual Description:**
A central loop diagram showing the active learning cycle: (1) "Labeled Pool (small)" → (2) "Train Model" → (3) "Query Strategy: which unlabeled example is most informative?" → (4) "Oracle/Human Annotator" → (5) "Add to labeled pool" → back to (2). Three query strategy sub-diagrams branch off step (3):
- **Uncertainty Sampling:** probability bar chart; model is 51%/49% split → high uncertainty → query
- **Query-by-Committee:** three committee model predictions disagree on one example → query
- **Diversity Sampling:** example far from existing labeled examples in embedding space → query

**Bullet Points / Text:**
- Active learning: minimize labeling cost while maximizing model improvement
- Uncertainty sampling: query where P(y=1|x) ≈ 0.5, or maximize entropy H(y|x)
- Query-by-committee: query where a committee of models most disagrees
- Critical for expensive annotation: clinical coding, legal classification, fieldwork data

**Instructor Notes:**
Active learning is the bridge between sampling theory and practical AI development. When annotation is expensive — think medical diagnosis, legal coding, or hand-coded survey responses — you cannot label everything. Active learning lets the model tell you which examples are most valuable to annotate next. Uncertainty sampling is the simplest form: find the example the model is most confused about and ask a human to label it. Query-by-committee is more robust because it's less sensitive to model miscalibration.

---

## Slide 10: The Fine-Tuning Landscape

**Visual Description:**
A vertical pyramid diagram with four tiers, each labeled with approximate parameter counts and GPU cost:
- **Tier 4 (top, smallest):** "Prompt Engineering" — 0 trainable params, $0 compute, but limited adaptation
- **Tier 3:** "Adapter / Prefix Tuning" — ~1M params, low cost
- **Tier 2:** "LoRA / QLoRA" — ~10M params, 1–2 GPU-hours
- **Tier 1 (bottom, largest):** "Full Fine-Tuning" — all 7B+ params, 100+ GPU-hours
An arrow on the right side labels the continuum from "cheapest / least powerful" (top) to "most expensive / most powerful" (bottom). A red star next to Tier 2 labeled "This week's focus."

**Bullet Points / Text:**
- Full fine-tuning: update all weights — expensive, prone to catastrophic forgetting
- Adapter methods: insert small trainable modules between frozen layers
- LoRA: reparametrize weight updates as low-rank matrices — elegant and efficient
- QLoRA: combine 4-bit quantization + LoRA for single-GPU fine-tuning of 7B+ models

**Instructor Notes:**
Place LoRA and QLoRA in the context of the full fine-tuning spectrum. The key insight is that you almost never need full fine-tuning for domain adaptation — the general knowledge is already there, you just need to redirect it. LoRA exploits the observation that weight update matrices during fine-tuning have intrinsically low rank, meaning they can be approximated by the product of two thin matrices. This insight is both computationally and intellectually elegant.

---

## Slide 11: LoRA — Low-Rank Adaptation

**Visual Description:**
A detailed weight matrix diagram. The original pre-trained weight matrix W₀ (large, d × d, shown as a grey block labeled "frozen") is shown alongside two thin matrices: A (d × r, blue) and B (r × d, green), where r << d. The product A·B (shown with a small multiplication symbol) is added to W₀ to produce W = W₀ + A·B. Below the diagram:
```
W = W₀ + ΔW = W₀ + A·B
where A ∈ ℝ^{d×r}, B ∈ ℝ^{r×d}, r << d

Parameters saved:
  Full ΔW: d² = (4096)² ≈ 16.7M
  LoRA ΔW: 2·d·r = 2·4096·8 ≈ 65K
  Compression ratio: ~256×
```
A bar chart comparing parameter counts at r=4, 8, 16, 64.

**Bullet Points / Text:**
- W = W₀ + A·B: only A and B are trained; W₀ is frozen
- Rank r controls the expressiveness vs. efficiency tradeoff (typical: r = 4 to 16)
- Applied to Q, K, V, and O matrices in attention layers
- At inference: merge W₀ + A·B — zero latency overhead

**Instructor Notes:**
The LoRA insight is that fine-tuning doesn't need to use the full d×d parameter space — the directions of change are much lower-dimensional than the full weight matrix. Mathematically, this is equivalent to saying that the fine-tuning gradient lies in a low-dimensional subspace, which is empirically observed and theoretically motivated by the lottery ticket hypothesis. Walk through the parameter count calculation explicitly so students understand the concrete savings: for a 7B-parameter model, LoRA at rank 8 might train fewer than 5 million parameters.

---

## Slide 12: QLoRA — Fine-Tuning on a Consumer GPU

**Visual Description:**
A stacked architecture diagram showing three innovations layered on top of each other:
- **Layer 1 (bottom):** "4-bit NF4 Quantization" — a weight histogram showing how 16-bit floats are bucketed into 16 quantization bins, with the formula: w_q = round(w / scale) where scale = max(|W|)/7
- **Layer 2 (middle):** "Double Quantization" — the quantization constants are themselves quantized
- **Layer 3 (top):** "Paged Optimizers" — GPU memory overflow handled by unified memory with CPU pages
A sidebar shows GPU memory comparison: "LLaMA-65B full fine-tuning: ~780 GB GPU RAM. QLoRA: ~48 GB (fits on 2× A100 or 1× H100)."

**Bullet Points / Text:**
- NF4 quantization: 4-bit normal float optimized for normally distributed weights
- Double quantization: quantize the quantization constants to save additional memory
- Paged optimizers: handle memory spikes gracefully via CPU offloading
- Result: fine-tune a 7B model on a single 24GB consumer GPU

**Instructor Notes:**
QLoRA (Dettmers et al., 2023) was a genuine democratization milestone. Before QLoRA, fine-tuning a 7B-parameter model required expensive cloud GPU clusters. After QLoRA, a graduate student with a gaming PC could do it. This matters enormously for social science AI: it means domain-specific models are no longer the exclusive province of large labs. Emphasize the three technical innovations — 4-bit NF4 quantization, double quantization, and paged optimizers — and explain how each contributes to memory reduction.

---

## Slide 13: Fine-Tuning GPT-2 — A Concrete Example

**Visual Description:**
A two-panel code + output display. Left panel: Python code block with syntax highlighting:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model

model = GPT2LMHeadModel.from_pretrained("gpt2")
config = LoraConfig(r=8, lora_alpha=32,
                    target_modules=["c_attn"],
                    lora_dropout=0.1)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# → trainable params: 294,912 / 124,734,720 (0.24%)
```
Right panel: a before/after perplexity table:
| Epoch | Train PPL | Val PPL |
|-------|-----------|---------|
| 0     | 89.4      | 91.2    |
| 1     | 43.1      | 48.7    |
| 3     | 21.6      | 29.3    |
| 5     | 12.4      | 23.8    |
A note below: "Perplexity = exp(cross-entropy loss). Lower = better fit to domain."

**Bullet Points / Text:**
- Fine-tune GPT-2 (124M params) with LoRA — train only 0.24% of parameters
- Perplexity: measure of how well the model predicts held-out text in your domain
- Overfitting signal: train PPL >> val PPL at later epochs
- Application: domain-specific text generation, classification, knowledge probing

**Instructor Notes:**
Walk students through the code live if possible — this is what Module 2 of the homework covers. The key number to emphasize is 0.24%: training less than a quarter of a percent of the parameters, but getting dramatic perplexity improvements on domain text. This is the power of LoRA — the frozen base model already knows how to use language; you're just redirecting which patterns of language it applies in your domain.

---

## Slide 14: Perplexity as a Social Science Instrument

**Visual Description:**
A line chart with multiple colored lines representing different LLMs (GPT-4, Claude 3, Llama 3, Mistral 7B). The x-axis is "Year of Publication (1990–2024)"; the y-axis is "Perplexity on Test Set." The lines decline steeply until about 2020, then flatten. A dashed vertical line at 2020 is labeled "LLM training cutoff cliff." A callout box quotes Zhang & Evans (2025): "Language model perplexity on abstracts predicts scientific surprise — low-perplexity findings are conventional; high-perplexity findings are novel." Below the chart, a scatter plot showing PPL vs. citation count, with a positive correlation for highly novel papers.

**Bullet Points / Text:**
- Perplexity measures surprise: how unexpected is this text to the model?
- Zhang & Evans (2025): perplexity predicts scientific novelty and citation impact
- Low-perplexity claims = expected, conventional; high-perplexity claims = surprising, novel
- Application: use fine-tuned domain models to detect unusual claims or emerging ideas

**Instructor Notes:**
Perplexity is not just a training metric — it's a substantive social science instrument. When you fine-tune a model on a corpus of prior literature and then measure the perplexity of new publications, you're measuring how surprising that new work is relative to the existing knowledge base. Zhang and Evans found this predicts both scientific impact and eventual influence. This connects the technical concept of perplexity to the sociological question of novelty and recognition in science.

---

## Slide 15: Benchmarking — What Does "Good" Mean?

**Visual Description:**
A decision flowchart for benchmark design. Starting node: "What do you want your agent to do?" Three branches: (1) "Classify text → F1, AUC, MCC," (2) "Generate text → BLEU, ROUGE, BERTScore, human eval," (3) "Reason / plan → task completion rate, trajectory efficiency." Below the flowchart: a table contrasting "Leaderboard benchmarks" (MMLU, HellaSwag, GSM8K) vs. "Domain-specific benchmarks" (your custom task), with columns for "coverage," "contamination risk," "generalizability," and "social science validity." A warning box: "Benchmark contamination: if your test set appeared in training data, scores are meaningless."

**Bullet Points / Text:**
- Benchmark design is a hypothesis about what "intelligence" means in your domain
- Metric selection: accuracy (balanced classes), F1 (imbalanced), BERTScore (semantic), human eval (nuanced)
- Standard benchmarks may be contaminated — many LLMs have seen MMLU in training
- Design your own benchmarks for social science tasks: they test what actually matters

**Instructor Notes:**
The choice of benchmark is itself a theoretical claim about what matters. If you benchmark a political persuasion model using accuracy on a balanced test set, you're saying all persuasion attempts are equally important. If you use weighted recall on an underrepresented demographic, you're saying minority community persuasion is what you care about. Be explicit about this with students — the benchmark encodes your values about the task.

---

## Slide 16: Centered Kernel Alignment (CKA)

**Visual Description:**
A mathematical diagram explaining CKA step by step.
Step 1: Two representation matrices X (n × p) and Y (n × q) — activations from two different layers or models on the same n inputs.
Step 2: Gram matrices K = X·Xᵀ and L = Y·Yᵀ (both n × n).
Step 3: Center: K̃ = H·K·H where H = I - (1/n)11ᵀ
Step 4: CKA formula:
```
CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))

where HSIC(K, L) = (1/(n-1)²) tr(K̃L̃)
```
A 5×5 heatmap showing CKA values between layers 1–5 of two different models, with high similarity (yellow) in early layers and low similarity (blue/purple) in later layers.

**Bullet Points / Text:**
- CKA: measure similarity between representations in two neural networks
- Invariant to orthogonal transformation and isotropic scaling
- Use case: compare fine-tuned vs. base model representations; diagnose where divergence occurs
- Social science application: do two models trained on different corpora form the same semantic structure?

**Instructor Notes:**
CKA is a principled way to ask "do these two models think about this domain the same way?" It's invariant to the specific basis of the representation space, meaning it compares the geometry of what is represented rather than the specific numbers. Social science application: you can use CKA to test whether a model fine-tuned on conservative media and a model fine-tuned on liberal media form qualitatively different representations of political concepts, which would be evidence for meaningful ideological conditioning.

---

## Slide 17: Model Context Protocol — Standardizing Tool Use

**Visual Description:**
A system architecture diagram with three layers. Top layer: "LLM / AI Agent." Middle layer: "MCP Interface" — a standardized plug with labeled ports: Tools, Resources, Prompts. Bottom layer: three server blocks representing real-world services — "File System," "Web Search / Browser," "Database / API." Arrows show request/response flow between layers. A code snippet on the right:
```python
# MCP tool definition
@mcp.tool()
def search_pubmed(query: str, max_results: int = 10):
    """Search PubMed for biomedical literature."""
    results = pubmed_api.search(query, retmax=max_results)
    return [{"pmid": r.pmid, "title": r.title,
             "abstract": r.abstract} for r in results]
```
Caption: "MCP: the USB standard for AI tools."

**Bullet Points / Text:**
- MCP (Anthropic, 2024): standardized protocol for AI agents to call external tools
- Three primitives: Tools (actions), Resources (data access), Prompts (context templates)
- Enables: web search, file I/O, database queries, API calls — without ad-hoc engineering
- Social science application: agents that read court records, pull census data, query PubMed

**Instructor Notes:**
The USB analogy works well here: before USB, every device had a different connector. MCP is trying to do for AI tools what USB did for computer peripherals — create a standard interface so that any agent can use any tool. Walk through the code snippet and emphasize that from the model's perspective, a tool is just a function with a name and description; the model decides when to call it and with what arguments. The power for social science is that you can now build agents that query real databases, call real APIs, and act on real information.

---

## Slide 18: Tool-Using Agents in Practice

**Visual Description:**
A trace diagram showing a tool-using agent solving the task: "Find the three most-cited papers on algorithmic bias published after 2020, download their abstracts, and summarize the key claims." The trace shows:
1. Agent thought: "I need to search for papers → call search_scholar()"
2. Tool call: `search_scholar(query="algorithmic bias", year_min=2020, sort_by="citations")`
3. Tool response: JSON list of 3 papers with titles, DOIs, abstracts
4. Agent thought: "Now I need to synthesize → generate summary"
5. Agent output: structured summary with citations
Each step is in a colored box (yellow for thoughts, blue for tool calls, green for tool responses, white for final output). An arrow shows the ReAct loop.

**Bullet Points / Text:**
- ReAct pattern: Reason → Act → Observe → Reason → ...
- Agent decides which tool to call, with what arguments, based on context
- Tool use grounds agent outputs in real, verifiable data
- Social science applications: literature review, data collection, longitudinal monitoring

**Instructor Notes:**
Tool-using agents are the practical realization of what it means for an AI to be an assistant rather than a text generator. When an agent can call a search API, read a file, or run code, it can perform genuine research tasks rather than just generating plausible-sounding text. The ReAct loop is the key architectural pattern: after every tool call, the model reads the result and decides what to do next. This is essentially how a human researcher works — search, read, synthesize, search again.

---

## Slide 19: Bias in Training Data — The Obermeyer et al. Case

**Visual Description:**
A reproduction of the key figure from Obermeyer et al. (2019). A scatter plot with x-axis "Predicted Risk Score" (percentile 0–100) and y-axis "Actual Health Status" (number of active chronic conditions). Two colored regression lines: one for Black patients (orange), one for White patients (blue). The lines diverge significantly — at the same predicted risk score, Black patients have substantially more active conditions than White patients. A callout box quotes the paper: "At a given risk score, Black patients have substantially more active chronic conditions than White patients." A second panel shows the algorithmic decision boundary: patients above the 97th percentile receive care management. At this threshold, 17.7% of high-risk Black patients are enrolled vs. 46.5% of comparably sick White patients.

**Bullet Points / Text:**
- Algorithm used healthcare costs as proxy for health needs
- Black patients received less care historically → lower cost scores → lower predicted risk
- Systematic underestimation of illness severity for Black patients
- Lesson: proxy labels encode historical inequity — audit every label source

**Instructor Notes:**
This is one of the most important empirical papers in AI ethics and it lives squarely in the social science domain. The algorithm wasn't designed to be racist; it was designed to be accurate. But accuracy was measured against a biased historical record. The label — healthcare spending — was itself the product of decades of unequal access to care. This is why sampling and label design are moral choices, not just technical ones. Ask students: what are the proxy labels in your own research, and what historical inequities might they encode?

---

## Slide 20: Semantic Biases in Word Embeddings — Caliskan et al. (2017)

**Visual Description:**
A 2D UMAP projection of word embeddings with word categories color-coded: blue circles = "career" words (executive, salary, professional), pink triangles = "family" words (home, parents, children), green squares = "male" names (John, David, Michael), red diamonds = "female" names (Mary, Sarah, Emily). The projection shows clear clustering: female names near family words; male names near career words. A math callout box shows the WEAT test statistic:
```
WEAT effect size d:
d = [mean_sim(A,C) - mean_sim(A,D)] - [mean_sim(B,C) - mean_sim(B,D)]
    ─────────────────────────────────────────────────────────────────
                      std_dev(all similarities)

where A=male names, B=female names, C=career, D=family
Observed d = 1.81 (p < 0.001)
```

**Bullet Points / Text:**
- Word Embedding Association Test (WEAT): statistical test for stereotypical associations
- Male names cluster with career words; female names with family words — mirrors IAT results
- Effect sizes match psychological IAT tests: AI inherits human cultural biases
- Implication: every model trained on natural language text contains social stereotypes

**Instructor Notes:**
Caliskan et al.'s contribution was to show that bias isn't a bug to be fixed — it's a structural feature of models trained on human language. Human language encodes human culture, including its inequities. The WEAT test is elegant because it maps directly onto the Implicit Association Test from psychology, allowing direct comparison between human cognitive bias and AI representational bias. The effect sizes they find are comparable to human IAT scores, which raises a deep question: are we measuring the model's bias, or are we measuring our own bias as reflected in the training corpus?

---

## Slide 21: Model Collapse — The Ouroboros Problem

**Visual Description:**
A circular diagram showing the collapse cycle: (1) "Real human data" → train LLM Generation 1 → (2) "Synthetic data generated by Gen 1" → train LLM Generation 2 → (3) "Synthetic data from Gen 2" → ... → (n) "Generation n: distribution collapsed, tails gone." On the right side, two probability density curves: Gen 1 output (wide, fat-tailed, resembling a normal distribution with real variance) vs. Gen 10 output (narrow spike, thin tails — Gaussian with collapsed variance). Below: citation "Shumailov et al. (2024). AI Models Collapse when Trained on Generated Data. Nature." A snake eating its own tail (ouroboros) as a small decorative icon.

**Bullet Points / Text:**
- Model collapse: iteratively training on AI-generated data destroys distributional diversity
- Statistical tails disappear — the model loses its understanding of rare, unusual cases
- Web content increasingly generated by AI → contamination problem for future training
- Social science implication: AI-generated research corpora will homogenize AI research outputs

**Instructor Notes:**
Shumailov et al.'s 2024 Nature paper identifies a genuine existential threat to the current training paradigm. If we train models on web data, and the web increasingly contains AI-generated content, each generation of models is less diverse than the last. The tails of the distribution — which contain the rare, unusual, creative cases — collapse first. For social science, this is especially alarming: the very phenomena we care about most (unusual events, minority perspectives, fringe ideologies) are exactly what gets lost first in model collapse.

---

## Slide 22: Model Collapse — What the Data Shows

**Visual Description:**
A replication of the key figure from Shumailov et al. (2024). A 2D scatter plot showing text samples from successive model generations projected onto two principal components. Generation 0 (human text): a wide cloud of points in many colors, spread across the plane. Generations 1–5: progressively tighter clusters, converging toward a central point. Generation 9: a tight cluster covering perhaps 10% of the original spread. Annotated with arrows showing the collapse trajectory. A second panel shows histograms of n-gram frequency distributions at each generation, showing the progressive narrowing of vocabulary diversity.

**Bullet Points / Text:**
- Across generations: diversity collapses, outlier cases disappear
- Text becomes repetitive, average, "safe" — loses cultural richness
- Already detectable in current LLMs trained on post-2022 web data
- Mitigation: curate and preserve human-generated corpora; mark synthetic data

**Instructor Notes:**
This slide makes model collapse concrete and empirical. The visualization of the PCA scatter plot is striking — students can see the progressive collapse in a single image. The key social science framing is: AI-generated content is culturally homogenizing in the same way that globalized media homogenizes local culture. The question is whether this is a manageable engineering problem (preserve human corpora, label synthetic content) or a fundamental epistemological problem for the next generation of AI research.

---

## Slide 23: Active Learning for Social Science Annotation

**Visual Description:**
A comparison chart showing three annotation strategies applied to the same classification task (content moderation for online hate speech, 10,000 unlabeled examples, budget of 200 labels). Line chart with x-axis = "number of labeled examples" (0 to 200) and y-axis = "F1 score on held-out test set." Three lines:
- Random sampling (baseline): slow, steady improvement
- Uncertainty sampling: steeper early improvement, plateaus earlier
- Query-by-committee with 3 models: steepest improvement, best final score
Annotation cost callout: "200 human labels with active learning ≈ 800 labels with random sampling."

**Bullet Points / Text:**
- Active learning reaches higher accuracy with fewer annotations
- Critical for social science: expert annotation is expensive (legal, medical, ethnographic)
- Uncertainty sampling: start with any trained model, query borderline cases
- Committee methods: train 3–5 models with different initializations; query where they disagree

**Instructor Notes:**
Connect this back to the beginning of the lecture: sampling isn't just about choosing training data from a corpus. Active learning is about continuously making strategic sampling decisions as you build your labeled dataset. The ROI is dramatic in practice — a project that would require 1000 expert annotations can often achieve equivalent performance with 250 strategically chosen annotations. This is especially important for projects involving sensitive data where annotation budgets are tight (clinical records, court cases, conflict reports).

---

## Slide 24: Combining Tools — A Research Agent Pipeline

**Visual Description:**
An end-to-end pipeline diagram for a social science research agent. Each component is a labeled box connected by arrows:
1. "Research Question" (user input)
2. "Literature Search Agent" (MCP + PubMed/Scholar tools)
3. "Document Retrieval & Chunking" (RAG pipeline with FAISS)
4. "Fine-Tuned Domain LLM" (LoRA-adapted Mistral-7B)
5. "Benchmark Evaluation Module" (CKA + task-specific metrics)
6. "Output: Structured Research Report" (with citations, confidence scores)
Color-coded by component type: blue for data components, green for model components, orange for evaluation components, white for I/O. A small clock icon showing estimated runtime: "~2 minutes per query."

**Bullet Points / Text:**
- Research agents integrate: retrieval, fine-tuned LLMs, evaluation, and tool-use
- Each component from this week feeds into the pipeline
- Benchmarking and CKA run continuously to detect performance degradation
- MCP tools enable real-time data access — agents that stay current

**Instructor Notes:**
This is the "putting it all together" slide. Everything from today — sampling (how was the fine-tuning data collected?), LoRA (how was the domain LLM adapted?), benchmarking (how do we evaluate outputs?), CKA (does the fine-tuned model represent the domain differently from the base model?), and MCP tools (how does the agent access current data?) — feeds into this pipeline. This is a realistic architecture for the kind of research assistant agents many students will build for their final projects.

---

## Slide 25: Module 1 Preview — Sampling Lab

**Visual Description:**
A Jupyter notebook screenshot (mockup) showing Module 1 structure. The notebook has three visible cells:
- Cell 1 header: "## 1.1 Sampling Strategy Comparison" with a table comparing SRS, stratified, and systematic on the same corpus
- Cell 2 shows a class imbalance histogram before/after SMOTE
- Cell 3 shows an active learning learning curve (F1 vs. annotation budget)
On the right: a data card showing the dataset to be used — "Political Speech Corpus: 12,847 speeches, 1990–2024, 12 demographic categories, class imbalance ratio 8:1 (anti-immigration vs. neutral)."

**Bullet Points / Text:**
- Dataset: political speech corpus with demographic imbalance
- Tasks: implement SRS, stratified, systematic; apply SMOTE + Tomek Links; run active learning loop
- Deliverable: learning curve plot + 200-word reflection on sampling choices
- Hint: stratify by speaker gender × party × decade before any other sampling

**Instructor Notes:**
Orient students to the homework module before they leave. The political speech corpus is a realistic social science dataset with meaningful class imbalance — anti-immigration rhetoric is a minority class, but it's the class we most care about detecting accurately. Walk through the expected deliverables: a learning curve plot showing active learning outperforming random, and a reflection connecting sampling choices to the social science question.

---

## Slide 26: Module 2 Preview — Fine-Tuning Lab

**Visual Description:**
A table with four columns and five rows comparing LoRA ranks on a fine-tuning task:
| LoRA Rank | Trainable Params | Val Perplexity | GPU Hours |
|-----------|-----------------|----------------|-----------|
| r=1       | 16,384          | 34.2           | 0.3       |
| r=4       | 65,536          | 22.7           | 0.8       |
| r=8       | 131,072         | 18.4           | 1.4       |
| r=16      | 262,144         | 17.1           | 2.6       |
| r=64      | 1,048,576       | 16.8           | 9.1       |
A callout: "r=8 is the sweet spot for most social science corpora." Below, a code snippet header showing the key PEFT library call.

**Bullet Points / Text:**
- Fine-tune GPT-2 (or Mistral-7B with QLoRA) on your project corpus
- Compare perplexity at r = 4, 8, 16 — find the efficiency-performance Pareto frontier
- Evaluate generation quality qualitatively: does the fine-tuned model "sound right"?
- Advanced: implement QLoRA to fine-tune a larger model on a consumer GPU

**Instructor Notes:**
Students should pick a corpus closely related to their final project for this module — it will directly inform their project work. The rank comparison table shows that r=8 is almost always the sweet spot: going to r=64 recovers only marginal perplexity gains at 6× the compute cost. Encourage students to do both the quantitative perplexity comparison and the qualitative generation evaluation — sometimes perplexity doesn't capture what matters most about domain fit.

---

## Slide 27: Module 3 Preview — Benchmarking Lab

**Visual Description:**
A flowchart for designing a custom benchmark. Five steps in colored boxes:
(1) "Define the social science task" — e.g., "Classify political ideology of editorial"
(2) "Collect/annotate test set" — 100 labeled examples, human annotated, stratified
(3) "Define metric(s)" — F1 for binary ideology, macro-F1 for 5-point scale
(4) "Test baseline models" — GPT-4, base Llama-3, fine-tuned Llama-3
(5) "Compute CKA between base and fine-tuned representations" — identify where they diverge
Below: a sample CKA heatmap (10×10 layer-by-layer comparison) with clearly divergent later layers highlighted.

**Bullet Points / Text:**
- Build a benchmark that tests what your research actually cares about
- Contamination check: verify test set did not appear in any model's training data
- CKA analysis: which layers change most during fine-tuning?
- Report: benchmark description + metric justification + CKA visualization

**Instructor Notes:**
The benchmark design exercise is where students learn to think like evaluators rather than just builders. The CKA analysis is particularly instructive: it usually shows that early layers (syntactic processing) change very little during fine-tuning, while later layers (semantic representation) change substantially. This is exactly what you'd expect if LoRA is correctly targeting the semantic content of the domain adaptation rather than changing fundamental language processing.

---

## Slide 28: Module 4 Preview — MCP Tool Agent

**Visual Description:**
A two-panel display. Left: a Python code snippet showing an MCP tool server definition:
```python
from mcp import FastMCP
mcp = FastMCP("Social Science Tools")

@mcp.tool()
def query_census(state: str, year: int, variable: str) -> dict:
    """Query US Census Bureau API for demographic data."""
    ...

@mcp.tool()
def search_congress_votes(bill_id: str) -> list[dict]:
    """Retrieve congressional voting records by bill ID."""
    ...
```
Right: an agent conversation trace showing the agent using these tools to answer "How did demographic change in Arizona between 2010 and 2020 correlate with shifts in congressional voting?"

**Bullet Points / Text:**
- Build an MCP tool server for your research domain (census, court, news, academic API)
- Test with a research question that requires multi-step tool use
- Log all tool calls and measure: accuracy, latency, tool-use efficiency
- Reflection: what can the agent not do that a human researcher could?

**Instructor Notes:**
The MCP homework is the most immediately practical for final projects. Students who build a working MCP tool server for their research domain essentially have a research assistant that can continuously update itself with new data. Encourage creativity in tool design — the best tools are those that access data sources that are otherwise laborious to query (federal agency APIs, court records systems, historical newspaper archives). The reflection question about limitations is important: tool-using agents are powerful but brittle when data is unstructured or when the task requires judgment that doesn't reduce to a function call.

---

## Slide 29: Discussion — The Epistemics of AI-Generated Data

**Visual Description:**
A discussion prompt slide with a large quote in the center on a dark blue background:
> *"If we train models on AI-generated data, and AI-generated data reflects AI models' existing representations of the world, then the next generation of models will know the world as AI described it, not as humans experienced it."*

Below the quote: three discussion questions in white text on separate colored panels:
- "What kinds of social phenomena are most at risk of being lost in model collapse?"
- "How should journals and datasets mark AI-generated vs. human-generated content?"
- "Is there a version of model collapse that isn't purely bad — could it select for accuracy?"

**Bullet Points / Text:**
- Model collapse is an epistemic problem, not just a technical one
- The tails of distributions contain minority voices, rare experiences, dissenting views
- Social science has always worried about self-fulfilling prophecies — AI amplifies this risk

**Instructor Notes:**
Give students 10 minutes for this discussion in small groups, then bring the class together. The self-fulfilling prophecy angle is particularly rich: sociologists have long studied how classification systems shape the things they classify (Hacking's "looping effects"). AI model collapse is a technological version of this — the model's representation of the social world becomes the training data for the next model, progressively erasing the gap between the map and the territory.

---

## Slide 30: Week 5 Closing — Domain Agents as Social Science Infrastructure

**Visual Description:**
A full-width infographic showing the complete conceptual arc of Week 5. A winding road graphic with five milestones labeled with icons: (1) sampling funnel, (2) LoRA matrix, (3) benchmark scorecard, (4) MCP tool plug, (5) model collapse warning sign. At the end of the road: "A fine-tuned, well-sampled, properly benchmarked, tool-equipped agent — the workhorse of social science AI." Below the road: a quote and citation for each of the four social science readings: Obermeyer (healthcare bias), Caliskan (embedding bias), Shumailov (model collapse), Chu (media diet prediction).

**Bullet Points / Text:**
- Fine-tuned agents outperform general models on domain-specific tasks — by design
- Rigorous sampling and benchmarking are what make AI outputs scientifically defensible
- Tool-use grounds agents in the world — transforms them from text generators to research tools
- But: every design choice embeds assumptions — sample carefully, audit constantly

**Instructor Notes:**
Close with the core thesis of the week: the gap between a generic LLM and a genuine social science research instrument is filled by the technical practices we covered today. Sampling, fine-tuning, benchmarking, and tool-use are not optional extras — they are the methods section of AI social science. Remind students that next week (Week 6) we go inside these agents — not just what they do, but why they do it, and how we can change what they represent.

---

*End of Week 5 Slides | Next: Week 6 — Auto-Encoders, Interpretability, and Steering Agents*
