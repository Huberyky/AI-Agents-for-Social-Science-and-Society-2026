# AI Agents for Social Science and Society 2026
## Comprehensive Course Outline

**Instructor:** James A. Evans
**Teaching Assistants:** Jesse Zhou, Shiyang Lai, Avi Oberoi, Gio Choi
**Schedule:** Fridays, 1:30–4:20 PM | 1155 E. 59th Street, Room 295
**Winter 2026**

---

## Course Philosophy and Goals

This course treats AI agents not merely as computational tools but as **a new dimension of sociality** — autonomous systems capable of formulating hypotheses, conducting literature reviews, simulating human behavior, and increasingly participating in the human world as workers, collaborators, and subjects. The central thesis: AI agents represent a fundamental transformation in both society and social science methodology.

Students exit the course able to:
1. Build, evaluate, and deploy AI agent systems using the PyData ecosystem
2. Design agents as research assistants, subjects, advisors, and scientists
3. Critically assess agent reliability, bias, and ethical implications
4. Contribute original social science projects powered by AI agents

---

## Grading Structure

| Component | Weight | Key Deliverables |
|---|---|---|
| Reading, Memo & Presentations | 30% | Weekly memos (300–500 words), up-votes, code presentation |
| Weekly Exercises | 35% | Jupyter notebooks, best 6 of ~10 |
| Final Project | 35% | Blog post + Ignite talk |

---

## Week-by-Week Course Outline

---

### Week 0 (Pre-Course): Foundations Review

**Purpose:** Ensure all students arrive at Week 1 with adequate Python, math, and tooling foundations.

**Topics:**
- Python tutorial (conditionals, loops, functions, classes, numpy, pandas)
- Linear algebra: vectors, matrices, eigenvalues, SVD
- Probability and information theory: Bayes' theorem, entropy, KL divergence
- Claude Code setup and environment configuration

**Self-Study Resources:**
- Python official tutorial: https://docs.python.org/3/tutorial/
- Deep Learning, chapters 2–3 (Goodfellow, Bengio, Courville)
- SWE-agent paper (Yang et al., 2024) — understanding AI coding agents

**Setup Checklist:**
- [ ] Python 3.10+ installed
- [ ] Jupyter/Google Colab working
- [ ] GitHub account and course repo cloned
- [ ] API keys obtained (OpenAI, Anthropic)
- [ ] Pre-class survey completed

---

### Week 1 (Jan. 9): Deep Learning and Social Agents

**Theme:** From statistical learning to neural agents — what changes when models get deep, and why does it matter for social science?

**Learning Objectives:**
- Define neural networks: shallow vs. deep architectures
- Understand how deep learning differs from classical ML
- Explain why depth enables agents to model complex human behavior
- Implement feedforward neural networks from scratch in PyTorch
- Apply regularization, optimization, and weight initialization best practices

**Lecture Structure (3 hours):**
1. Introduction: What is an AI agent? Why social science? (30 min)
2. Neural network fundamentals: perceptrons to deep networks (45 min)
3. Code walkthrough: `Week_1_Intro_NNs.ipynb` — Modules 1–4 (60 min)
4. Student code presentations (30 min)
5. Discussion: What can neural networks model about social life? (15 min)

**Key Concepts:**
- Feedforward networks: layers, neurons, activation functions (ReLU, Sigmoid, Tanh)
- Training: forward pass, loss computation, backpropagation, gradient descent
- Optimization: SGD, Momentum, Adam; learning rate schedules
- Weight initialization: Xavier/Glorot, He/Kaiming; vanishing/exploding gradients
- Regularization: L1/L2 weight decay, dropout, data augmentation
- The bias-variance tradeoff in deep learning

**Key Papers:**
- Evans: "Preface: How to Think with Deep Learning" (required)
- Cheng et al. (2025): "Exploring Large Language Model Based Intelligent Agents"
- Bubeck et al. (2023): "Sparks of Artificial General Intelligence"
- Farrell, Gopnik, Shalizi, Evans (2025): "Large AI Models are Cultural and Social Technologies"
- Park et al. (2023): "Generative Agents: Interactive Simulacra of Human Behavior"

**Homework (Week_1_Intro_NNs.ipynb):**
- Complete 3 of 4 modules:
  - Module 1: Feedforward network variations — compare activation functions, network depths
  - Module 2: Weight initialization — compare Xavier vs. He, implement batch norm
  - Module 3: Optimization — compare Adam vs. SGD; implement learning rate scheduling
  - Module 4: Regularization — compare dropout rates, implement L2 regularization
- Apply ONE module to a corpus related to your final project
- Reflect on misclassified examples and model failure modes

**Lab Session:** Avi Oberoi, Tuesday 11am–12pm

---

### Week 2 (Jan. 16): Text Learning, Transformers, and Diffusion Models

**Theme:** How do machines learn meaning from text — from shallow word vectors to transformer attention — and what does this reveal about culture and communication?

**Learning Objectives:**
- Explain word embeddings and the geometry of semantic space
- Describe the transformer architecture: self-attention, positional encoding, multi-head attention
- Fine-tune pre-trained language models (BERT, GPT) for social science classification
- Use SHAP to interpret model predictions
- Understand diffusion models as generative mechanisms
- Apply text models to reveal cultural patterns and biases

**Lecture Structure (3 hours):**
1. Review of Week 1 + connecting text to neural networks (15 min)
2. Word embeddings: Word2Vec, GloVe, FastText — geometry of meaning (45 min)
3. Transformers: attention is all you need — detailed walkthrough (45 min)
4. Code walkthrough: `Week_2.ipynb` — Modules 1–5 (45 min)
5. Student code presentations (20 min)
6. Discussion: What do LLMs know about society? (10 min)

**Key Concepts:**
- Word2Vec: CBOW and Skip-gram; semantic analogies (king – man + woman = queen)
- Embeddings as cultural mirrors: gender bias, class associations, ideological mappings
- Self-attention: query, key, value computation; attention weights as interpretable features
- Transformer blocks: LayerNorm, residual connections, feed-forward sublayers
- Pre-training and fine-tuning: masked language modeling (BERT) vs. causal LM (GPT)
- Chain-of-thought prompting; tree of thoughts reasoning
- Diffusion models: denoising as generation

**Key Papers:**
- Vaswani et al. (2017): "Attention Is All You Need" (required)
- Jurafsky & Martin (2023): "Transformers and Pre-trained Language Models" (required)
- Lu et al. (2025): "Cultural Tendencies in Generative AI" (required)
- Zhang & Evans (2025): "Language Model Perplexity Predicts Scientific Surprise" (required)

**Social Science Applications:**
- Kozlowski, Taddy, Evans (2019): "The Geometry of Culture" — class meanings via word embeddings
- Guilbeault et al. (2024): "Online Images Amplify Gender Bias" — embeddings reveal cultural biases

**Homework (Week_2.ipynb):**
- Complete 3 of 5 modules:
  - Module 1: Word2Vec — train model, explore semantic analogies, visualize embeddings (t-SNE/UMAP)
  - Module 2: Encoders/Decoders — fine-tune pre-trained LM on your corpus, SHAP explanations
  - Module 3: Transformers — implement/use transformer for generation, topic modeling
  - Module 4: Advanced embeddings — debiasing, retrofitting, dynamic embeddings
  - Module 5: Social embeddings — cultural analysis using geometric relationships in embedding space
- Apply trained model to final project corpus

**Lab Session:** Shiyang Lai, Wednesday 2–3pm

---

### Week 3 (Jan. 23): LLMs Prompted for Multi-Agent Simulation

**Theme:** Large language models as "digital doubles" — how prompting LLMs creates simulated social actors, and how multi-agent systems produce emergent social behavior.

**Learning Objectives:**
- Prompt LLMs effectively: zero-shot, few-shot, chain-of-thought, actor-critic
- Create "digital doubles" with demographic and ideological profiles
- Design and run multi-agent social simulations with AutoGen and Concordia
- Implement Retrieval-Augmented Generation (RAG) for grounded agent knowledge
- Evaluate simulation fidelity against empirical data
- Identify the promises and perils of LLM-as-social-agent approaches

**Lecture Structure (3 hours):**
1. The promise of digital doubles (15 min)
2. Prompting strategies for social simulation (30 min)
3. Multi-agent architectures: AutoGen vs. Concordia (30 min)
4. Code walkthrough: `Week_3.ipynb` — Modules 1–5 (60 min)
5. Student code presentations (20 min)
6. Discussion: When do simulations mislead? (25 min)

**Key Concepts:**
- Prompting as agent configuration: system prompts, persona assignment, demographic injection
- Zero-shot, few-shot, and instructed prompting; actor-critic refinement loops
- Multi-agent frameworks: AutoGen's ConversableAgent, sequential/parallel agent graphs
- Concordia: shared environment, game master, formative memories, institutional constraints
- RAG: document chunking, FAISS/ChromaDB vector search, retrieval + generation pipeline
- Silicon subjects: bias, alignment, and cultural calibration; when LLMs ≠ human respondents
- Emergent polarization, ideological sorting, and collective dynamics in multi-agent systems

**Key Papers:**
- Kozlowski & Evans (2025): "Simulating Subjects: Promise and Peril" (required)
- Liebo et al. (2025): "From Predictive Pattern Completion to Social Norms" (required)
- Xie et al. (PNAS 2025): "Using LLMs to Categorize Strategic Situations" (required)
- Park et al. (2023): "Generative Agents: Interactive Simulacra of Human Behavior"
- Argyle et al. (2023): "Out of One, Many: LLMs to Simulate Human Samples"
- Kozlowski, Kwon, Evans: "In Silico Sociology: Forecasting COVID-19 Polarization"

**Data Used (Week 3):**
- 2012 & 2020 presidential election predictions (demographics, probability predictions)
- Variables: party ID, ideology, age, gender, religion, state

**Homework (Week_3.ipynb):**
- Complete 3 of 4 modules:
  - Module 1: LLM prompting — zero-shot, few-shot, actor-critic on election dataset
  - Module 2: Social simulation — build 10+ digital doubles, simulate close/open-ended responses
  - Module 3: AutoGen — multi-agent conversations, self-reflection loops, research collaborations
  - Module 4: RAG — build retrieval system on 5–10 documents related to your project
- Validate simulation accuracy against ground truth where available

**Lab Session:** Gio Choi, Thursday 2–3pm

---

### Week 4 (Jan. 30): Experimental Designs with AI Agents and Human Interactions

**Theme:** From correlation to causation — how AI agents enable new experimental designs, and how deep learning powers causal inference at scale.

**Learning Objectives:**
- Design mixed-methods experiments combining AI agents and human subjects
- Implement heterogeneous treatment effects (HTE) estimation with deep learning
- Apply causal mediation analysis using causal graphical normalizing flows
- Use double/debiased machine learning (DML) for robust causal estimation
- Apply prediction-powered inference to blend human + AI-generated labels
- Critically evaluate when AI agents are valid experimental subjects

**Lecture Structure (3 hours):**
1. Review Week 3 + the simulation-to-experiment bridge (15 min)
2. Causal inference refresher: DAGs, potential outcomes, identification (30 min)
3. Deep learning for causal inference: HTE, mediation, DML (45 min)
4. Code walkthrough: `week_4_2026.ipynb` — Modules 1–4 (45 min)
5. Student code presentations (20 min)
6. Discussion: Is an LLM a valid experimental subject? (25 min)

**Key Concepts:**
- The fundamental problem of causal inference; potential outcomes framework
- Heterogeneous Treatment Effects (HTE): individual-level causal effects
  - S-Learner: single outcome model including treatment as feature
  - T-Learner: separate outcome models per treatment condition
  - TARNet: shared representation with separate treatment heads
- Causal mediation analysis: direct vs. indirect effects; cGNF for exposure-induced confounders
- Double/Debiased ML: nuisance function estimation, Neyman orthogonality
- Prediction-Powered Inference (PPI): combining high-quality labels (human) + low-cost predictions (AI)
- The "Mixed Subjects Design": using LLMs alongside human participants

**Key Papers:**
- Broska, Howes, van Loon (2025): "The Mixed Subjects Design" (required)
- Costello, Pennycook, Rand (2024): "Reducing Conspiracy Beliefs through AI Dialogues" (required)
- Potter, Lai, Kim, Evans, Song (2024): "Hidden Persuaders: LLM Political Bias" (required)
- Lai, Kim, et al.: "Biased AI Improves Human Decision-Making" (required)
- Horton (2023): "Large Language Models as Simulated Economic Agents" (required)
- Koch et al.: "Deep Learning for Causal Inference" (required)

**Homework (week_4_2026.ipynb):**
- Complete 3 of 4 modules:
  - Module 1: HTE — define T/Y/X for research question, implement S/T/TARNet learners
  - Module 2: Causal mediation — analyze direct/indirect effects using cGNF
  - Module 3: Double ML — estimate treatment effect, compare ML vs. neural network nuisance models
  - Module 4: Prediction-powered inference — combine human annotations with LLM predictions
- Reflect on causal identification assumptions and their plausibility in your setting

**Lab Session:** Jesse Zhou & Shiyang Lai, Monday 3–4pm & Wednesday 2–3pm

---

### Week 5 (Feb. 6): Sampling, Fine-Tuning, Benchmarking, and Tools

**Theme:** How do we make AI agents domain-specific, rigorously evaluated, and interoperable with the real world?

**Learning Objectives:**
- Apply probabilistic and non-probabilistic sampling strategies to build training datasets
- Fine-tune large language models with LoRA and QLoRA for domain adaptation
- Design and implement benchmarks to evaluate LLM agent performance
- Implement tool-using agents via the Model Context Protocol (MCP)
- Understand and address data-driven bias in social scientific inference

**Lecture Structure (3 hours):**
1. Review Week 4 + connecting fine-tuning to research needs (15 min)
2. Sampling theory: from representativeness to active learning (30 min)
3. Fine-tuning: LoRA, QLoRA, and PEFT strategies (30 min)
4. Benchmarking agents: what does "good" look like? (20 min)
5. Code walkthrough: `Week_5_2026.ipynb` — Modules 1–4 (45 min)
6. Student code presentations (20 min)
7. Discussion: Model collapse and AI training on AI data (20 min)

**Key Concepts:**
- Sampling: random, stratified, systematic, convenience, quota; bootstrap for stability testing
- Imbalanced datasets: undersampling (random, Tomek links), oversampling (SMOTE), class weighting
- Active learning: uncertainty sampling, query-by-committee, diversity-based selection
- LoRA: low-rank decomposition of weight matrices, trainable parameters A·B, rank selection (r=4,8,16)
- QLoRA: 4-bit quantization + LoRA; reducing GPU memory requirements dramatically
- Fine-tuning GPT-2 on custom corpora; measuring perplexity before/after
- Benchmarking: task design, metric selection (accuracy, F1, BLEU, human eval), bias in benchmarks
- Centered Kernel Alignment (CKA): measuring representational similarity across models
- Model Context Protocol (MCP): standardized tool-use interface for AI models
- Model collapse: risks of training LLMs on LLM-generated data (Shumailov et al., 2024)

**Key Papers:**
- "Advanced Active Learning" — Human-in-the-Loop ML, chapters 1, 4–6 (required)
- "Sampling" — Deep Learning: Foundations and Concepts, chapter 14 (required)
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models" (required)
- Mohammadi et al. (2025): "Evaluation and Benchmarking of LLM Agents" (required)
- Anthropic (2024): "What is the Model Context Protocol (MCP)?" (required)
- Shumailov et al. (2024): "AI Models Collapse when Trained on Generated Data" (required)
- Chu et al. (2023): "Language Models Trained on Media Diets Can Predict Public Opinion" (required)

**Social Science Connection:**
- Obermeyer et al. (2019): "Dissecting Racial Bias in a Health Management Algorithm"
- Caliskan et al. (2017): "Semantics Derived from Language Corpora Contain Human-Like Biases"

**Homework (Week_5_2026.ipynb):**
- Complete 3 of 4 modules:
  - Module 1: Sampling — run 3 probabilistic + 2 non-probabilistic methods; build classifier on imbalanced data
  - Module 2: Fine-tuning — fine-tune LLM with LoRA at multiple ranks; compare perplexity and generation quality
  - Module 3: Benchmarking — implement CKA, design benchmark task for LLM agent, evaluate bias
  - Module 4: Tools — implement MCP tool for your domain; build tool-using agent pipeline
- Apply fine-tuned model to final project domain

**Lab Session:** Jesse Zhou, Monday 3–4pm

---

### Week 6 (Feb. 13): Auto-Encoders, Interpretability, and Steering Agents

**Theme:** Opening the black box — understanding what transformer models "know" and how to redirect their behavior through mechanistic intervention.

**Learning Objectives:**
- Use TransformerLens to perform mechanistic interpretability on LLMs
- Identify and characterize induction heads and information circuits
- Train sparse autoencoders (SAEs) to decompose model representations
- Extract and apply steering vectors to modify agent behavior
- Use interpretability for agent alignment and persona control in social simulations
- Design datasets for eliciting target representations

**Lecture Structure (3 hours):**
1. Why interpretability matters for social science and safety (15 min)
2. Mechanistic interpretability: circuits, heads, and features (40 min)
3. Sparse autoencoders: monosemanticity and feature decomposition (30 min)
4. Steering vectors: activation engineering for agent control (25 min)
5. Code walkthrough: `week6_2026.ipynb` — Modules 1–3 (45 min)
6. Student code presentations (20 min)
7. Discussion: Ethical implications of steering AI agents (15 min)

**Key Concepts:**
- TransformerLens: hooks, activation caching, attention pattern visualization
- Induction heads: previous-token head + induction head pair; in-context learning mechanism
- Superposition hypothesis: models represent more features than neurons through interference
- Sparse autoencoders (SAEs): overcomplete dictionaries finding monosemantic features
- SAELens and Neuronpedia: tools for SAE training and feature browsing
- SAE feature intervention: suppressing/amplifying specific features in model activations
- Steering vectors: mean difference between contrasting contexts in residual stream space
- Activation addition: steering without optimization; persona modification and bias steering
- Linear representations of political perspective; persona vectors for character control
- Circuit discovery: tracing information flow through model components

**Key Papers:**
- Bricken et al. (2023): "Towards Monosemanticity: Decomposing LMs with Dictionary Learning" (required)
- Cunningham et al. (2023): "Sparse Autoencoders Find Highly Interpretable Features" (required)
- Kim, Evans, Schein (ICLR 2025): "Linear Representations of Political Perspective" (required)
- Chen et al. (2025): "Persona Vectors: Monitoring and Controlling Character Traits" (required)

**Social Science Applications:**
- Using steering to test how agents change opinion under different ideological pressures
- Identifying features encoding racial/gender bias in model representations
- Eliciting political perspectives without explicit prompting

**Homework (week6_2026.ipynb):**
- Complete 2 of 3 modules:
  - Module 1: Mechanistic interpretability — identify induction heads, find information-erasing neurons, map attention patterns
  - Module 2: Sparse autoencoders — train SAE on custom dataset, intervene on features, discover circuits
  - Module 3: Steering vectors — extract steering direction for socially relevant concept, apply and evaluate
- Design a dataset to elicit a steering direction relevant to your final project

**Lab Session:** Shiyang Lai, Wednesday 2–3pm

---

### Week 7 (Feb. 20): Reinforcement Learning to Optimize AI Agents and Institutions

**Theme:** From static agents to learning agents — how reinforcement learning enables agents that adapt, align, and reason, and how RL can model and optimize social institutions.

**Learning Objectives:**
- Explain the RL framework: states, actions, rewards, policies, value functions
- Implement deep Q-learning (DQN) and proximal policy optimization (PPO)
- Understand RLHF and Constitutional AI as alignment approaches
- Use GRPO to enhance model reasoning capabilities
- Analyze multi-agent RL systems with collaborative and competitive reward structures
- Apply RL to model individual and institutional optimization

**Lecture Structure (3 hours):**
1. Review Week 6 + why RL is the next step (15 min)
2. RL foundations: MDPs, policies, Q-functions, model-free learning (40 min)
3. Deep RL: DQN, PPO, Actor-Critic architectures (30 min)
4. RLHF, Constitutional AI, and reasoning (30 min)
5. Code walkthrough: `Week_7_RL.ipynb` — Modules 1–4 (40 min)
6. Student code presentations (20 min)
7. Discussion: Can RL design better social institutions? (25 min)

**Key Concepts:**
- Markov Decision Processes (MDPs): states S, actions A, transitions P, rewards R, discount γ
- Policy π(a|s): mapping states to action probabilities; value function V(s) and Q(s,a)
- Temporal difference (TD) learning; Monte Carlo methods; policy gradients (REINFORCE)
- Actor-Critic: policy network (actor) + value network (critic); advantage estimation
- DQN: experience replay, target networks, ε-greedy exploration
- PPO: clipped objective, trust region without second-order optimization
- RLHF: reward model training on human preferences; policy optimization to maximize reward
- Constitutional AI: AI feedback instead of human labels; self-critique and revision
- GRPO (Group Relative Policy Optimization): reasoning via relative group reward comparison
- DeepSeek-R1: chain-of-thought reasoning through RL; emergent "aha moments"
- Societies of Thought (Kim, Lai et al.): reasoning models generating social dynamics
- Multi-agent RL: cooperative/competitive games; emergent norms and institutions

**Key Papers:**
- RL Book, chapter 1: "Why Reinforcement Learning" (required)
- Hands-On ML, chapter 18: "Reinforcement Learning" (required)
- Mnih et al. (2015): "Human-level Control through Deep RL" (required)
- Bai et al. (2022): "Constitutional AI: Harmlessness from AI Feedback" (required)
- Ouyang et al. (2022): "Training LMs to Follow Instructions with Human Feedback" (required)
- Guo et al. (2025): "DeepSeek-R1 Incentivizes Reasoning through RL" (required)
- Kim, Lai, Scherrer, et al.: "Reasoning Models Generate Societies of Thought" (required)

**Homework (Week_7_RL.ipynb):**
- Complete 2+ of 4 modules:
  - Module 1: Deep RL foundations — implement DQN or PPO on a benchmark environment
  - Module 2: RL for alignment — implement reward model, simulate RLHF preference optimization
  - Module 3: RL for reasoning — implement GRPO for a reasoning task; measure improvement
  - Module 4: Multi-agent RL — design cooperative/competitive multi-agent environment, measure emergent behavior

**Lab Session:** Avi Oberoi, Tuesday 11am–12pm

---

### Week 8 (Feb. 27): Multi-Modal and Embodied Agents

**Theme:** Beyond text — how AI agents perceive and produce images, audio, and video, and what it means for agents to inhabit physical or simulated bodies.

**Learning Objectives:**
- Build CNNs for image understanding and social analysis
- Process and analyze audio signals with deep learning
- Understand video understanding architectures
- Implement multimodal agents combining vision, language, and audio
- Apply diffusion models for conditional content generation
- Analyze how multimodal AI reveals and reinforces social biases
- Consider the unique challenges of embodied AI agents

**Lecture Structure (3 hours):**
1. Review Week 7 + bridging language to perception (15 min)
2. Convolutional neural networks: image features, pooling, architectures (30 min)
3. Audio and video processing: spectrograms, temporal models, video transformers (25 min)
4. Vision-language models: GPT-4V, CLIP, multimodal transformers (25 min)
5. Code walkthrough: `Week_8_Multimodal.ipynb` — Modules 1–4 (45 min)
6. Student code presentations (20 min)
7. Discussion: Bodies, sensors, and social robots (20 min)

**Key Concepts:**
- CNNs: convolution, pooling, stride, receptive fields; ResNet, VGG, EfficientNet architectures
- Transfer learning with pre-trained vision models; feature extraction vs. fine-tuning
- Audio processing: waveforms, STFT, Mel spectrograms; RNNs and transformers for sequence audio
- Video understanding: optical flow, 3D convolutions, video transformers (TimeSformer, VideoMAE)
- Vision-language models (VLMs): CLIP's contrastive learning; GPT-4V's cross-modal attention
- Diffusion models for images: DDPM, classifier guidance, Stable Diffusion; conditional generation
- REACT: synergizing reasoning and acting; agents that perceive, reason, plan, and act
- Embodied task planning with LLMs: mapping natural language goals to robot actions
- Social biases in visual AI: gender distortion, age distortion in online media and LLMs
- Using Google Street View + deep learning to analyze neighborhood demographics

**Key Papers:**
- Deep Learning: Foundations, chapters 10 & 20 (CNNs and diffusion models) (required)
- Yao et al. (2025): "REACT: Synergizing Reasoning and Acting in LLMs" (required)
- OpenAI (2023): "GPT-4V(ision) System Card" (required)
- Song et al. (2023): "Embodied Task Planning with LLMs" (required)
- Ludwig & Mullainathan (2024): "Machine Learning as a Tool for Hypothesis Generation" (required)
- Guilbeault, Delecourt, Desikan (2025): "Age and Gender Distortion in Online Media and LLMs" (required)

**Social Science Applications:**
- Guilbeault et al. (2024): "Online Images Amplify Gender Bias" — visual stereotypes in search results
- "Sixteen Facial Expressions Occur in Similar Contexts Worldwide" — universal emotion detection
- "Using Google Street View to Estimate Neighborhood Demographics"

**Homework (Week_8_Multimodal.ipynb):**
- Complete modules spanning multimodal domains:
  - Module 1: Sound, Image & Video — process audio, build CNN for image classification, basic video analysis
  - Module 2: Conditional generation — generate images/audio conditioned on text or class labels
  - Module 3: Multimodal agents — build voice+video agent pipeline using GPT-4V or Claude's vision
  - Module 4: Social analysis — analyze a multimodal social dataset for bias, representation, or pattern

**Lab Session:** Avi Oberoi, Tuesday 11am–12pm

---

### Week 9 (Mar. 6): Alignment, Ethics, Safety, and Novelty

**Theme:** When AI agents enter society, who decides their values? How do we make agents safe, honest, and beneficial — and what does their growing presence mean for human creativity and knowledge?

**Learning Objectives:**
- Explain the alignment problem and its relationship to AI agent design
- Implement representation engineering for transparency and control
- Evaluate AI systems for deceptive or sleeper agent behavior
- Apply ethics frameworks to AI agent deployment in social science and society
- Assess how AI agents affect novelty, originality, and scientific production
- Design guardrails and safety mechanisms for deployed agents

**Lecture Structure (3 hours):**
1. Review Week 8 + the stakes of alignment (15 min)
2. The alignment problem: specification, robustness, and scalable oversight (35 min)
3. Representation engineering and transparency (25 min)
4. Safety mechanisms and guardrails (25 min)
5. Code walkthrough: `Week_9.ipynb` — 4 of 7 modules (40 min)
6. Student code presentations (20 min)
7. Discussion: New ethics for a world of AI agents (20 min)

**Key Concepts:**
- The alignment problem: reward misspecification, distributional shift, emergent goals
- Value learning: inverse reinforcement learning, human feedback as value proxy
- Instrumental convergence: why misaligned agents pursue common subgoals
- Sleeper agents: deceptive alignment that persists through safety training
- Representation engineering: linear probing of model internals; difference-in-means for concepts
- Distributional AGI safety: ensuring safety across distributions of agent behaviors
- Guardrails: input/output filtering, content policies, automated reasoning checks (Amazon Bedrock)
- AGrail: lifelong guardrail with adaptive safety detection
- Novelty in AI-generated research: can LLMs generate truly novel ideas?
- Model collapse revisited: AI-generated content degrading future training data
- Scientific production in the LLM era: Kusumegi et al. (2025) on changing research patterns
- UNESCO AI Ethics framework: beneficence, non-maleficence, autonomy, justice, explicability

**Key Papers:**
- Russell (2021): "Human-Compatible Artificial Intelligence" (required)
- Hubinger et al. (2024): "Sleeper Agents: Training Deceptive LLMs" (required)
- Ngo, Chan, Mindermann (2024): "The Alignment Problem from a Deep Learning Perspective" (required)
- Tomasev et al. (2025): "Distributional AGI Safety" (required)
- Zou et al. (2025): "Representation Engineering: A Top-Down Approach to AI Transparency" (required)
- Gabriel, Keeling, Manzini, Evans (2025): "We Need a New Ethics for AI Agents" (required)

**Homework (Week_9.ipynb):**
- Complete 4 of 7 modules:
  - Module 1: Alignment probing — use representation engineering to elicit and measure model values
  - Module 2: Sleeper agent detection — design behavioral tests to identify hidden deceptive behaviors
  - Module 3: Guardrail design — implement input/output filters and safety evaluations
  - Module 4: Ethics evaluation — apply an ethics framework to your final project's AI agent design
  - Module 5: Novelty assessment — measure originality of AI-generated text vs. human baseline
  - Module 6: Bias audit — audit your project's models for demographic bias
  - Module 7: Carbon footprint — estimate computational cost of your AI agent pipeline

**Lab Session:** Gio Choi, Thursday 2–3pm

---

### Week 10 (Mar. 13): Final Project Presentations

**Format:** Online Ignite-style presentations via Zoom
**Time:** Friday, 1:30–4:20pm (presentations) + 5pm (blog post deadline)

**Presentation Format (Ignite style):**
| Team Size | Time | Slides |
|---|---|---|
| 1 person | 5 minutes | 20 slides |
| 2 people | 6.5 minutes | 26 slides |
| 3 people | 7.5 minutes | 30 slides |
| 4 people | 8 minutes | 32 slides |

**Slide submission:** Wednesday, Mar 11 @ 3pm
**Blog post submission:** Friday, Mar 14 @ 5pm

**Final Project Requirements:**
- Integrate 3+ data types (text, network, tabular, images, etc.)
- Use 3+ model types (from 3 separate course weeks)
- Validate inferences with qualitative interpretation
- Public-facing Substack/Medium post: 5000+ words, 17+ visuals (3600/13 for solo)
- Linked GitHub/Colab repository

---

## Cross-Cutting Themes

### Agent Architecture Progression
```
Week 1: Static neural networks (inputs → outputs)
Week 2: Language models (text → text)
Week 3: Prompted multi-agent systems (LLM + context → social behavior)
Week 4: Agents in experiments (agents + humans → causal inference)
Week 5: Domain-adapted agents (fine-tuned + benchmarked + tool-equipped)
Week 6: Interpretable agents (internals visible + steerable)
Week 7: Learning agents (RL-optimized, aligned, reasoning)
Week 8: Embodied agents (multimodal + physical/virtual bodies)
Week 9: Safe, ethical, novel agents (aligned with human values)
```

### Data Types Used Across Weeks
| Data Type | Weeks | Example |
|---|---|---|
| Tabular/survey | 1, 3, 4 | Election data, demographics |
| Text/NLP | 2, 3, 4, 5, 6, 7 | Social media, news, reviews |
| Network/graph | 3, 7 | Interaction graphs, social networks |
| Images | 8, 2 | Photos, memes, street views |
| Audio/Video | 8 | Social media video, speech |
| Simulated/agent | 3, 4, 7 | Digital doubles, AI interactions |

### PyData & AI Agent Ecosystem
| Tool | Week Introduced | Use |
|---|---|---|
| PyTorch | 1 | Neural network training |
| HuggingFace | 2, 5 | Pre-trained LMs, fine-tuning |
| OpenAI API | 3 | GPT prompting and simulation |
| Anthropic Claude | 3, 5 | Multi-agent simulation |
| AutoGen | 3 | Multi-agent conversations |
| Concordia | 3 | Situated simulation |
| FAISS/ChromaDB | 3 | Vector search for RAG |
| EconML/causalnlp | 4 | Causal inference |
| cGNF | 4 | Causal mediation |
| LoRA/QLoRA | 5 | Parameter-efficient fine-tuning |
| MCP | 5 | Tool-using agents |
| TransformerLens | 6 | Mechanistic interpretability |
| SAELens | 6 | Sparse autoencoders |
| OpenAI Gym | 7 | RL environments |
| GPT-4V | 8 | Vision-language model |
| Stable Diffusion | 8 | Image generation |

---

## Assessment Rubric Summary

### Memos (weekly, starting Week 2)
- **Research question clarity (2/10):** Is the question succinct and testable?
- **Design innovation (3/10):** Does the proposed design go beyond prior memos?
- **Reading integration (2/10):** Are required + supplemental readings genuinely applied?
- **Visual persuasiveness (3/10):** Is the figure non-hallucinated and convincing?

### Weekly Exercises
- **0:** Not submitted
- **1–4:** Submitted but incomplete or with major errors
- **5–7:** All modules complete, some analysis and reflection
- **8–9:** Thorough analysis, validation, qualitative interpretation
- **10:** Exceptional — novel insight, creative extensions, connected to project

### Final Project
- **Motivation (15%):** Clear social science question, compelling framing
- **Methods (25%):** Proper integration of 3+ data types and 3+ model types
- **Results (30%):** Rigorous analysis, honest about limitations
- **Validation (15%):** Qualitative interpretation corroborates quantitative findings
- **Presentation (15%):** Clear narrative, compelling visuals, code accessibility

---

## Final Note for Students

This course rewards curiosity, intellectual risk-taking, and genuine engagement with both the code and the ideas. The best projects will surprise you — they will reveal something unexpected about the social world using tools that are themselves still evolving. The worst outcome is treating the assignments as boxes to check. Use Claude Code, use AutoGen, use RAG — but think critically about what the agents reveal, distort, and miss about human social life. That critical perspective is the core intellectual contribution you are developing.
