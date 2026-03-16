# Week 1 Lecture Script: Deep Learning and Social Agents
## AI Agents for Social Science and Society 2026
**Date:** January 9, 2026
**Instructor:** James A. Evans (script may be delivered by a visiting instructor)
**Notebook:** `Week_1_Intro_NNs.ipynb` (247 cells)
**Duration:** 3 hours (1:30–4:20 PM)

---

## Instructor Preparation Checklist

- [ ] Clone the course repo and open `Week_1_Intro_NNs.ipynb` in Colab or a local Jupyter server
- [ ] Pre-download the Covertype dataset (Cell 17 will pull from UCI; verify the URL is live)
- [ ] Load MNIST in Cell 41 ahead of time to avoid delays
- [ ] Have slides or a whiteboard for diagrams of feedforward networks
- [ ] Print or display the reading list: Evans "Preface: How to Think with Deep Learning"; Bubeck et al. 2023 "Sparks of AGI"; Park et al. 2023 "Generative Agents"
- [ ] Test GPU availability in your runtime; note that Module 4 uses image data and benefits from GPU

---

## Section 1: Introduction — What Is an AI Agent, and Why Should Social Scientists Care? (30 min)

### Opening Remarks (5 min)

**Say:**
> "Welcome to AI Agents for Social Science and Society. This course is built on a single, provocative premise: artificial intelligence has moved from a tool we point at data to something that increasingly acts in the world — reasoning, predicting, collaborating, and even simulating human behavior. Over the next ten weeks, you are going to build those systems from the ground up, and you are going to apply them to genuine social science questions."

> "Before we write a single line of code today, I want to establish something. Every technical topic in this course has a social science consequence. We are not here to become machine learning engineers, although you will develop those skills. We are here because the same models that run recommendation engines and content moderation also encode cultural biases, reproduce social hierarchies, and increasingly stand in for human subjects in research. Understanding how they work, from the inside, is now essential social science literacy."

**Transition:**
> "So: what is an AI agent? Let's start with the most basic possible version."

---

### What Is an AI Agent? (10 min)

**Write on board:**
```
Input → [Model] → Output
```

**Say:**
> "In the most minimal sense, an agent is a system that takes in information about the world and produces a response. A thermometer is not an agent — it only observes. A thermostat is the simplest possible agent — it observes and acts. What makes modern AI agents qualitatively different is that the mapping from input to output is learned from data, at massive scale, in ways that capture something approximating human judgment."

**Write on board:**
```
Week 1:  Static networks (inputs → outputs)
Week 3:  Prompted LLMs (context → social behavior)
Week 7:  RL agents (learn from consequences)
Week 9:  Aligned, safe agents (values + guardrails)
```

**Say:**
> "The arc of this course follows the progressive enrichment of what an 'agent' can do. This week we start at the foundation: the feedforward neural network. By Week 3 you will be simulating voting behavior in the 2020 presidential election. By Week 9 you will be auditing those systems for safety and bias. Every building block we lay today carries through to the end."

**Ask students:**
> "Before we go further — how many of you have trained a neural network before? How many have interacted with an LLM API? And how many have used an AI tool to help with a research task — searching literature, summarizing text, generating hypotheses?"

*(Pause for a show of hands. Use this to gauge the room. If most students have LLM experience but limited training experience, emphasize that the course will teach them what is actually happening inside the systems they already use.)*

---

### The Social Science Motivation (15 min)

**Say:**
> "Let me give you two examples that will orient everything else we do today."

**Example 1 — Neural Life Trajectories:**
> "Savcisens and colleagues, in a 2024 paper published in Nature Computational Science, trained a transformer model on the complete administrative records of more than 6 million Danish citizens — birth, education, employment, health, income, residence, across entire lifespans. They treated each life as a sequence of tokens, exactly the way a language model treats text. The model could predict future health events, educational attainment, and mortality with striking accuracy. This is not a futuristic scenario. This is a deployed research methodology. The question for social science is: what does it mean to predict a life? What does the model capture that classical regression misses? What does it miss entirely?"

**Example 2 — Generative Agents (Park et al. 2023):**
> "Park and colleagues built a simulated social environment — a village called Smallville — populated entirely by LLM-powered agents with persistent memory. These agents held conversations, formed relationships, spread rumors, organized events. None of their social behavior was hand-programmed. It emerged from the same kind of transformer architecture you will build intuitions about today. The question for social science is: when simulated agents produce recognizably human-like social dynamics, what have we learned? And what are the risks of mistaking the simulation for the reality?"

**Ask students:**
> "Evans's preface argues that deep learning doesn't just provide better predictions — it changes the kind of questions we can ask. What questions do you think become newly askable with these systems that weren't askable before?"

*(Expected responses: questions about individual-level trajectories rather than population averages; questions about emergent social dynamics; questions about how language encodes ideology and culture. Write the most interesting responses on the board and connect them to specific weeks in the syllabus.)*

---

## Section 2: Neural Network Fundamentals — From Perceptrons to Deep Networks (45 min)

### The Perceptron and Feedforward Architecture (15 min)

**Write on board:**
```
x₁ ──┐
x₂ ──┤ → Σ(wᵢxᵢ + b) → activation → ŷ
x₃ ──┘
```

**Say:**
> "Every neural network, no matter how large, is built from this basic unit: the neuron. It takes a weighted sum of its inputs, adds a bias term, and passes the result through an activation function. What makes this powerful is composition — many neurons, arranged in layers, with each layer's output becoming the next layer's input."

**Write on board:**
```
Layer 0 (input)  →  Layer 1 (hidden)  →  Layer 2 (hidden)  →  Layer 3 (output)
  [x₁, x₂, ..., xₙ]      [h₁...h_k]          [h₁...h_m]          [ŷ₁...ŷ_c]
```

**Say:**
> "The crucial insight — and I want you to hear this carefully, because it is the mathematical foundation for everything else — is the Universal Approximation Theorem. A neural network with even a single hidden layer containing enough neurons can approximate any continuous function on a bounded domain to arbitrary precision. Depth isn't strictly necessary for representational power. But in practice, deep networks — many layers with fewer neurons per layer — learn more efficiently. They share features across layers, building hierarchical representations: edges become corners become shapes become objects, or in text: characters become words become phrases become semantics."

**Ask students:**
> "Suppose you want to predict whether a person votes in the next election from their demographic profile — age, income, education, party registration, zip code. Why might a neural network outperform logistic regression here?"

*(Expected response: neural networks can learn non-linear interactions among features without you specifying them explicitly. A 45-year-old, high-income suburban Republican behaves differently from a 45-year-old, high-income urban Republican. Logistic regression needs you to specify those interaction terms. The network learns them.)*

---

### Activation Functions (10 min)

**Write on board:**
```
Sigmoid:  σ(x) = 1/(1+e^(-x))    → output in (0,1), saturates at extremes
Tanh:     tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)  → output in (-1,1), centered
ReLU:     f(x) = max(0, x)        → sparse, no saturation for positives
```

**Say:**
> "Activation functions introduce non-linearity — without them, any stack of linear layers collapses to a single linear transformation and depth buys you nothing. Sigmoid and tanh were the historical workhorses. ReLU — Rectified Linear Unit — became dominant because it does not saturate for positive inputs, making gradients flow cleanly through many layers. When you run the MNIST experiments in Module 1, you will see this difference empirically: sigmoid networks train slowly and plateau early; ReLU networks converge faster and reach higher accuracy."

---

### Forward Pass, Loss, and Backpropagation (10 min)

**Write on board:**
```
Forward:   x → ŷ = f_θ(x)
Loss:      L = CrossEntropy(ŷ, y)
Backward:  ∂L/∂θ  (chain rule through all layers)
Update:    θ ← θ - η · ∂L/∂θ
```

**Say:**
> "Training is a three-step loop you will run thousands of times. The forward pass produces a prediction. The loss function measures how wrong it is. Backpropagation computes the gradient of that loss with respect to every single weight in the network, using the chain rule of calculus. Then gradient descent takes a step in the direction that reduces loss. This is the complete algorithm. Everything else in this course — attention mechanisms, RL, alignment — is an elaboration on this foundation."

---

### Optimization: SGD, Momentum, Adam (10 min)

**Write on board:**
```
SGD:        θ ← θ - η·∇L
Momentum:   v ← βv + ∇L;  θ ← θ - η·v
Adam:       Combines momentum + adaptive per-parameter learning rates
```

**Say:**
> "Vanilla SGD treats every weight identically and uses a fixed learning rate. The problem: some directions in parameter space are steep, others are nearly flat. Momentum helps by accumulating velocity — the optimizer keeps moving in directions it has been moving, smoothing out oscillations. Adam goes further by tracking both the mean and variance of past gradients for each parameter individually, so it can adjust its effective step size. In practice, Adam is the default. You will compare all three in Module 3."

**Instructor Note:** *This is a good moment to draw an analogy: gradient descent as hiking down a foggy mountain with only local information. Momentum is like having some inertia. Adam is like having a GPS that adjusts your stride length based on how steep the terrain has been.*

---

## Section 3: Code Walkthrough — `Week_1_Intro_NNs.ipynb` (60 min)

### Setup and Module 1 — Feedforward Network Variations (20 min)

**Say:**
> "Open the notebook. Cells 0–3 are the course header and module checklist. Read the module summaries in Cell 2 before diving into code — they describe what each module teaches and what the homework tasks ask you to do."

**Navigate to Cell 5.**

**Say:**
> "The first import block. Note we are using `torch`, `torch.nn`, and `torch.optim`. PyTorch is our primary framework throughout the course — it gives us explicit control over the computational graph, which matters when we get to interpretability in Week 6."

**Navigate to Cell 6.**

**Say:**
> "Here is the simplest possible network: one hidden layer of 200 neurons, a sigmoid activation, and a binary output. Notice the structural pattern: `nn.Linear` for the weight matrix plus bias, then an explicit activation. This three-line class definition captures everything the perceptron diagram showed."

**Show the code:**
```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(200, 200)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(200, 1)
```

**Say:**
> "Cell 8 shows Adam with default hyperparameters — learning rate 0.001, beta_1 of 0.9, beta_2 of 0.999. These are nearly always reasonable starting points. Cell 9 defines Binary Cross-Entropy loss for a two-class problem."

**Navigate to Cells 14–17 (Covertype dataset).**

**Say:**
> "For the deep network experiments, we use the UCI Covertype dataset: 581,000 samples of forest terrain, each described by 54 features including soil type, elevation, and hillshade. The task is to predict which of seven forest cover types is present. This is a structured tabular dataset — the kind social scientists work with all the time, whether it is census data, survey responses, or administrative records."

**Navigate to Cell 30–33.**

**Say:**
> "The baseline model in Cell 31 defines a network with embedding layers for categorical features and linear layers for numeric features — the Wide & Deep architecture pattern from Google's 2016 recommendation system paper. This is directly applicable to any social science dataset with a mix of categorical variables (race, education, party ID) and continuous variables (income, age, vote share)."

**Navigate to Cells 40–46 (MNIST activation comparison).**

**Say:**
> "Now the core Module 1 experiment. Cell 42 defines an MLP that takes the same architecture but swaps the activation function. Cell 44 trains it with different learning rates. Cell 46 tests different depths. Run Cells 41–46 now and watch the training curves."

**Ask students:**
> "What do you notice about sigmoid versus ReLU at depth 5? At what depth does sigmoid start to fail, and what does that failure look like in the loss curve?"

*(Expected observation: sigmoid networks show vanishing gradients at deeper architectures — loss plateaus early or becomes noisy. ReLU maintains clean convergence.)*

---

### Module 2 — Weight Initialization (20 min)

**Navigate to Cell 53 header.**

**Say:**
> "Module 2 addresses a subtle but critical problem: if you initialize your weights randomly with no thought to scale, deep networks can fail to train at all. Let me show you why."

**Navigate to Cell 57 (`DeepNetPoorInit`).**

**Say:**
> "A ten-layer network with zero initialization. Every layer receives the same gradient and learns nothing — a pathological case. With random Gaussian initialization at large scale, you get exploding activations. Small scale gives you vanishing gradients."

**Write on board:**
```
Xavier/Glorot:   W ~ Uniform[-√(6/(n_in+n_out)), +√(6/(n_in+n_out))]
                 Goal: preserve variance through sigmoid/tanh layers

He/Kaiming:      W ~ Normal(0, √(2/n_in))
                 Goal: compensate for ReLU zeroing half the neurons
```

**Say:**
> "Xavier initialization — derived analytically by Glorot and Bengio in 2010 — keeps the variance of activations stable across layers for symmetric activations like tanh. He initialization adjusts for ReLU's property of setting half its inputs to zero. PyTorch applies He initialization by default for Linear layers with ReLU, which is why you rarely need to think about this in practice. But knowing why matters — especially when you are debugging a deep network that refuses to learn."

**Navigate to Cell 83 (Batch Normalization).**

**Say:**
> "Batch Normalization is the practical solution to residual initialization problems. It normalizes each layer's input across the batch to have mean zero and variance one, then applies learnable scale and shift parameters. The result: training is faster, less sensitive to initialization choices, and effective at much greater depth. Cell 87 shows the comprehensive comparison — run it and note the accuracy improvement with batch norm."

**Say:**
> "The connection to LLMs: GPT has 96+ layers. Without careful initialization and normalization, it would be untrainable. Everything in Cell 91 explains how these Module 2 techniques carry directly to modern transformer architectures."

---

### Module 3 — Optimization (10 min, overview only)

**Navigate to Cells 109–116.**

**Say:**
> "Module 3 is a comparative study of optimizers: vanilla SGD, SGD with momentum, RMSprop, and Adam. Cell 116 runs all four on MNIST and plots training and test error simultaneously. For the homework, you will apply this to your own dataset."

**Key point to make:**
> "Notice the argument structure in Cell 105 — the `main(args)` function takes a dictionary of hyperparameters. This is good practice: separating your experimental configuration from your model code makes it easy to run systematic comparisons. You should do this in your homework."

**Navigate to Cell 138 (ResNet, optional).**

**Say:**
> "For students who want to go deeper: Cell 138 introduces Residual Networks. The key insight is the skip connection — output is `F(x) + x` rather than just `F(x)`. This allows gradients to bypass layers entirely, enabling effective training at hundreds of layers. We will see skip connections again in transformer architectures in Week 2."

---

### Module 4 — Regularization (10 min, overview)

**Navigate to Cell 156 header.**

**Say:**
> "Module 4 addresses overfitting. Cell 176 contains one of my favorite experiments in the notebook: training three identical networks on three versions of the AnimalFaces dataset — true labels, partially randomized labels, and fully randomized labels. The network with true labels generalizes. The network with random labels memorizes. This is an empirical demonstration that neural networks have sufficient capacity to memorize arbitrary training data, which tells us that generalization is not guaranteed — it must be achieved through regularization."

**Say:**
> "The two primary regularization techniques you need to know: L2 weight decay adds a penalty proportional to the squared magnitude of all weights to the loss function, discouraging large weights. Dropout randomly zeros out neurons during training, forcing the network to learn redundant representations that are more robust. Cell 167's training function accepts regularization functions as arguments — study how they are applied."

---

## Section 4: Student Code Presentations (30 min)

**Instructor Note:** *Beginning in Week 2, students will present code from the previous week's homework. In Week 1, use this time for a structured discussion of the notebook experiments. If students have already done preliminary exploration, invite them to share their results from Cells 40–46.*

**Ask students:**
> "Who ran the activation function comparison? What did you find? Did the results match the theoretical prediction — that ReLU would outperform sigmoid on deeper architectures?"

**Ask students:**
> "Did anyone apply the network to a dataset related to their final project? What challenges did you encounter in preprocessing the data to feed into a PyTorch `Dataset` class?"

*(If time permits, have two or three students walk through their preprocessing code. The `ForestDataset` class in Cell 28 is a good template to reference.)*

---

## Section 5: Discussion — What Can Neural Networks Model About Social Life? (15 min)

### Structured Discussion

**Say:**
> "I want to end today with the hardest question, not the easiest one. We have seen that neural networks are universal approximators. They can, in principle, learn any function. The question for social science is not 'can they do it?' but 'should we trust what they learn, and what does it mean?'"

**Write on board:**
> Core tension: **Predictive power ≠ causal understanding**

**Ask students:**
> "Farrell, Gopnik, Shalizi, and Evans argue that large AI models are 'cultural and social technologies' — not merely tools but participants in social processes. What do you think they mean by that? How is a language model different from a calculator?"

*(Expected responses: it encodes human values and biases; it shapes how people think and communicate; it reflects the culture of its training data; it participates in knowledge production. Guide students toward the idea that the model's outputs are not neutral — they reflect statistical patterns in a corpus produced by particular humans at a particular moment in history.)*

**Ask students:**
> "The Bubeck et al. paper on 'Sparks of AGI' argues that GPT-4 shows early signs of general intelligence. What does that claim mean — and what would it mean for social science if it were true?"

*(This should generate substantive debate. Do not resolve it — leave it as an open provocation for the rest of the course.)*

---

## Closing Summary (5 min)

**Say:**
> "Let me give you the three things to take away from today. First: neural networks are function approximators trained by gradient descent. The entire architecture — feedforward, transformers, diffusion models — is variations on this theme. Second: depth works because it enables hierarchical feature learning, but it introduces new problems: vanishing gradients, overfitting, sensitivity to initialization. We have tools for each of these. Third: every design choice in a neural network — architecture, loss function, regularization — is also a social choice when the model is applied to human data. The 'predictive accuracy' of a model on demographic features is a methodological question and an ethical question simultaneously."

> "Next week we move to text: word embeddings, transformers, and what it means for a model to learn the geometry of culture from language. Read Vaswani et al. — 'Attention Is All You Need' — before class. It is the most important paper in modern AI, and you should be able to sketch the architecture from memory by Week 2."

---

## Homework Briefing (5 min)

**Say:**
> "The homework is `Week_1_Intro_NNs.ipynb`. You must complete at least three of the four modules. Module 1 — the feedforward network variations — is recommended for everyone; it is the foundation. Choose two more based on your background: if you feel strong on optimization, do Module 2 on initialization; if you want more practice with training dynamics, do Module 3 on optimizers."

> "The most important requirement: apply one module to a dataset related to your final project. This is not optional. You should be thinking about your final project from Day 1, and every homework is an opportunity to build toward it. If you do not have a dataset yet, come to Avi's lab session Tuesday at 11am — he will help you identify appropriate data sources."

**Write on board:**
```
Lab: Avi Oberoi — Tuesday 11am–12pm
Homework due: Before Week 2 (Jan. 16)
Required: Complete 3 of 4 modules + apply one to your project
```

---

## Timing Guide

| Section | Content | Time |
|---|---|---|
| 1 | Introduction: AI agents and social science motivation | 30 min |
| 2 | Neural network fundamentals: perceptrons, activations, backprop, optimization | 45 min |
| 3 | Code walkthrough: Modules 1–4 of notebook | 60 min |
| Break | — | 10 min |
| 4 | Student code presentations / discussion | 30 min |
| 5 | Discussion: neural networks and social life | 15 min |
| Closing | Summary + homework briefing | 10 min |
| **Total** | | **3 hr** |

---

## Instructor Notes

**Common student confusion points:**
1. The difference between the loss function and the activation function — emphasize that the loss is computed at the output only; activations are at every hidden layer.
2. Why we need both training and validation sets — the validation set tunes hyperparameters; the test set estimates generalization. Students who have only done classical statistics may conflate these.
3. The distinction between a deep network and a wide network — depth enables hierarchical representations; width adds capacity at a single level of abstraction.

**If students seem confused by backpropagation:**
> "PyTorch handles the calculus for you — `loss.backward()` computes all gradients automatically. What you need to understand conceptually is that the gradient of the loss with respect to each weight tells us which direction to adjust that weight to reduce error. The chain rule makes this tractable through many layers."

**Connections to upcoming weeks:**
- The transformer (Week 2) is a deep network with a specialized architecture for sequences
- Digital doubles (Week 3) use the same basic API call pattern as the toy examples in Cell 6
- Mechanistic interpretability (Week 6) looks inside the network at the activations and gradients you are computing here
