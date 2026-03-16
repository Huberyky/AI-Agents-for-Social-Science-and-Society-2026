# Week 1: Deep Learning and Social Agents
## AI Agents for Social Science and Society 2026
### Ignite-Style Slide Deck — 30 Slides

---

## Slide 1: The Question That Drives This Course

**Visual Description:**
Split-screen image. Left side: a classic sociological survey form (clipboard, checkboxes, demographic grid). Right side: a neural network diagram glowing with activation flows. A large bold arrow between them with the text: "Same question. Different tools."

**Bullet Points / Text:**
- Who are we? What do we do? Why?
- AI agents are now answering these questions — at scale

**Instructor Notes:**
Welcome students by framing the course's central tension: social science has always been about understanding human behavior, and AI agents are the newest — and most powerful — instruments we have. Emphasize that every concept today has a social science consequence, not just a technical one. The arrow between the two images is intentional: we are not replacing social science, we are supercharging it.

---

## Slide 2: What Is an AI Agent?

**Visual Description:**
Minimalist diagram on a dark background:
```
 WORLD ──────► [OBSERVE] ──► [DECIDE] ──► [ACT] ──────► WORLD
                  ▲                              │
                  └──────────────────────────────┘
                           (feedback loop)
```
Below the diagram: three icons — a thermometer (no arrow back), a thermostat (one arrow), a ChatGPT logo (full loop with memory cloud above).

**Bullet Points / Text:**
- Thermostat: the world's simplest agent
- LLM: observation → reasoning → action → consequence
- The loop is what makes it social

**Instructor Notes:**
Walk through the three examples carefully. A thermometer observes but never acts; a thermostat acts but doesn't learn; a modern LLM-powered agent observes, builds a model, acts, and can update. The loop and memory are what make these systems genuinely agentive — and genuinely sociologically interesting.

---

## Slide 3: AI Agents Are Entering Society

**Visual Description:**
A 3x3 grid of high-quality icons/thumbnails:
1. Customer service chatbot (headset icon)
2. Hiring resume screener (document with checkmark)
3. Content moderator (flag icon)
4. News recommendation engine (newspaper)
5. Scientific literature reviewer (microscope + text)
6. Political opinion simulator (ballot box)
7. Social companion robot (heart + circuit)
8. Autonomous financial trader (stock chart)
9. Clinical diagnostic assistant (stethoscope)

Caption: "Which of these is doing social science? All of them."

**Bullet Points / Text:**
- AI agents are workers, companions, gatekeepers
- They encode values, reproduce biases, shape outcomes

**Instructor Notes:**
Ask students to pick one of these nine roles and think for 30 seconds about what data it was trained on, what it optimizes for, and who it might harm. This primes the course's recurring theme: deployment is a social act. Every cell in this grid is a potential final project topic.

---

## Slide 4: The Arc of This Course

**Visual Description:**
A rising staircase diagram, each step labeled and color-coded:
```
Step 9: Safe, Ethical Agents ──────────────────────── [WEEK 9]
Step 8: Multimodal + Embodied ──────────────────────  [WEEK 8]
Step 7: Reinforcement Learning ─────────────────────  [WEEK 7]
Step 6: Interpretability + Steering ────────────────  [WEEK 6]
Step 5: Fine-Tuning + Benchmarking ─────────────────  [WEEK 5]
Step 4: Experiments + Causation ────────────────────  [WEEK 4]
Step 3: Multi-Agent Simulation ─────────────────────  [WEEK 3]
Step 2: Text + Transformers ────────────────────────  [WEEK 2]
Step 1: Neural Networks ════════════════════════════  [YOU ARE HERE]
```

**Bullet Points / Text:**
- Each week builds on the last
- Week 3: simulate an election. Week 9: audit an agent for safety.

**Instructor Notes:**
Emphasize that this is not a survey course — it is a cumulative build. The feedforward network they implement today is the conceptual ancestor of the GPT-style transformer in Week 2 and the multi-agent simulation in Week 3. Students who understand the foundations will get far more out of the later weeks.

---

## Slide 5: The Perceptron — Where It All Started

**Visual Description:**
Clean diagram of a single perceptron:
```
  x₁ ──[w₁]──┐
  x₂ ──[w₂]──┼──► [ Σ wᵢxᵢ + b ] ──► [ σ(z) ] ──► ŷ
  x₃ ──[w₃]──┘
```
Below: portrait of Frank Rosenblatt (1957) alongside a photo of the original Mark I Perceptron hardware at Cornell. Caption: "1957. One neuron. The seed of everything."

**Bullet Points / Text:**
- Input × weight → sum → activation → output
- McCulloch & Pitts (1943): the neuron as logic gate
- Rosenblatt (1957): the first learning machine

**Instructor Notes:**
The perceptron is both historically important and conceptually foundational. Spend a moment on Rosenblatt's original claim — that the machine could recognize patterns the way a human does — and the backlash from Minsky and Papert's "Perceptrons" (1969) which showed single-layer networks couldn't learn XOR. That limitation is precisely what depth solves, which is the story of the next three slides.

---

## Slide 6: One Neuron Can't Learn XOR

**Visual Description:**
Side-by-side scatter plots:
- Left: AND gate — blue dots separable by a single straight line
- Right: XOR gate — blue and red dots interspersed, no single straight line can separate them
Caption: "A single line can't solve every problem. Neither can a single neuron."

Below, small diagram:
```
(0,0)→0   (0,1)→1
(1,0)→1   (1,1)→0
```

**Bullet Points / Text:**
- Linear decision boundary = fundamental limit
- XOR requires a hidden layer
- Depth = capacity to carve non-linear spaces

**Instructor Notes:**
This is the conceptual motivation for deep networks. The XOR problem is beautifully simple — show students that adding just one hidden layer with two neurons solves it. This is the moment where the geometric intuition of neural networks clicks: layers are folding and reshaping the input space until it becomes linearly separable.

---

## Slide 7: From Shallow to Deep

**Visual Description:**
Three network diagrams side by side, drawn with consistent node-and-edge style:
- Left: Perceptron (1 layer, 3 inputs, 1 output) — labeled "Shallow"
- Center: 2-layer MLP (3→4→1) — labeled "Less shallow"
- Right: Deep network (3→8→8→8→1) — labeled "Deep"

Below each, a caption:
- "Can learn: linear boundaries"
- "Can learn: smooth curves"
- "Can learn: almost anything (in theory)"

**Bullet Points / Text:**
- Depth = representational power
- Universal approximation theorem (Cybenko 1989)
- But power requires training

**Instructor Notes:**
The Universal Approximation Theorem guarantees that a sufficiently wide one-hidden-layer network can approximate any continuous function — but it doesn't tell us how to train it, how wide "sufficient" is, or whether it generalizes. Deep networks solve practical training problems that wide-shallow networks don't, which is why the field moved toward depth.

---

## Slide 8: Anatomy of a Neural Network

**Visual Description:**
Large, detailed feedforward network diagram with annotations:
- Input layer: nodes labeled x₁, x₂, x₃ (e.g., "age," "income," "education")
- Two hidden layers with 5 nodes each, edges shown with varying weights (thick = high weight, thin = low)
- Output layer: single node labeled ŷ (e.g., "vote probability")
- Annotated arrows pointing to: "weights W", "biases b", "activation function σ", "layer l"

**Bullet Points / Text:**
- Layers: input → hidden → output
- Each connection is a learned weight
- Bias shifts the activation threshold

**Instructor Notes:**
This is the diagram students should be able to draw from memory by the end of class. Emphasize that every edge in the diagram is a number that gets updated during training, and every node is a transformation. The social science application is in the labels: we will use this exact architecture this week to predict voting behavior from demographics.

---

## Slide 9: Activation Functions — What They Do

**Visual Description:**
Three side-by-side function plots, each with the mathematical formula and shape:
- Left: Sigmoid σ(z) = 1/(1+e⁻ᶻ) — S-curve from 0 to 1, shaded "saturates at extremes"
- Center: Tanh(z) — S-curve from -1 to 1, zero-centered, shaded "better for hidden layers"
- Right: ReLU(z) = max(0,z) — hockey stick, red annotation: "dead neurons if z<0 always"
Small additional plot: Leaky ReLU and ELU in lighter lines for comparison.

**Bullet Points / Text:**
- Activation = the non-linearity that enables depth
- ReLU dominates in practice (fast, sparse)
- Choice matters: vanishing gradients vs. dead neurons

**Instructor Notes:**
The activation function is why deep networks can learn complex functions. Without non-linearity, any number of linear layers collapse to a single linear transformation. ReLU's simplicity — just clamp negatives to zero — turns out to be enormously powerful and avoids the saturation problems of sigmoid and tanh in deep networks.

---

## Slide 10: The Forward Pass

**Visual Description:**
Step-by-step animated-style diagram showing a single data point moving through a 3-layer network:
```
Input x = [0.8, 0.2, 0.6]
    ↓
z¹ = W¹x + b¹  →  a¹ = ReLU(z¹)
    ↓
z² = W²a¹ + b²  →  a² = ReLU(z²)
    ↓
ŷ = σ(W³a² + b³)  =  0.73
```
Each transformation highlighted with a different color. The final output 0.73 is circled and labeled "Predicted vote probability."

**Bullet Points / Text:**
- Forward pass: input → prediction
- Matrix multiply → add bias → apply activation
- Every layer transforms the representation

**Instructor Notes:**
Code this live or in the notebook. The forward pass is pure arithmetic — matrix multiplication and element-wise non-linearities. What makes it powerful is that the representations at each layer become increasingly abstract: raw demographics → latent political orientation → vote probability. This is representation learning, and it is the core of everything that follows in the course.

---

## Slide 11: Loss Functions — How Wrong Are We?

**Visual Description:**
Two plots side by side:
- Left: Regression scenario. Scatter of actual vs. predicted values, with vertical lines showing residuals (errors). Formula: MSE = (1/n)Σ(yᵢ - ŷᵢ)²
- Right: Classification scenario. Bar chart of predicted probabilities [0.73, 0.27] vs. true label [1, 0]. Formula: BCE = -[y log(ŷ) + (1-y)log(1-ŷ)]

Below both: "Loss = how far the model is from the truth. Training = minimizing loss."

**Bullet Points / Text:**
- MSE for regression; Cross-entropy for classification
- Loss is the model's performance signal
- Lower loss = better predictions (on training data)

**Instructor Notes:**
The loss function is the definition of success. Making an explicit choice about your loss function is a deeply normative act — minimizing MSE penalizes outliers heavily; cross-entropy punishes confident wrong predictions more than uncertain ones. In social science contexts, who your model gets wrong matters as much as the average error.

---

## Slide 12: Gradient Descent — The Learning Algorithm

**Visual Description:**
A 3D loss landscape visualization — a bowl-shaped surface with contour lines at the bottom, and a red dot with an arrow showing the gradient descent path winding downward toward the minimum. Inset: 1D version showing the parabola with derivative ∂L/∂w shown as a tangent line pointing downhill.

Formula overlay:
```
w ← w - η · ∂L/∂w
```
where η is the learning rate (annotated "step size").

**Bullet Points / Text:**
- Walk downhill on the loss surface
- Learning rate η: too big → overshoot; too small → stall
- Local minima are usually fine in practice

**Instructor Notes:**
The key intuition is walking downhill blindfolded: you can only feel the slope at your feet, and you take a small step in whichever direction goes down most steeply. The learning rate controls how big that step is. In social science terms: gradient descent is how the model gradually adjusts its view of the world to match the data it has seen.

---

## Slide 13: Backpropagation — Credit Assignment

**Visual Description:**
A network diagram showing the backward pass with red gradient arrows flowing from output to input:
```
ŷ──►[Loss L]
         │ ∂L/∂ŷ
         ▼
    [Layer 3] ◄── ∂L/∂W³
         │ ∂L/∂a²
         ▼
    [Layer 2] ◄── ∂L/∂W²
         │ ∂L/∂a¹
         ▼
    [Layer 1] ◄── ∂L/∂W¹
```
Caption: "Chain rule, applied repeatedly. Each weight gets its share of blame."

**Bullet Points / Text:**
- Chain rule: ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂w
- Backprop flows error backward through the network
- Rumelhart, Hinton & Williams (1986) — the breakthrough

**Instructor Notes:**
Backpropagation is the reason deep learning became viable. It is just the chain rule from calculus applied repeatedly, but its computational efficiency — storing activations from the forward pass and reusing them — made training networks with millions of parameters feasible. The social science analogy: attributing outcomes to causes is the core of causal inference, and backprop is how the network does it.

---

## Slide 14: Optimization — SGD, Momentum, Adam

**Visual Description:**
Three side-by-side trajectory plots on the same loss surface contour map, each showing a different optimizer's path to the minimum:
- SGD: jagged, noisy path (many small random steps)
- SGD + Momentum: smoother, faster, slight overshoot
- Adam: direct, smooth, fast convergence

Below: a table
```
Optimizer  | Speed  | Memory | Notes
-----------+--------+--------+------------------
SGD        | slow   | low    | simple, robust
Momentum   | medium | low    | smooths oscillation
Adam       | fast   | higher | adapts per-weight
```

**Bullet Points / Text:**
- SGD: pure gradient steps (noisy but generalizes)
- Momentum: remember previous direction
- Adam: per-parameter adaptive learning rates

**Instructor Notes:**
Adam is the default for most practitioners because it works well out of the box, but recent research suggests SGD with careful tuning often generalizes better. For social science applications where interpretability matters, simpler optimizers can produce more stable and understandable results. In the notebook today, students will compare these head-to-head.

---

## Slide 15: Weight Initialization — Why It Matters

**Visual Description:**
Two training loss curves on the same plot:
- Red curve (poor initialization — all zeros or too-large values): flat for many epochs, then diverges or plateaus
- Green curve (Xavier/He initialization): immediate smooth descent

Inset diagram showing vanishing gradients (signals getting smaller and smaller from output to input) vs. exploding gradients (signals growing exponentially).

Formula box:
```
Xavier:  W ~ N(0, 2/(nᵢₙ + nₒᵤₜ))
He:      W ~ N(0, 2/nᵢₙ)
```

**Bullet Points / Text:**
- All-zeros: every neuron learns identically (symmetry problem)
- Too large: exploding gradients; too small: vanishing
- Xavier for sigmoid/tanh; He for ReLU

**Instructor Notes:**
Weight initialization is one of those details that separates practitioners from beginners. The symmetry problem — why initializing all weights to zero means no neuron ever differentiates from any other — is a good test of conceptual understanding. Xavier and He initialization preserve variance across layers, keeping the signal from shrinking or exploding as it passes through depth.

---

## Slide 16: Regularization — Fighting Overfitting

**Visual Description:**
Three scatter plots showing model fits:
- Left: Underfitting — straight line through curved data (high bias, "the model doesn't care enough about your data")
- Center: Good fit — smooth curve through data ("just right")
- Right: Overfitting — wiggly curve hitting every point ("memorized the training data, useless on new data")

Below: icons for L1 (sparse weights, some exactly zero), L2 (small weights all nonzero), Dropout (random neurons "crossed out" during training).

**Bullet Points / Text:**
- Overfitting: memorizes noise, fails on new data
- L1/L2: penalize large weights in the loss function
- Dropout: random neuron silencing during training

**Instructor Notes:**
The bias-variance tradeoff is one of the most important conceptual frameworks in all of machine learning, and it applies directly to social science: a model that perfectly fits your survey data from 2016 will not necessarily generalize to 2020. Regularization is the formal mechanism for building models that capture patterns rather than idiosyncrasies.

---

## Slide 17: The Bias-Variance Tradeoff

**Visual Description:**
Classic bull's-eye diagram with four quadrants:
- High bias, low variance: all shots clustered far from center (consistent but wrong)
- Low bias, high variance: shots scattered around center (right on average but unreliable)
- High bias, high variance: shots scattered far from center (worst of both)
- Low bias, low variance: shots clustered at center (the goal)

Below: a U-shaped double curve plot: x-axis = model complexity; y-axis = error. Blue curve = training error (decreasing); red curve = test error (U-shaped). "Sweet spot" marked at minimum of test error.

**Bullet Points / Text:**
- Bias: systematic error from wrong assumptions
- Variance: sensitivity to fluctuations in training data
- Regularization trades variance for bias (wisely)

**Instructor Notes:**
This tradeoff is more than a technical concept — it maps directly onto the social science tension between parsimony and fit. A simple regression model (high bias) gives interpretable coefficients but misses complex patterns. A deep neural network (potentially high variance) captures complexity but may not generalize. The goal in both traditions is the same: the simplest model that explains what needs to be explained.

---

## Slide 18: Dropout — Forced Robustness

**Visual Description:**
Side-by-side diagrams of the same neural network:
- Left: full network, all connections active (labeled "Training time without dropout")
- Right: same network with ~40% of neurons grayed out and connections dashed (labeled "Training with dropout p=0.4")

Below: illustration showing that at test time, all neurons are active but weights are scaled by (1-p), labeled "Test time: all neurons, scaled weights."

Code snippet:
```python
nn.Dropout(p=0.5)  # kills 50% of neurons per forward pass
```

**Bullet Points / Text:**
- Randomly zero out neurons during training
- Forces network to learn redundant representations
- Equivalent to ensemble of thinned networks

**Instructor Notes:**
Srivastava et al. (2014) originally described dropout as training an exponential number of different architectures simultaneously — each minibatch, you train a slightly different network. The result is a model that cannot rely on any single feature pathway, which improves generalization. In social science terms: don't let your model become too dependent on any one predictor.

---

## Slide 19: Application — Predicting Voting from Demographics

**Visual Description:**
A map of the United States with counties shaded in a blue-red gradient showing predicted Republican vote share. Overlaid: a feedforward network diagram with labeled inputs:
```
Input features:
  age         ──┐
  income      ──┤
  education   ──┼──► [NN] ──► P(vote Republican)
  religion    ──┤
  urban/rural ──┘
```
Below: a small table showing actual vs. predicted probabilities for 5 counties.

**Bullet Points / Text:**
- X: demographics; y: party vote share
- Neural network vs. logistic regression: what do we gain?
- What do errors tell us about the limits of prediction?

**Instructor Notes:**
This is the hands-on application in Module 1 of today's notebook. Walk students through the framing: we are not trying to build a perfect predictor, we are trying to understand which features matter, how they interact, and where the model breaks down. Counties where the neural network is most wrong are often the most sociologically interesting.

---

## Slide 20: What Deep Learning Finds That Regression Misses

**Visual Description:**
Two decision boundary plots for the same demographic data:
- Left: Logistic regression boundary — a single straight line separating predicted Republican from Democrat counties
- Right: Neural network boundary — a complex non-linear boundary with "islands" of exception

Highlighted in circles: counties where they disagree. Caption: "The model's errors are a research agenda."

**Bullet Points / Text:**
- Logistic regression: additive, interpretable, limited
- Deep networks: interactions, thresholds, non-linearities
- The gap between them = complexity in human behavior

**Instructor Notes:**
This slide is the conceptual payoff of the technical section. Social science has historically relied on linear models because they are interpretable — each coefficient tells a story. Deep networks capture more of the truth but tell that story less clearly. The field is now developing tools (SHAP, LIME, attention weights) to recover interpretability — we will see these in Week 2 and 6.

---

## Slide 21: Neural Life Trajectories — Savcisens et al. 2024

**Visual Description:**
A timeline visualization showing an individual's life as a sequence of tokens:
```
[born_DK] [edu_primary] [moved_Copenhagen] [employed_finance]
[income_high] [married] [child] [diagnosis_T2D] [retired] [deceased]
```
Below: the cover image or figure from the Nature Computational Science paper showing AUC curves for life event prediction. Caption: "6.4 million Danish lives. One transformer. Life as language."

**Bullet Points / Text:**
- Life as a sequence = transformer input
- Predicts: health, income, mortality, social mobility
- What are the ethics of predicting a life?

**Instructor Notes:**
Savcisens et al. trained a BERT-like model on complete administrative records — every formal interaction a citizen has with Danish institutions from birth. The model achieves striking predictive accuracy. This is a profound social science achievement and a profound ethical question simultaneously: the same architecture we use to translate text is here being used to predict whether someone will die before 35.

---

## Slide 22: Generative Agents — Park et al. 2023

**Visual Description:**
A screenshot-style image of the "Smallville" simulation environment: a pixel-art village with labeled AI agents going about their daily activities. Overlaid text bubbles showing two agents having a conversation. One agent is labeled with its memory context:
```
System: You are Isabella Rodriguez. You run the coffee shop.
Memory: [Last Thursday: talked to Tom about the election]
        [Today: planning a Valentine's Day party]
```

**Bullet Points / Text:**
- 25 LLM agents with persistent memory and daily schedules
- Emergent: gossip, romance, coordination — nobody programmed it
- Passing "social Turing test" behaviors at the village level

**Instructor Notes:**
Park et al. is one of the landmark papers of 2023 precisely because the social behavior in Smallville was not hand-coded — it emerged from LLMs with memory and goals. This is the direction Week 3 goes: from single-agent prediction to multi-agent social simulation. The question that should haunt us is whether emergent social behavior in simulation tells us anything real about emergent social behavior in society.

---

## Slide 23: Why Social Scientists Need to Understand These Systems

**Visual Description:**
A Venn diagram with two large overlapping circles:
- Left circle: "Social Science" — icons: survey, interview, regression, theory, ethnography
- Right circle: "AI/ML" — icons: neural networks, embeddings, LLMs, agents
- Overlap: "AI Agents for Social Science" — icons: simulation, causal inference, bias detection, cultural analysis, prediction

Below the diagram: two quotes in italic
- "AI is a social technology" — Farrell, Gopnik, Shalizi, Evans 2025
- "The same models that study society are also changing it."

**Bullet Points / Text:**
- AI encodes social relations → social scientists must decode them
- New methods for old questions — and new questions entirely
- The tools of social science are now AI agents

**Instructor Notes:**
This slide crystallizes the course's intellectual argument. Farrell, Gopnik, Shalizi, and Evans argue that large AI models should be understood as cultural and social technologies — not neutral instruments but artifacts that absorb and reproduce the social structures they were trained on. Social scientists are therefore not optional participants in AI development; they are essential critics and designers.

---

## Slide 24: PyTorch — Our Building Material

**Visual Description:**
Code snippet on a dark terminal background:
```python
import torch
import torch.nn as nn

class VotingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)
```
To the right: PyTorch logo and a caption: "Define it. Train it. Interrogate it."

**Bullet Points / Text:**
- PyTorch: define-by-run (dynamic computation graphs)
- `nn.Module`: the base class for all networks
- `forward()`: defines the computation

**Instructor Notes:**
Open the notebook now if you haven't already. The `nn.Module` class is the right abstraction level for this course — it lets us think in layers rather than individual matrix operations, while still giving us access to gradients and weight inspection. Students who understand this basic PyTorch pattern will be equipped for every neural network we build for the rest of the course.

---

## Slide 25: The Training Loop

**Visual Description:**
Annotated code block on dark background:
```python
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:   # ① sample
        optimizer.zero_grad()              # ② clear gradients
        y_pred = model(X_batch)            # ③ forward pass
        loss = criterion(y_pred, y_batch)  # ④ compute loss
        loss.backward()                    # ⑤ backpropagation
        optimizer.step()                   # ⑥ update weights
```
Each numbered step has a color-coded annotation arrow pointing to the corresponding concept from earlier slides (forward pass, loss, backprop, gradient descent).

**Bullet Points / Text:**
- Six lines. This is how every neural network learns.
- The loop repeats for every batch, every epoch
- Monitoring loss per epoch tells the health story

**Instructor Notes:**
Memorize these six steps. Walk through them slowly and connect each one explicitly to the conceptual slides from earlier: step 3 is Slide 10 (forward pass), step 4 is Slide 11 (loss), step 5 is Slide 13 (backprop), step 6 is Slide 12 (gradient descent). The training loop is the heartbeat of the entire field.

---

## Slide 26: Reading Loss Curves

**Visual Description:**
A dual-axis plot with epoch on x-axis and loss on y-axis:
- Two curves: training loss (blue, steadily declining) and validation loss (orange)
- Three annotated regions:
  - Early (both declining): "Learning phase — good"
  - Middle (training continues down, validation flattens): "Approaching capacity"
  - Late (training low, validation rising): "Overfitting zone — stop here"
- A vertical dashed line at the optimal stopping point labeled "Early stopping"

**Bullet Points / Text:**
- Training loss ≠ validation loss
- Gap between them = degree of overfitting
- Early stopping: stop when validation loss stops improving

**Instructor Notes:**
Reading loss curves is a core practical skill. Students should always plot both training and validation loss — the gap between them is one of the most informative diagnostics available. In social science applications, overfitting often looks like a model that explains your survey perfectly but fails on a replication sample from a different year.

---

## Slide 27: What Neurons Learn — Feature Visualization

**Visual Description:**
A grid of 16 small square patches showing what each neuron in a convolutional network's first layer has learned to detect: edges at various angles, color blobs, simple textures. Caption: "Even simple images reveal structure. What do social data neurons learn?"

Below: two activation maps for text, showing which input words most activate a neuron predicting "positive sentiment" — words like "excellent," "loved," "recommend" are highlighted in yellow.

**Bullet Points / Text:**
- Neurons in early layers learn simple patterns
- Deep layers learn abstract combinations
- We can visualize what the model "sees"

**Instructor Notes:**
Feature visualization is the bridge between Week 1 (networks learn) and Week 6 (we look inside networks). Even at this introductory stage, students should internalize that neural networks are not black boxes by definition — they are learnable, inspectable representations. The social science question is always: what concept did this neuron encode, and is that concept socially fair?

---

## Slide 28: Limitations — What Neural Networks Cannot Do

**Visual Description:**
A two-column comparison table, styled as a report card:
```
NEURAL NETWORKS: REPORT CARD
─────────────────────────────────────────────────
Interpolation within training distribution:    A
Generalization to similar data:                B
Causal reasoning:                              D
Uncertainty quantification (default):          D
Explanation of predictions:                    C
Handling distribution shift:                   C
Sample efficiency:                             D
─────────────────────────────────────────────────
```
Footnote: "These are not bugs waiting to be fixed — they are properties of the architecture."

**Bullet Points / Text:**
- Correlation, not causation — by default
- Distribution shift: the 2016 model fails in 2020
- Explanation is an add-on, not a feature

**Instructor Notes:**
This slide is crucial for intellectual honesty. Neural networks are extraordinarily powerful pattern matchers, but they do not understand causation, they do not generalize gracefully outside their training distribution, and they do not naturally produce uncertainty estimates. The remainder of this course is partly about mitigating each of these limitations — Week 4 will directly tackle causation, and Week 9 tackles safety and reliability.

---

## Slide 29: What Makes This Social Science?

**Visual Description:**
A single large quote on a plain background, typeset in elegant serif font:
> "Deep learning doesn't just provide better predictions — it changes the kind of questions we can ask."
> — James Evans, Preface: How to Think with Deep Learning

Below the quote: three icons in a row with labels:
- Magnifying glass: "New questions" (individual trajectories, emergent culture)
- Scale: "New scale" (millions of documents, billions of observations)
- Mirror: "New reflexivity" (the model is part of the social world it studies)

**Bullet Points / Text:**
- New questions: from means to individuals, from structure to dynamics
- New scale: population-level patterns in real time
- New risk: the model shapes what it studies

**Instructor Notes:**
Evans's preface should be the intellectual touchstone for the course. Read students the quote and pause. The shift from asking "what is the average effect?" to "what is this specific person's trajectory?" represents a genuine epistemological transformation in social science methodology — not just better tools, but different ways of knowing.

---

## Slide 30: Everything Builds on This

**Visual Description:**
A dramatic closing visual: the same feedforward network diagram from Slide 8 is shown on the left, but now it is connected by a glowing arrow to a series of progressively more complex architectures on the right:
- An RNN (curved feedback arrow)
- A Transformer (attention matrix visualization)
- A multi-agent graph (multiple interconnected agents)
- A reinforcement learning loop (agent-environment-reward triangle)

All connected back to the original perceptron with the caption: "Same idea. More depth. More power. More responsibility."

**Bullet Points / Text:**
- Every complex model is built on what you learned today
- Week 2: these weights become language
- Week 3: these networks become societies
- The foundations matter — come back to them

**Instructor Notes:**
Close with this forward-looking visual to motivate students who may have found today's material demanding. The transformer that powers GPT-4, the multi-agent system that simulates an election, the reinforcement learning agent that optimizes policy — all of them are variations on the same core ideas: layers, weights, activations, and gradient descent. What changes is scale, architecture, and application. What stays constant is the mathematics from today.

---

*End of Week 1 Slide Deck — 30 Slides*
*AI Agents for Social Science and Society 2026*
*Instructor: James A. Evans | January 9, 2026*
