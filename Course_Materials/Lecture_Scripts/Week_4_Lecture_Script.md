# Week 4 Lecture Script: Experimental Designs with AI Agents and Human Interactions
**Course:** AI Agents for Social Science and Society 2026
**Instructor:** James A. Evans
**Date:** January 30, 2026
**Room:** 1155 E. 59th Street, Room 295
**Duration:** 3 hours (1:30–4:20 PM)
**Notebook:** `week_4_2026.ipynb` (273 cells)

---

## Instructor Preparation Notes

Before class:
- Load the notebook in Colab and confirm TensorFlow 2.8, cGNF, and DoubleML all install correctly
- Pre-run cells 1–25 (S-Learner) so outputs are cached — these take time on CPU
- Have the IHDP dataset URLs handy: `http://www.fredjo.com/files/ihdp_npci_1-100.train.npz`
- Print or bookmark the Koch et al. "Deep Learning for Causal Inference" paper
- Bring up the DAG diagram for the Reddit gender/upvotes example

---

## Section 1: Opening and Week 3 Bridge (0:00–0:15)

**[Stand at front, notebook projected but closed]**

Say: "Last week we asked whether LLMs could simulate human subjects — whether a digital double of a 45-year-old Republican from Ohio gives us the same responses as an actual person. Some of you found that GPT-4 nailed vote predictions, and some of you found systematic divergence. That's exactly the tension we're going to resolve today."

Write on board:
```
Week 3: SIMULATION — Do AI agents behave like humans?
Week 4: EXPERIMENTATION — Can AI agents help us CAUSE knowledge about humans?
```

Say: "There is a crucial difference. Simulation asks whether an LLM output resembles human output. Experimentation asks whether we can recover a *causal* effect from data that includes AI agents. The methods are completely different, and the epistemological stakes are higher."

Ask students: "What is the fundamental problem of causal inference?"

**[Pause for responses. Expected: we can only observe one potential outcome per unit.]**

Say: "Right. For any unit — a Reddit post, a survey respondent, a voter — we observe what actually happened, not what would have happened under the counterfactual. Every causal inference technique, from randomized experiments to the deep learning methods we cover today, is a response to that fundamental problem."

Write on board:
```
ATE = E[Y(1) - Y(0)]   — we observe either Y(1) OR Y(0), never both
CATE = E[Y(1) - Y(0) | X = x]   — same problem, now conditioned on features
```

**[Timing: 10 minutes]**

---

## Section 2: Causal Inference Refresher — DAGs, Potential Outcomes (0:15–0:45)

Say: "Before we get to the deep learning methods, everyone needs to be on the same page about identification. Let's use the core running example from the notebook."

**[Open notebook. Go to Cell 4.]**

Read aloud from Cell 4: "You have data from Reddit and want to determine how the author's stated gender — that's your treatment T — influences the number of upvotes a post receives — that's your outcome Y. There's a complexity: gender may influence the post's content X — such as tone, style, or topics — which in turn affects upvotes."

Draw on board:
```
        Gender (T)
       /          \
      ↓            ↓
   Content (X) → Upvotes (Y)
      ↑
   (X also confounds T→Y)
```

Say: "This is a classic confounder setup. If you just regress Y on T, you get a biased estimate because content is both caused by gender and causes upvotes. To recover the *direct* causal effect of gender on upvotes — holding content constant — you need to control for X."

Ask students: "What assumptions do we need to identify ATE from observational data?"

**[Expected responses: unconfoundedness/selection on observables, positivity, SUTVA.]**

Say: "Exactly. The selection-on-observables assumption says: conditional on X, treatment assignment is independent of potential outcomes. Positivity says: every unit has some probability of receiving each treatment. Today's methods all rely on these assumptions — deep learning does not buy you free identification. What it buys you is better estimation when X is high-dimensional or complex."

Write on board:
```
Identification assumptions (no stats can fix these):
1. Unconfoundedness: Y(t) ⊥ T | X
2. Positivity: P(T=t | X=x) > 0 for all x
3. SUTVA: no interference between units

Estimation (this is where deep learning helps):
- E[Y | X, T] is complex → neural networks have lower bias
- X includes images, text, networks → needs representation learning
```

**[Timing: 15 minutes]**

---

## Section 3: Heterogeneous Treatment Effects — S-Learner, T-Learner, TARNet (0:45–1:15)

Say: "Now let's build up the three main architectures. The paper we're working from is Koch et al. 'Deep Learning for Causal Inference' — it's a masterclass in how to adapt neural networks for causal estimation."

**[Navigate to Cell 2 header, then Cell 8.]**

Say: "The dataset is IHDP — the Infant Health and Development Program dataset. Hill (2011) took real experimental covariates from 747 mother-infant pairs — 25 covariates including birth weight, maternal education, presence of twins — and simulated outcomes under a known causal effect. This gives us ground truth CATE values to evaluate against."

### S-Learner

**[Navigate to Cell 12.]**

Say: "Attempt number one: the S-Learner. S stands for Single. The simplest possible approach: treat treatment T as just another input feature alongside X."

Write on board:
```python
# S-Learner: Treatment as a feature
# Input: [X, T]  →  Output: Y_hat
# CATE_i = Y_hat([X_i, 1]) - Y_hat([X_i, 0])
```

**[Show Cell 15. Point to architecture.]**

Say: "Here's the architecture — three hidden layers of 200 neurons with ELU activations, L2 regularization. After training, we make predictions twice: once with T set to all ones, once with T set to all zeros, and take the difference."

**[Show Cell 22–23.]**

Say: "Notice what's happening here. We create fake tensors of all zeros and all ones and concatenate them to X. This is how we counterfactually 'assign' treatment to every unit — including those who were actually in the control group."

Ask students: "What's the fundamental limitation of the S-Learner?"

**[Expected: The model can learn to ignore treatment, especially if treatment is rare or underpowered.]**

Say: "Exactly. If treated units are only 139 out of 747 — as in IHDP — the model may learn the main effect of X and treat T as noise. Cell 26 spoils this: the S-Learner basically learns nothing. The loss flatlines."

### T-Learner

**[Navigate to Cell 27.]**

Say: "Attempt two: the T-Learner. Two independent networks — one trained only on control units, one only on treated units. Each head sees only the factual outcomes for its treatment condition."

**[Show Cell 29. Point to the TensorFlow Functional API structure.]**

Say: "Here's the key insight in the loss function at Cell 34. We use t_true as a switch:"

Write on board:
```python
loss0 = tf.reduce_sum((1. - t_true) * tf.square(y0_pred - y_true))
loss1 = tf.reduce_sum(t_true * tf.square(y1_pred - y_true))
loss = loss0 + loss1
```

Say: "The treatment indicator t_true routes gradients only through the appropriate head. Head 0 only updates from control-group observations; Head 1 only from treated observations. This is cleaner than the S-Learner because treatment heterogeneity is explicitly modeled."

**[Show output discussion at Cell 39.]**

Say: "The T-Learner actually converges. ATE around 3.54, PEHE around 0.95. Better — but still biased toward lower treatment effects."

### TARNet

**[Navigate to Cell 40–43.]**

Say: "The key idea in TARNet — Treatment Agnostic Regression Network — is representation learning. The shared encoder layers force both outcome models to learn from the same feature representation. Why does this help?"

Write on board:
```
TARNet Architecture:
  [X] → Shared Representation Φ(X) → [Head 0: Ŷ(0)]
                                    → [Head 1: Ŷ(1)]

Key: Φ(X) must be useful for BOTH potential outcomes
→ This creates implicit covariate balance between treatment arms
```

Ask students: "In terms of econometrics, what is this shared representation analogous to?"

**[Expected: propensity score balancing, or matching on covariates.]**

Say: "Exactly. The representation layer learns to map X into a space where treated and control units are more similar, making the outcome comparison more credible. It's not perfect — there are DragonNet extensions that explicitly add a propensity score head — but TARNet is a major step forward."

**[Show Cell 55: TARNet results. Note lower bias in CATE distribution.]**

Say: "Visually, the CATE distribution from TARNet is much closer to the true distribution. This is the key takeaway: shared representation learning = implicit covariate balance = lower bias in HTE estimation."

**[Code demonstration — do not run but walk through Cell 45 (TARNet definition)]**

Say: "Notice the structure: three shared layers that compress X into Φ(X), then two separate heads each with two more layers. The input goes through `Phi` and then both `Y0` and `Y1` heads receive Φ(X) as input. This is implemented using TensorFlow's Functional API, not Sequential, because we have branching architecture."

**[Timing: 30 minutes]**

---

## Section 4: Causal Mediation Analysis with cGNF (1:15–1:35)

Say: "HTE answers 'how much does T affect Y?' But social scientists often care about *how* — through what mechanisms does a treatment work? That's mediation analysis, and it's significantly harder."

**[Navigate to Cell 66.]**

Write on board:
```
Direct effect:    T → Y  (not through mediator)
Indirect effect:  T → M → Y  (through mediator)
Total effect = Direct + Indirect
```

Say: "The classic example is a job training program. The total effect on wages could work directly — employers see the credential — or indirectly through skills. If skills are the mediator, the NDE is the credential signaling effect and the NIE is the skills acquisition effect."

Ask students: "Why can't we just add M as a control variable in a regression?"

**[Expected: M is a post-treatment variable, controlling for it opens collider bias, etc.]**

Say: "Exactly. M is downstream of T. Controlling for M in a regression of Y on T gives you the 'wrong' causal quantity — it blocks the pathway you're trying to estimate. The literature on mediation is full of this mistake."

**[Navigate to Cell 68 — the DAG diagram.]**

Say: "The setup: pre-treatment confounder C, treatment A, mediator M, outcome Y. The natural direct effect — NDE — is E[Y(1, M(0)) - Y(0, M(0))]. That subscript (1, M(0)) means: set treatment to 1 but let the mediator take the value it would have taken under treatment zero. This is a cross-world counterfactual — it refers to two different potential worlds simultaneously."

**[Navigate to Cell 70 — exposure-induced confounders.]**

Say: "Things get worse when we have exposure-induced confounders. Variable L is affected by treatment A but also confounds the M-Y relationship. Traditional mediation methods — like the Baron-Kenny approach — completely fail here."

Write on board:
```
Problem: L is both caused by A and confounds M→Y
Baron-Kenny: control for L → wrong (opens backdoor through L←A)
Baron-Kenny: don't control for L → wrong (omits confounder of M→Y)
Solution: cGNF — model the full joint distribution as a normalizing flow
```

**[Navigate to Cells 72–74 — normalizing flows explanation.]**

Say: "cGNF — causal graphical normalizing flows — solves this by modeling the entire joint distribution P(C, A, L, M, Y) as a series of invertible transformations from a standard normal. The three-step workflow is clean: first process() your data and adjacency matrix; second train() the flow; third sim() to estimate any causal estimand you want."

**[Show Cell 77 — DAG specification.]**

Say: "Notice how the causal structure is encoded as an adjacency matrix. The model doesn't 'see' the causal graph during training — it learns the joint distribution. But the graph constraints enforce which variables can cause which, ensuring we get valid causal estimates."

**[Do not run — reference output description.]**

Say: "The full workflow takes 10–30 minutes on CPU. The key diagnostic is the latent space plot: if training succeeded, passing your data through the trained flow should produce standard normal distributions. If the histograms don't match N(0,1), the model hasn't learned the data-generating process correctly."

**[Timing: 20 minutes]**

---

## Section 5: Double/Debiased Machine Learning (1:35–1:55)

Say: "Module 3 introduces Double/Debiased ML — one of the most important methodological contributions to causal inference in the last decade. It was introduced by Chernozhukov et al. in 2018 and it solves a subtle but devastating problem."

**[Navigate to Cell 107.]**

Write on board:
```
DGP:
  y_i = θ₀·d_i + g₀(x_i) + ζ_i    [outcome equation]
  d_i = m₀(x_i) + v_i              [treatment equation]

We want θ₀ (the causal effect of d on y)
But g₀ and m₀ are complex, unknown functions → use ML
```

Say: "The naive approach: fit a ML model for g₀ on your full dataset, then estimate θ. What goes wrong?"

Ask students: "What happens when you use regularized ML to estimate g₀?"

**[Expected: regularization introduces bias that bleeds into the estimate of θ₀.]**

Say: "Exactly. Cell 116 shows this mathematically. The bias term is proportional to the product of estimation errors in g₀ and m₀. If both errors converge slowly — as they do with regularized models — their product may not vanish fast enough, and your standard errors for θ₀ are wrong."

Write on board:
```
Bias from naive ML:
√n(θ̂₀ - θ₀) = term1 + bias_term
bias_term ∝ (error in ĝ₀) × (error in m̂₀)

DML solution: Neyman Orthogonality
- Partial out X from both Y and D
- Cross-fit on held-out samples
- Residual-on-residual regression for θ₀
```

**[Navigate to Cell 117 — orthogonalization.]**

Say: "The fix has two parts. First, orthogonalization: instead of regressing Y on D, regress the residual of Y (after removing g₀(X)) on the residual of D (after removing m₀(X)). Second, cross-fitting: estimate nuisance functions g₀ and m₀ on one fold, then apply them on another fold."

**[Navigate to Cell 124–130 — the 401k example.]**

Say: "The real data application is elegant. We want to know: does 401(k) eligibility — that's e401, our treatment — causally increase net financial assets — net_tfa? The naive estimate shows a $19,559 gap. But eligible workers are systematically different — they're more likely to be higher earners who choose to work for companies offering retirement benefits."

**[Show Cell 135: unconditional APE.]**

Say: "After DML with Lasso as the nuisance learner, controlling for age, income, education, family size, and marital status — in flexible polynomial form — we get a considerably different estimate. The DML estimate accounts for selection into eligibility."

**[Show Cells 144–149: comparison of estimators.]**

Say: "Compare Lasso, Random Forest, Decision Trees, and Boosted Trees as nuisance estimators. They give somewhat different point estimates, but the key insight is that DML makes all of them valid inferential tools — without it, none of them produce valid standard errors."

**[Timing: 20 minutes]**

---

## Section 6: Prediction-Powered Inference — AI + Humans (1:55–2:20)

Say: "The last module brings us to the core question for AI agents in social science: what do we do when we have some expensive human labels and many cheap AI-generated labels? Prediction-Powered Inference answers this question rigorously."

**[Navigate to Cell 182.]**

Say: "The Mixed Subjects Design from Broska, Howes, and van Loon (2025) is the umbrella framework. PPI is one implementation. The key insight: AI-generated labels are not just noisy human labels — they have systematic biases. PPI incorporates a bias correction."

Write on board:
```
PPI setup:
  - Labeled dataset: {(X_i, Y_i)}^n    [human labels, expensive]
  - Unlabeled dataset: {X̃_i}^N         [AI predictions Ŷ_i, cheap]

PPI estimator:
  θ_PPI = θ_classical + correction_term
  correction_term = bias in AI predictions, estimated from labeled data
```

**[Navigate to Cells 189–192 — effective sample size.]**

Say: "The critical parameter is ρ — the PPI correlation. This measures how well AI predictions substitute for human labels. Higher ρ means more informative AI predictions, which means PPI effectively adds more sample size."

**[Show Cell 189 code.]**

```python
def n0(rho: float, n: float, k: float) -> float:
    """Effective sample size. k = N/n (ratio of AI to human labels)."""
    return (n * (k + 1)) / (k * (1 - rho**2) + 1)
```

Say: "With rho=0.9 and k=10 — ten AI labels per human label — your effective sample size is about 5x your human label count. With rho=0.7, it's about 2x. But notice: if rho is close to zero, you gain almost nothing from AI labels, no matter how many you have."

Ask students: "What does ρ depend on in practice?"

**[Expected: quality of the AI model, similarity between AI training distribution and target population, task difficulty.]**

Say: "Exactly. For a task like coding in Python where GPT-4 is near-human, ρ will be high. For a task like predicting subjective political views of rural Guatemalan voters, ρ will be low or even negative if the model has systematic biases."

**[Navigate to Cells 212–218 — Moral Machine example.]**

Say: "The substantive application is brilliant. The Moral Machine experiment asked people worldwide which of two groups a self-driving car should sacrifice in a trolley-problem scenario. They varied attributes: age, gender, social status, number of people, criminal history. Getting human judgments on all possible combinations is prohibitively expensive. AI judgments are cheap but biased. PPI lets you use AI judgments to fill in the gaps while using human judgments to correct the bias."

**[Timing: 25 minutes]**

---

## Section 7: Student Code Presentations (2:20–2:40)

**[Call on 3–4 students to briefly share what module they completed and one result.]**

Instructor prompts if needed:
- "What was your treatment, outcome, and covariate set?"
- "Did the S-Learner, T-Learner, or TARNet give the most different result — and why do you think that is?"
- "If you did the DML module, how did Lasso compare to Random Forest as a nuisance estimator?"

---

## Section 8: Discussion — Is an LLM a Valid Experimental Subject? (2:40–3:05)

Say: "Let's end with the hardest question. Last week we simulated humans with LLMs. Today we've seen designs where AI agents participate alongside human subjects. The Broska et al. 'Mixed Subjects Design' paper takes this seriously — but on what grounds is it valid?"

Write on board:
```
When is an LLM a valid subject?
1. When we're studying AI behavior itself
2. When LLM responses correlate with human responses (ρ > 0)
3. When we're doing preliminary hypothesis generation
4. When costs of full human data collection are prohibitive

When is it NOT valid?
1. When the population of interest are real humans, not AI
2. When LLM training data is systematically biased (gender, race, culture)
3. When the behavior under study wasn't in training data
4. When individual-level variation matters (not just averages)
```

Ask students: "The Costello et al. 2024 paper shows AI dialogue durably reduces conspiracy beliefs. The AI was acting as a persuader, not a subject. Is that different from the mixed-subjects case?"

**[Allow 5–7 minutes of discussion. Push students toward the distinction between AI as instrument vs. AI as subject.]**

Ask students: "Potter, Lai et al. found that LLMs have detectable political biases that influence human readers. Does this make LLMs less useful as experimental subjects, or more interesting as experimental objects?"

**[Expected tension: biased AI is a confound if you want to study humans, but is itself a research object if you want to study AI influence on political opinion.]**

Say: "The Lai et al. 'Biased AI Improves Human Decision Making' paper shows this tension is empirically rich. Strategically biased AI can sometimes improve human decisions by compensating for human biases in the opposite direction. That's a fascinating result — but it requires knowing something about the AI's bias first. Which means interpretability — our Week 6 topic — is not just philosophically interesting. It's a prerequisite for sound experimental design."

**[Timing: 25 minutes]**

---

## Section 9: Closing Summary and Homework Briefing (3:05–3:20)

Write on board:
```
Week 4 Core Takeaways:
1. Deep learning → lower bias estimation of complex response surfaces
2. HTE methods: S-Learner < T-Learner < TARNet (representation balance)
3. cGNF handles exposure-induced confounders classically impossible cases
4. DML: orthogonalize + cross-fit → valid inference with any ML nuisance learner
5. PPI: use AI predictions to augment human labels with rigorous bias correction
6. AI agents are valid subjects under specific, checkable conditions — not by default
```

Say: "The homework asks you to complete three of four modules. My recommendation: start with Module 1 (HTE) because it gives you the clearest intuition, then do either Module 3 (DML) or Module 4 (PPI) depending on what's most relevant to your project. Module 2 (mediation) is the hardest to run — the cGNF training takes 20–30 minutes — so budget your time."

Say: "For the homework exercises, you need to define your own T, Y, and X for a research question relevant to your project. The code is there — your contribution is the research design and the interpretation. Cell 58 asks you explicitly: What is your causal claim? What are your covariates? What identification assumption are you invoking? These are social science questions, not Python questions."

Say: "Memos are due before class. The memo should articulate your causal question, your proposed identification strategy, and a figure showing either your data or your estimated effect. Remember: memos are graded on research design innovation, not just completion."

**[Final question:]**

Ask students: "Before you go — one sentence: what is the key difference between Neyman orthogonality in DML and standard residual regression?"

**[Expected: DML cross-fits nuisance functions to avoid overfitting bias contaminating the causal estimate; standard OLS residual regression does not account for this.]**

---

## Appendix: Key Code Snippets for Board Reference

**S-Learner CATE estimation:**
```python
zeros = np.expand_dims(np.zeros(data['x'].shape[0]), 1)
ones = np.expand_dims(np.ones(data['x'].shape[0]), 1)
x_untreated = np.concatenate([data['x'], zeros], 1)
x_treated = np.concatenate([data['x'], ones], 1)
y0_pred = s_learner.predict(x_untreated)
y1_pred = s_learner.predict(x_treated)
cate_pred = (y1_pred - y0_pred).squeeze()
```

**TARNet architecture note:**
```python
# Shared representation
phi = Dense(200, activation='elu')(inputs)
phi = Dense(200, activation='elu')(phi)
phi = Dense(200, activation='elu')(phi)
# Separate heads
y0_pred = Dense(100, activation='elu')(phi)
y0_pred = Dense(1)(y0_pred)
y1_pred = Dense(100, activation='elu')(phi)
y1_pred = Dense(1)(y1_pred)
```

**DML core idea:**
```python
# Estimate nuisances on fold 1
ml_g.fit(X_train, Y_train)
ml_m.fit(X_train, D_train)
# Estimate theta on fold 2 (cross-fitting)
residual_Y = Y_test - ml_g.predict(X_test)
residual_D = D_test - ml_m.predict(X_test)
theta = (residual_D @ residual_Y) / (residual_D @ residual_D)
```

---

## Timing Summary

| Section | Duration | Cumulative |
|---|---|---|
| Opening and Week 3 Bridge | 15 min | 0:15 |
| Causal Inference Refresher | 15 min | 0:30 |
| HTE: S/T/TARNet | 30 min | 1:00 |
| Causal Mediation / cGNF | 20 min | 1:20 |
| Double/Debiased ML | 20 min | 1:40 |
| Prediction-Powered Inference | 25 min | 2:05 |
| Student Presentations | 20 min | 2:25 |
| Discussion: AI as Subject | 25 min | 2:50 |
| Closing + Homework | 15 min | 3:05 |
| Buffer | 15 min | 3:20 |
