# Week 4: Experimental Designs with AI Agents and Human Interactions
## AI Agents for Social Science and Society 2026
### Ignite-Style Slide Deck — 30 Slides

---

## Slide 1: Correlation Is Not Causation — But It Keeps Fooling Us

**Visual Description:**
A large scatter plot showing a near-perfect positive correlation (r = 0.99) between two completely unrelated variables:
- X-axis: "US per capita cheese consumption (lbs)"
- Y-axis: "Number of people who died tangled in their bedsheets"

Both trend lines rise together over time (2000–2009). Source: Tyler Vigen's "Spurious Correlations."

Below, in large bold text: "A model that perfectly predicts this is perfectly useless for policy."

**Bullet Points / Text:**
- Prediction ≠ explanation ≠ causation
- Machine learning excels at the first, struggles with the last
- Social science needs all three

**Instructor Notes:**
Start with the provocative humor of spurious correlations to make the central problem viscerally clear. Every machine learning system we have built in this course so far is fundamentally a correlation machine — it finds patterns in historical data and uses them to predict outcomes. That is enormously useful, but it cannot answer the question that policy depends on: if we change X, what happens to Y? That requires causal inference.

---

## Slide 2: The Fundamental Problem of Causal Inference

**Visual Description:**
A split timeline diagram showing the "impossible experiment":
```
Reality:
  Person A ──► [Received treatment] ──► Outcome Y(1) = 7   ✓ OBSERVED
  Person A ──► [No treatment]       ──► Outcome Y(0) = ?   ✗ NEVER OBSERVED

What we want:
  Individual Treatment Effect = Y(1) - Y(0) = 7 - ? = ???

What we get:
  We can only observe ONE potential outcome per person.
  The counterfactual is permanently missing.
```

Below: Holland's (1986) phrase: "The fundamental problem of causal inference."

**Bullet Points / Text:**
- Counterfactual: what would have happened otherwise?
- We can only observe one potential outcome per person
- All causal inference is an answer to an unobservable question

**Instructor Notes:**
This is the foundational philosophical problem that every causal inference method is designed to address. The key insight is that causation requires comparing what happened to what would have happened under a different treatment — and the counterfactual is never observed. Every technique we cover today (randomization, matching, instrumental variables, DML) is a different strategy for credibly estimating this unobservable counterfactual.

---

## Slide 3: Potential Outcomes Framework — Rubin Causal Model

**Visual Description:**
A formal notation diagram:
```
For each unit i:
  Yᵢ(0) = outcome if NOT treated (potential outcome under control)
  Yᵢ(1) = outcome if treated (potential outcome under treatment)
  Tᵢ ∈ {0,1} = treatment indicator (observed)
  Yᵢ = Tᵢ·Yᵢ(1) + (1-Tᵢ)·Yᵢ(0) = observed outcome

Individual Treatment Effect:
  τᵢ = Yᵢ(1) - Yᵢ(0)   [never fully observed]

Average Treatment Effect:
  ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]   [estimable]
```

**Bullet Points / Text:**
- Potential outcomes: both possible worlds, only one realized
- ATE = average effect across all units
- SUTVA: one unit's treatment doesn't affect another's outcomes

**Instructor Notes:**
Walk through the notation carefully. The potential outcomes framework is the mathematical language of causal inference, and students who internalize it will be equipped for everything else in the week. Emphasize SUTVA (Stable Unit Treatment Value Assumption) — the assumption that what treatment I receive doesn't affect your outcome — because this assumption is routinely violated in social settings where people interact and influence each other.

---

## Slide 4: DAGs — Drawing Causation

**Visual Description:**
A clean Directed Acyclic Graph (DAG) showing a social science example:
```
Education ──────────────────────────────► Income
    │                                        ▲
    │                                        │
    └──► Occupation ──────────────────────────┘
    │
    └──► Social Network ──────────────────► Income

Background (confound): Family SES ──► Education
                                   └──► Income
```
Nodes: circles. Directed edges: arrows showing causal direction. "Back-door path" highlighted in red (SES → Education and SES → Income).

**Bullet Points / Text:**
- DAG: directed acyclic graph — causal structure as picture
- Confounders: variables that cause both X and Y
- Back-door criterion: block confounders to identify the effect

**Instructor Notes:**
DAGs are a beautiful tool for making causal assumptions explicit and visual. The key insight is that a confounding path goes from treatment to outcome through a back-door — a common cause of both. Blocking the back-door (by conditioning on the confounder, or by randomization) is what gives us a causal estimate. Pearl's do-calculus formalizes this graphically, and the regression adjustment, matching, and IV approaches are all implementations of back-door blocking.

---

## Slide 5: What Is a Treatment Effect?

**Visual Description:**
A concrete example diagram for the Reddit gender experiment:
```
TREATMENT DESIGN:
  Same comment posted in two conditions:
    Condition A: Comment posted under male username ("Mike_87")
    Condition B: Same comment under female username ("Emma_87")

OUTCOME:
  Upvotes received within 24 hours

TREATMENT EFFECT:
  ATE = E[Upvotes | female username] - E[Upvotes | male username]
      = difference attributable to perceived gender signal
```
A bar chart below showing the difference in upvote distributions between conditions, with error bars.

**Bullet Points / Text:**
- Treatment: the thing that varies (randomly or quasi-randomly)
- Outcome: what we measure as a result
- Treatment effect: the causal difference — not the raw difference

**Instructor Notes:**
The Reddit experiment is a memorable empirical example because it is both methodologically clean and sociologically striking. By holding the comment content constant and varying only the gender signal of the username, researchers can isolate the causal effect of perceived gender on engagement. This is the logic of randomized experiments applied to natural online settings — and it is something AI agents can now help design and scale.

---

## Slide 6: Heterogeneous Treatment Effects — Beyond the Average

**Visual Description:**
Two plots side by side:
Left: A single bar labeled "ATE = +2.3 upvotes for male usernames" — appears to show a uniform positive effect of perceived male gender.

Right: A forest plot showing the same effect broken down by subgroups:
- Political subreddits: large positive effect for male usernames
- Science subreddits: near-zero effect
- Gaming subreddits: moderate positive effect for female usernames
- Support subreddits: large positive effect for female usernames

Caption: "The average hides more than it reveals."

**Bullet Points / Text:**
- ATE: one number, hides heterogeneity
- CATE: Conditional ATE — varies by subgroup
- HTE: the effect is different for different people, contexts, platforms

**Instructor Notes:**
Heterogeneous Treatment Effects (HTE) are where deep learning adds genuine value to causal inference. The average treatment effect is often the least interesting causal quantity — what matters for policy is knowing for whom and in what contexts the treatment works. Deep learning models can estimate individual-level treatment effects by learning complex interaction patterns between treatment assignment and covariates that linear models miss entirely.

---

## Slide 7: S-Learner, T-Learner, TARNet — The HTE Toolkit

**Visual Description:**
Three architectural diagrams side by side:

S-Learner (Single model):
```
Input: [X, T] → [Model μ(X,T)] → Ŷ
CATE estimate: μ(X,1) - μ(X,0)
```

T-Learner (Two separate models):
```
Control group → [μ₀(X)] → Ŷ₀
Treatment group → [μ₁(X)] → Ŷ₁
CATE estimate: μ₁(X) - μ₀(X)
```

TARNet (Treatment-Agnostic Representation):
```
Input X → [Shared Representation Φ(X)]
                  │
         ┌────────┴────────┐
    [μ₀(Φ(X))]      [μ₁(Φ(X))]
         │                │
      Control head    Treatment head
CATE estimate: μ₁(Φ(X)) - μ₀(Φ(X))
```

**Bullet Points / Text:**
- S-Learner: simplest, may underfit treatment variation
- T-Learner: two models, may overfit small treated/control groups
- TARNet: shared representation + separate heads (best of both)

**Instructor Notes:**
Walk through each architecture carefully. The TARNet (Shalit et al. 2017) is the most theoretically principled for HTE estimation because the shared representation Φ(X) is encouraged to be balanced between treatment and control groups, mimicking the randomization that makes RCTs work. In the notebook (Module 1), students will implement all three and compare their CATE estimates on the Reddit dataset.

---

## Slide 8: TARNet — Architecture Deep Dive

**Visual Description:**
Detailed PyTorch-style diagram of TARNet with labeled layers:
```python
class TARNet(nn.Module):
    def __init__(self):
        # Shared representation
        self.rep = nn.Sequential(
            nn.Linear(d_x, 200), nn.ELU(),
            nn.Linear(200, 200), nn.ELU(),
            nn.Linear(200, 200), nn.ELU()
        )
        # Treatment head (T=1)
        self.head1 = nn.Sequential(
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, 1)
        )
        # Control head (T=0)
        self.head0 = nn.Sequential(
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, 1)
        )
    def forward(self, x, t):
        phi = self.rep(x)
        y1 = self.head1(phi)
        y0 = self.head0(phi)
        return t*y1 + (1-t)*y0, y1-y0  # outcome, CATE
```

**Bullet Points / Text:**
- Shared encoder: representation balanced across conditions
- Separate outcome heads: effects can differ by covariate pattern
- CATE at inference: run both heads, take the difference

**Instructor Notes:**
Showing the actual PyTorch code for TARNet makes the architecture concrete. Students who have implemented Week 1's feedforward networks will immediately recognize the pattern — it's the same `nn.Module` structure, just with a branching architecture at the prediction head. The key training insight is that you need to add a representation balancing term to the loss function (IPM or MMD) to prevent the shared encoder from learning to separate treatment and control groups.

---

## Slide 9: Causal Mediation Analysis

**Visual Description:**
A DAG showing mediation:
```
           ┌─────── M (Mediator) ──────────┐
           │    Indirect effect (IE)        │
           │                               ↓
Treatment T ──────────────────────────► Outcome Y
           Direct effect (DE)

Total Effect (TE) = DE + IE
```

Example:
```
T = Job training program
M = Employment status (does training get them a job?)
Y = Annual income (does income increase?)

DE: Direct: Training → Income (skills premium beyond employment)
IE: Indirect: Training → Employment → Income (income via job)
```

**Bullet Points / Text:**
- Mediation: the mechanism through which T affects Y
- Direct effect: T → Y not through M
- Indirect effect: T → M → Y
- Policy relevance: where should we intervene?

**Instructor Notes:**
Causal mediation analysis is one of the most policy-relevant causal methods because it answers the question: HOW does the treatment work? If a job training program increases income only by getting people employed, then a better employment policy might be more efficient. If there is a direct effect (the skills and certification have income value even for the already-employed), then training has broader value. Module 2 of today's notebook uses cGNF to estimate these decomposed effects.

---

## Slide 10: cGNF — Causal Graphical Normalizing Flows

**Visual Description:**
A diagram contrasting traditional mediation (linear/parametric) with cGNF:
Left (traditional):
```
Assumes: Y = α + βT + γM + δX + ε
         M = a + bT + cX + η
Problem: If T affects M, and M is a confounder of T→Y,
         standard regression is biased!
```

Right (cGNF):
```
Uses normalizing flows (neural networks) to model the
full joint distribution P(Y, M, T, X)

Allows: - Non-linear relationships
        - Exposure-induced confounding
        - Any structural causal model

Estimates: Natural direct and indirect effects
           without linearity or no-interaction assumptions
```

**Bullet Points / Text:**
- Traditional mediation: fails with exposure-induced confounders
- cGNF: neural network-based, handles complex causal structures
- Gold standard for modern causal mediation

**Instructor Notes:**
The key technical advance of cGNF (Causal Graphical Normalizing Flows) over traditional mediation is handling the case where the treatment affects a variable that confounds the mediator-outcome relationship — the exposure-induced confounding problem. Standard regression-based mediation analysis fails in this case (it produces biased estimates), but cGNF's flexible neural network approach can handle it correctly. Potter et al. (2024) uses this approach for the LLM political bias study.

---

## Slide 11: Double/Debiased Machine Learning

**Visual Description:**
A two-stage diagram illustrating the DML procedure:

Stage 1 — Nuisance estimation:
```
Model 1: E[T | X] → T̃ (predicted treatment)
         Residual: Ṽ = T - T̃ (variation in T unexplained by X)

Model 2: E[Y | X] → Ỹ (predicted outcome)
         Residual: Ũ = Y - Ỹ (variation in Y unexplained by X)
```

Stage 2 — Treatment effect:
```
Regress Ũ on Ṽ:
  θ̂ = (ṼᵀṼ)⁻¹ Ṽᵀ Ũ

Interpretation: Effect of T on Y, controlling for X using ML
```

Caption: "The residuals are what's left after ML has explained everything else."

**Bullet Points / Text:**
- DML: orthogonalize treatment and outcome on covariates
- Nuisance functions estimated by any ML method
- Neyman orthogonality: coefficient robust to nuisance errors

**Instructor Notes:**
Double ML (Chernozhukov et al. 2018) is one of the most elegant combinations of machine learning and causal inference. The key insight is Neyman orthogonality: by constructing an estimating equation whose sensitivity to the nuisance function errors is first-order zero, we can use flexible ML methods for the nuisance functions without contaminating our treatment effect estimate. This allows researchers to use neural networks, random forests, or boosted trees for confound adjustment without worrying about regularization bias.

---

## Slide 12: Double ML — The Partialling-Out Intuition

**Visual Description:**
A Venn diagram with three overlapping circles:
- Circle T: variation in treatment
- Circle Y: variation in outcome
- Circle X: variation in covariates (confounders)
- The T-X overlap and Y-X overlap are labeled "confounding variance"
- The T-Y overlap *outside* both T-X and Y-X is labeled "pure treatment effect"
- Arrow pointing to the pure overlap: "What DML estimates"

Caption: "Partial out the confounders with ML, then look at what's left."

**Bullet Points / Text:**
- Classic OLS: partialling-out via linear projection
- DML: partialling-out via ML (nonlinear, high-dimensional)
- Confidence intervals remain valid (unlike naive ML regression)

**Instructor Notes:**
The partialling-out intuition makes DML immediately relatable to students who know OLS. In OLS, the Frisch-Waugh-Lovell theorem shows that you can estimate any coefficient by regressing the residuals of Y on X against the residuals of T on X — and you get the same result as the full regression. DML extends this to the case where you use flexible ML for both residual regressions, allowing you to control for complex high-dimensional confounding that linear regression cannot handle.

---

## Slide 13: Prediction-Powered Inference

**Visual Description:**
A diagram contrasting three inference setups:
```
Setup 1: Human labels only
  N=100 human-annotated examples
  → Accurate but expensive; small N → wide confidence intervals

Setup 2: AI labels only
  N=10,000 AI-predicted labels
  → Cheap but biased; tight intervals but systematically wrong

Setup 3: Prediction-Powered Inference (PPI)
  N=100 human labels + N=10,000 AI labels
  Correction: θ̂_PPI = θ̂_AI + correction(θ̂_human - θ̂_AI)
  → Tight intervals AND unbiased!
```

Formula box:
```
θ̂_PPI = (1/n)Σᵢ f(Xᵢ)  +  (1/N)Σⱼ[Yⱼ - f(Xⱼ)]
          ─────────────      ────────────────────────
          AI predictions     Human rectification term
```

**Bullet Points / Text:**
- PPI: blend a small gold-standard set with large AI predictions
- Maintains statistical validity while leveraging AI scale
- Angelopoulos et al. 2023 — applicable to any parameter

**Instructor Notes:**
Prediction-Powered Inference is one of the most practically important methodological advances for computational social science. It solves a ubiquitous problem: we have millions of documents we want to annotate, but human annotation is expensive, while AI annotation is fast but imperfect. PPI uses a small human-labeled set to correct for the systematic errors in the AI labels, giving statistically valid inference over the full large dataset. Module 4 of today's notebook walks through this.

---

## Slide 14: The Mixed Subjects Design — Broska et al. 2025

**Visual Description:**
A 2x2 experimental design table:
```
                 | Human Subjects  | AI Agent Subjects
─────────────────+─────────────────+──────────────────
Treatment A      |    Cell 1       |     Cell 3
(positive frame) |   n=150 humans  |   n=1500 AIs
─────────────────+─────────────────+──────────────────
Treatment B      |    Cell 2       |     Cell 4
(negative frame) |   n=150 humans  |   n=1500 AIs
─────────────────+─────────────────+──────────────────
```
Annotations:
- "Compare Cells 1+2: human-only ATE"
- "Compare Cells 3+4: AI-only ATE"
- "Compare human vs. AI: validation of AI as proxy subject"
- "Use Cells 3+4 to power-up the human experiment"

**Bullet Points / Text:**
- Run the same experiment on humans AND AI agents
- AI cells: cheap, fast, large-N power
- Human cells: gold standard, validates AI proxy assumption

**Instructor Notes:**
The Mixed Subjects Design from Broska et al. (2025) is an elegant solution to the cost and scale constraints of human subject research. By running a small human experiment alongside a large AI experiment with the same stimuli and outcome measures, researchers can both validate the AI as a proxy for human subjects and use the AI results to generate hypotheses and power calculations for the human study. This is not "instead of" human subjects — it is "alongside" them.

---

## Slide 15: Are LLMs Valid Experimental Subjects?

**Visual Description:**
A weighing scale with two pans:
Left pan ("Evidence for validity"):
- "Replicates many known social psychology findings"
- "Produces realistic attitude distributions"
- "Responds to framing effects like humans"
- "Can be conditioned on demographic profiles"

Right pan ("Evidence against validity"):
- "Responses change with temperature, seed, paraphrase"
- "No genuine beliefs, preferences, or stakes"
- "Trained to be helpful, not authentic"
- "Systematic training biases may mimic but not equal human biases"

The scale is balanced, with a question mark at the pivot.

**Bullet Points / Text:**
- The question is not yes/no — it is for which questions?
- Validity varies by construct, population, and prompt design
- Use with human validation; never as sole evidence

**Instructor Notes:**
The question of LLM validity as an experimental subject is one of the most actively debated questions in social science methodology right now. Horton (2023) shows LLMs can replicate classic economics findings; critics show they can also produce systematically wrong results. The honest answer is that LLMs are valid proxies for some constructs and some populations in some contexts — and determining which requires the kind of careful validation work that Broska et al. formalize.

---

## Slide 16: Case Study — AI Persuading Humans to Abandon Conspiracy Theories

**Visual Description:**
Study design visualization for Costello, Pennycook & Rand (2024):
```
Study Population: 2,190 participants who believe at least 1 conspiracy theory

Random Assignment:
  Treatment: 3-round conversation with GPT-4 Turbo
             (personalized counter-arguments to their specific beliefs)
  Control:   3-round conversation about an unrelated topic

Primary Outcome: Change in conspiracy theory belief score (0-10)

Result: Treatment group: -20% reduction in belief strength
        Control group:   -0.8% reduction (regression to mean only)

Effect size: d ≈ 0.45  [Larger than most human persuasion interventions]
```

**Bullet Points / Text:**
- GPT-4 debunked conspiracy theories more effectively than human interlocutors
- Effect persisted 2 weeks later (follow-up survey)
- Personalization was key: generic debunking was less effective

**Instructor Notes:**
Costello et al. (2024) is one of the most striking papers on AI agents in social interaction. GPT-4 was not just as effective as human persuaders — it was more effective, likely because it could generate customized counter-arguments to each specific belief without the fatigue, frustration, or inconsistency of human debunkers. The 20% reduction in belief strength, persistent at 2-week follow-up, is a remarkable effect size and has obvious implications for public health communication, election integrity, and political polarization.

---

## Slide 17: Why AI Is Effective at Persuasion

**Visual Description:**
A comparison of human vs. AI persuader properties:
```
Property              | Human Persuader | GPT-4 Persuader
──────────────────────+─────────────────+─────────────────
Patience              | Runs out        | Infinite
Consistency           | Variable        | Highly consistent
Prior knowledge       | Limited         | Vast
Personalization       | Some            | High
Emotional regulation  | Can get angry   | Never escalates
Availability          | Expensive, rare | Cheap, scalable
Personal stake        | May have it     | None (perceived)
Persuasive tactics    | Learned (imperfect) | Evidence-based
```

Caption: "The properties that make AI effective at persuasion also make it ethically concerning."

**Bullet Points / Text:**
- Patience + personalization + scale = persuasion at unprecedented reach
- Who deploys this matters enormously
- Costello et al.: benign application. What about malign ones?

**Instructor Notes:**
The dual-use nature of AI persuasion technology is one of the most important ethical issues in this course. The same properties that make GPT-4 effective at reducing conspiracy beliefs — patience, personalization, scale, no emotional stake — also make it a powerful tool for manipulation, radicalization, and targeted influence campaigns. Potter et al. (2024) shows that LLMs already have measurable political biases that influence users. Students must engage with both sides of this.

---

## Slide 18: Hidden Persuaders — Potter et al. 2024

**Visual Description:**
A figure from the Potter, Lai, Kim, Evans, Song (2024) paper:
- A political attitude scale from strongly liberal to strongly conservative
- LLM responses shown as a distribution before and after interaction
- The distribution shifts left (more liberal) after conversation with an unsteered LLM
- The effect is measured across multiple political topics and multiple LLM models

Caption: "The LLM has a political lean. Users absorb it without knowing."

**Bullet Points / Text:**
- LLMs nudge users toward their political tendencies
- Effect is subtle, cumulative, and invisible to users
- Scale: millions of interactions daily

**Instructor Notes:**
Potter et al. (2024) is essential reading because it shows that the political bias of LLMs is not merely an academic concern — it has measurable effects on actual users' expressed political attitudes after interaction. The fact that users are unaware of the influence makes it more concerning, not less. This paper directly motivates the bias auditing methods students will learn in Week 9, and it is a reminder that the AI agents we are building are already part of the political environment.

---

## Slide 19: Biased AI Improves Human Decisions — Lai et al.

**Visual Description:**
Study design diagram for Lai, Kim, et al. "Biased AI Improves Human Decision-Making":
```
Task: Judges reviewing bail decisions

Setup 1: Baseline human judgment only
  → Outcome: [accuracy + racial disparity baseline]

Setup 2: Human + unbiased AI recommendation
  → Outcome: [improved accuracy, some disparity reduction]

Setup 3: Human + AI corrected for racial bias
  → Outcome: [highest accuracy, significant disparity reduction]

Surprise finding: Biased AI (corrected in opposite direction)
  outperforms unbiased AI in final human decision quality
```

**Bullet Points / Text:**
- Human + AI ≠ simply better; depends on how AI biases interact with human biases
- Debiased AI recommendations improve both accuracy AND fairness
- Calibrated AI bias can correct for human cognitive biases

**Instructor Notes:**
This paper contains one of the most counterintuitive findings in the AI + human decision-making literature: an AI system deliberately biased in the "corrective" direction — nudging human decision-makers away from their known biases — produces better outcomes than an unbiased AI. The implication is that the optimal human-AI partnership may involve asymmetric AI recommendations that account for human psychological tendencies. This is a rich research agenda and a direct application of causal inference methods.

---

## Slide 20: The Reddit Gender Experiment — Design Details

**Visual Description:**
A detailed experimental design infographic:
```
Platform: Reddit
Condition assignment: Random (coin flip per comment)
N: 2,000+ comments posted (100+ subreddits)

TREATMENT:
  Comment posted under: "UserName_M" (male signal)
  Comment posted under: "UserName_F" (female signal)
  Comment content: IDENTICAL across conditions

COVARIATES:
  - Subreddit category (politics, science, gaming, support)
  - Comment length
  - Time of day
  - Account age signal

OUTCOME:
  - Upvotes at 24h
  - Downvotes at 24h
  - Reply count
  - Sentiment of replies (BERT classifier)
```

**Bullet Points / Text:**
- Field experiment: real platform, real users, real behavior
- Randomization eliminates confounding by comment quality
- Heterogeneous effects by subreddit category = HTE analysis

**Instructor Notes:**
The Reddit experiment is a clean example of a natural field experiment where randomization is ethically and practically feasible. Students should notice how the heterogeneous treatment effects by subreddit category are the most interesting finding — the average effect of gender signal is much less informative than the interaction with topic domain. This motivates the TARNet architecture: you need a flexible model to capture these interaction patterns.

---

## Slide 21: Module 1 Walk-Through — HTE in Practice

**Visual Description:**
Annotated notebook output showing HTE analysis results:
1. A "CATE distribution" plot — histogram of individual treatment effect estimates from TARNet, showing a wide distribution (some individuals have large positive effects, others negative)
2. A "feature importance for CATE" horizontal bar chart — showing which covariates most strongly moderate the treatment effect (subreddit category ranks #1, gender of poster and commenter rank #2/#3)
3. A calibration plot — estimated CATE vs. realized outcome difference by quintile

Code snippet:
```python
model = TARNet(input_dim=X.shape[1])
trainer = CausalTrainer(model, ipm_weight=0.1)
trainer.fit(X, T, Y, epochs=200)
cate_estimates = model.predict_cate(X_test)
```

**Bullet Points / Text:**
- Fit TARNet on your T/Y/X data
- CATE: one number per person (individual effect)
- Calibration: does the model's uncertainty match reality?

**Instructor Notes:**
This slide previews what Module 1 of the notebook produces. Walk students through the three key outputs: the distribution of CATE estimates (is the effect really heterogeneous?), the feature importance analysis (what drives heterogeneity?), and calibration (can we trust the uncertainty estimates?). All three are necessary to make a credible HTE claim — too many published papers report point estimates without checking calibration.

---

## Slide 22: DAG Workshop — Drawing Your Research Question

**Visual Description:**
Three example DAGs for common social science research questions, drawn side by side:

DAG 1 — Social media and mental health:
```
Social media use ──► (Comparison) ──► Depression
Social media use ──► (Connection) ──► Depression
Background: Family history ──► Social media use
                           └──► Depression (confounder)
```

DAG 2 — Education and earnings:
```
Education ──► Occupation ──► Earnings
Education ──────────────────► Earnings
Family SES ──► Education (confounder)
           └──► Earnings
```

DAG 3 — AI persuasion:
```
AI conversation ──► Belief change
AI conversation ──► Trust in source ──► Belief change
Prior belief strength ──► Belief change (moderator)
```

**Bullet Points / Text:**
- Drawing the DAG forces you to state your assumptions
- Every assumption can be challenged
- The DAG determines the identification strategy

**Instructor Notes:**
Have students spend 5 minutes drawing the DAG for their own research question. The act of drawing forces explicit commitment to causal assumptions that are often left implicit in social science research. The rule is: if you can draw the DAG and identify your estimand, you know what method to use. If you can't draw the DAG, you don't yet understand your research question well enough to choose a method.

---

## Slide 23: Identification Strategies — Matching the Method to the Design

**Visual Description:**
A decision flowchart for causal identification:
```
Do you have randomization?
  YES → RCT (gold standard) → Standard ATE estimator
  NO  ↓
Is there an instrument (variable that affects T but not Y directly)?
  YES → Instrumental Variables (IV)
  NO  ↓
Is there a sharp cutoff in treatment assignment?
  YES → Regression Discontinuity (RD)
  NO  ↓
Did treatment occur at a specific time?
  YES → Difference-in-Differences (DiD)
  NO  ↓
Can you assume unconfoundedness after conditioning on observables?
  YES → DML / Matching / IPW
  NO  → Sensitivity analysis + honesty about limits
```

**Bullet Points / Text:**
- Different designs solve different identification problems
- No design works without assumptions
- State and defend your assumptions explicitly

**Instructor Notes:**
This flowchart is a practical tool students can use when designing their own research. The key message is that causal identification is a design problem, not a statistical problem — you cannot "analyze your way out of" a confounded design. The role of DML and the other techniques in this week is to make the most of the design you have, not to substitute for good design in the first place.

---

## Slide 24: Large Language Models as Simulated Economic Agents — Horton 2023

**Visual Description:**
A replication of classic economics experiments using GPT-3 as subjects:
```
Experiment 1: Ultimatum Game
  Human results: ~50% rejection of "unfair" offers (≤30%)
  GPT-3 results: Similar rejection rates when prompted as
                 "an individual with normal economic preferences"

Experiment 2: Dictator Game
  Human results: Significant portion give >0 (altruism)
  GPT-3 results: Positive giving rates, responsive to framing

Experiment 3: Public Goods Game
  Human results: Moderate contribution levels, declining
  GPT-3 results: Similar contribution levels, less decline
```

**Bullet Points / Text:**
- GPT-3 replicates known behavioral economics findings
- More consistent than human subjects (less random noise)
- But: does it generalize? Or does it mimic published findings?

**Instructor Notes:**
Horton (2023) is the paper that launched serious discussion of LLMs as synthetic economic subjects. The results are striking — GPT-3 behaves similarly to human subjects in classic behavioral economics experiments. But the critical methodological concern is Goodhart's Law applied to LLMs: if the model was trained on the papers describing these experiments, it may be "mimicking" published findings rather than independently replicating them. Students should distinguish between genuine behavioral replication and sophisticated pattern completion.

---

## Slide 25: Designing Experiments With AI Agents — A Protocol

**Visual Description:**
A structured protocol diagram:
```
Phase 1: DESIGN
  □ State the causal question and estimand
  □ Draw the DAG
  □ Choose identification strategy
  □ Power analysis (with AI subjects for cheap pilots)

Phase 2: IMPLEMENTATION
  □ Design stimuli and treatments
  □ Configure AI personas (if mixed design)
  □ Randomize assignment
  □ Pre-register (OSF or AsPredicted)

Phase 3: VALIDATION
  □ Manipulation checks
  □ Attention/quality checks (human subjects)
  □ Consistency checks (AI subjects)
  □ Compare AI vs. human results (mixed design)

Phase 4: ANALYSIS
  □ Pre-registered estimators first
  □ HTE analysis as secondary
  □ Mediation if theoretically motivated
  □ Sensitivity analysis for unconfoundedness
```

**Bullet Points / Text:**
- Pre-register before you collect data
- AI subjects: cheap pilot → refine for human study
- Sensitivity analysis: what would have to be true to overturn your result?

**Instructor Notes:**
This protocol is directly actionable for students' final projects. Emphasize pre-registration — declaring your hypotheses, design, and analysis plan before data collection — as a norm the field is increasingly requiring. The use of AI agents for cheap pilot studies to test manipulation quality and effect size estimates before investing in expensive human subject data is one of the most practical applications of this week's content.

---

## Slide 26: Statistical Power and AI Agents

**Visual Description:**
A power curve comparison:
- X-axis: Sample size N
- Y-axis: Statistical power (0 to 1)
- Dashed horizontal line at 0.80 ("conventional threshold")

Three curves:
- Blue: Human subjects experiment (typical cost ~$15/person on MTurk)
  → 80% power reached at N=400 → cost: $6,000
- Red: AI agent experiment (cost ~$0.02/agent via API)
  → 80% power reached at N=400 → cost: $8
- Green: Mixed design (small human N + large AI N + PPI)
  → 80% power reached at N=80 human + N=4000 AI → cost: $1,280

**Bullet Points / Text:**
- Human experiments: expensive, limited N
- AI experiments: cheap, large N, but validity questions
- Mixed design: valid inference + scale at moderate cost

**Instructor Notes:**
The power analysis comparison makes the economic logic of mixed designs immediately clear. Human subject experiments are expensive — recruiting, paying, and retaining participants limits sample sizes and therefore statistical power. AI agent experiments are orders of magnitude cheaper, allowing much larger samples. The mixed design gets you both: human-validated inferences at scale. This is the practical case for Broska et al.'s methodology.

---

## Slide 27: Ethics of AI Experimental Subjects — IRB in the Age of Digital Doubles

**Visual Description:**
A side-by-side comparison of human subjects protections and their AI analog:
```
Human Subjects (IRB)          AI Agent Subjects (Current Practice)
────────────────────────       ──────────────────────────────────────
Informed consent required      No consent mechanism
Right to withdraw              No equivalent
Anonymization required         API logs may store data
Risk/benefit assessment        Rarely formalized
Institutional oversight (IRB)  No institutional equivalent
Debrief requirement            N/A
```
Below: a large question mark and the caption: "The norms are still being written. You are writing them."

**Bullet Points / Text:**
- AI agents as subjects: current IRB frameworks don't apply
- New risks: manipulating LLMs may train on outputs
- Gabriel et al. 2025: "We Need a New Ethics for AI Agents"

**Instructor Notes:**
The absence of institutional review for AI agent experiments is both a practical convenience and a genuine ethical gap. While AI agents do not have wellbeing interests that IRB protections are designed to safeguard, experiments using AI agents still have ethical dimensions: they can be used to develop more effective manipulation techniques, the outputs may become training data that shapes future AI behavior, and the knowledge produced can be misused. Students should develop their own principled stance on these questions.

---

## Slide 28: From Correlation to Causation — The Full Stack

**Visual Description:**
A visual summary connecting this week's methods to the broader course:
```
WEEK 1: "The neural network correlates demographics with voting"
         → Prediction, not causation
              ↓
WEEK 3: "Digital doubles simulate voting behavior"
         → Mechanism, not validation
              ↓
WEEK 4: "Experimental design + DML + HTE"
         → CAUSATION: what actually changes behavior?
              ↓
WEEKS 5-9: "Fine-tuning, interpretability, RL, safety"
            → Causal mechanisms inside the model
```

**Bullet Points / Text:**
- We have built the prediction tools (Weeks 1-3)
- This week: add the causal layer
- The course progression: observe → predict → simulate → cause

**Instructor Notes:**
This architecture diagram shows where Week 4 sits in the course's intellectual arc. The progression from correlation (Week 1) to simulation (Week 3) to causal inference (Week 4) is not an accident — it mirrors the progression from descriptive to predictive to causal social science. Students who complete this week's notebook will have a more sophisticated causal inference toolkit than most practicing social scientists.

---

## Slide 29: What Makes This Experimental Design Hard

**Visual Description:**
A three-panel visualization of three fundamental threats to causal inference in AI experiments:

Panel 1 — SUTVA Violation:
```
Agent A changes its belief after talking with Agent B
→ Agent B's "treatment" affects Agent A's "outcome"
→ Violations of Stable Unit Treatment Value Assumption
```

Panel 2 — Demand Effects:
```
Human: "I'm testing whether GPT-4 is biased."
GPT-4: [knows it's being tested, performs accordingly]
→ Same as Hawthorne effect, but AI can be more sensitive to cues
```

Panel 3 — Prompt Sensitivity:
```
"Do you support increased government spending?" → 62% Yes
"Do you favor government waste?" → 22% Yes
→ Treatment effects depend on exact wording
→ Pre-registration of prompts is essential
```

**Bullet Points / Text:**
- SUTVA: agents can contaminate each other (social contagion)
- Demand effects: AI may behave differently when tested
- Prompt sensitivity: results may not generalize across phrasings

**Instructor Notes:**
These three threats are specific to AI experimental designs and do not all have established solutions. SUTVA violations require thinking carefully about agent network structure; demand effects require masking the research purpose from agents (which is harder than it sounds — LLMs are good at detecting evaluation contexts); prompt sensitivity requires systematic robustness testing across multiple phrasings. Students designing AI experiments should address all three explicitly.

---

## Slide 30: AI Agents Are Subjects and Scientists

**Visual Description:**
A closing visual showing the full arc of transformation:

Timeline of AI's role in social science:
```
2000s: AI as TOOL
  "Use ML to classify text, detect patterns"
  [AI is passive instrument]

2010s: AI as ANALYST
  "Use LLMs to read, summarize, code qualitative data"
  [AI is active collaborator]

2020s: AI as SUBJECT
  "LLMs as experimental participants in social studies"
  [AI is a research object AND participant]

2026+: AI as ACTOR
  "AI agents operating in society — persuading, deciding, governing"
  [AI has causal effects on the social world being studied]
```

Bold final statement: "The boundary between research instrument and research subject is dissolving."

**Bullet Points / Text:**
- AI is not just a tool for analysis — it is a participant in society
- Agents in our experiments are becoming actors in our world
- The same methods that study AI effects must now account for AI as cause

**Instructor Notes:**
Close with the course's most profound methodological and philosophical challenge. The tools we are building and studying are not neutral instruments — they are participating in the social world they are designed to study. An AI agent that persuades humans to change their beliefs, an LLM that moderate content, an algorithm that allocates jobs — all of these are now causal agents in the social fabric. The social scientist who studies these systems is studying something that studies back. That reflexivity is the defining intellectual challenge of the decade.

---

*End of Week 4 Slide Deck — 30 Slides*
*AI Agents for Social Science and Society 2026*
*Instructor: James A. Evans | January 30, 2026*
