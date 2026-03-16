# Week 7 Lecture Script
# AI Agents for Social Science and Society 2026
# Reinforcement Learning to Optimize AI Agents and Institutions
# Date: February 20, 2026 | 1:30–4:20 PM | Room 295

---

## Instructor Notes: Before Class

- Open `Week_7_RL.ipynb` in Colab and pre-run through Cell 10 to confirm installations work
- Have the OpenAI Gymnasium CartPole visualization ready to demo (Cell 22)
- Pre-load the IPD (Iterated Prisoner's Dilemma) results plot from Cell 64
- Write the MDP tuple on the board before students arrive: **(S, A, P, R, γ)**
- Estimated total class time: 170 minutes (with a 10-minute break around the 90-minute mark)

---

## Section 1: Opening and Connecting to Prior Weeks
### [0:00 – 0:15 | 15 minutes]

**Say:** "Good afternoon, everyone. Let's do a quick temperature check on last week. We spent Week 6 opening up the black box — mechanistic interpretability, sparse autoencoders, and steering vectors. We learned how to *read* what's inside an LLM. The obvious next question is: how do we change what's inside? Not through prompting, not through fine-tuning on fixed data, but through something fundamentally more active — through *experience and feedback*."

**Write on board:**
```
Week 6: Static agent → we can READ it
Week 7: Learning agent → it CHANGES from experience
```

**Ask students:** "Can anyone give me an example from everyday life of learning through feedback — not instruction, but direct trial-and-error experience?"

*Expected responses: learning to ride a bike, playing video games, training a dog, a child touching a hot stove.*

**Say:** "Exactly. Reinforcement learning is the computational formalization of that process. And it turns out to be one of the most powerful ideas in all of AI — it's what gave us AlphaGo, ChatGPT's fine-tuning, and DeepSeek-R1's chain-of-thought reasoning. Today we're going to build it from the ground up."

**Write on board:**
```
Core message: RL transforms static models into agents that pursue goals,
align with human values, and exhibit reasoning.
Multi-agent RL can model the emergence of social norms and institutions.
```

---

## Section 2: RL Foundations — MDPs, Policies, and Value Functions
### [0:15 – 0:55 | 40 minutes]

**Say:** "Let me start with an analogy that will carry us through this entire section. Imagine a rat in a maze. At each junction, it chooses left or right. Sometimes it finds cheese; sometimes a dead end. Over hundreds of trials, it learns which junctions lead to cheese. That rat is running reinforcement learning. Now, formally, we describe this with a **Markov Decision Process**, or MDP."

**Write on board (keep visible for the whole section):**
```
MDP = (S, A, P, R, γ)
  S = states (where you are)
  A = actions (what you can do)
  P(s'|s,a) = transition probability (what happens)
  R(s,a) = reward (what you get)
  γ ∈ [0,1] = discount factor (how much you care about the future)
```

**Say:** "The Markov property is the key simplifying assumption: the future depends only on the present state, not on the full history. This is why we say the state must be a *sufficient statistic* of history. If it isn't, the MDP is only an approximation — which is very often the case in real social science applications."

**Ask students:** "What would 'state' and 'action' look like for a social media recommendation algorithm? Think about what the agent knows and what it controls."

*Expected responses: state = user browsing history, current content queue, time-of-day; actions = which post to show next; reward = click, like, time spent.*

**Say:** "Perfect. And this is where it gets interesting for social science. The reward function is a moral claim. Whoever defines R is defining what the system is optimizing for. In the social media case, if R = engagement time, the system may learn to show outrage-inducing content because outrage maximizes time-on-platform. This is reward misspecification — and we'll come back to it in depth in Week 9."

**Transition to the notebook — Cell 6 (Multi-Armed Bandits):**

**Say:** "Before we get to full MDPs, let's start with the simplest RL setting: the **multi-armed bandit**. In the notebook, Cells 6–11 walk through this."

**Code Demo Note — Cell 9:**
```python
from arms.bernoulli import BernoulliArm
from algorithms.epsilon_greedy.standard import EpsilonGreedy
```
Point to the ε parameter. **Say:** "ε controls the exploration-exploitation tradeoff. If ε=0, the agent always exploits its current best guess. If ε=1, it acts randomly. What should ε be for a social science survey experiment where you want to learn something new but also maximize good outcomes for participants?"

*This is a genuine open question. Expect: "something in between," "depends on sample size," "smaller ε as you learn more."*

**Say:** "The plot in Cell 10 shows exactly this tension. Look at the left panel: high ε finds the good arm faster but keeps exploring even after finding it. Low ε settles quickly but might lock in a sub-optimal choice. The right panel shows smarter algorithms — UCB (Upper Confidence Bound) and Thompson Sampling — that adapt their exploration automatically."

**Transition to TD Learning — Cell 16:**

**Say:** "Now let's move from bandits to full MDPs. We have states, actions, and transitions. How do we estimate the value of being in a state? Two key methods: **Monte Carlo** and **Temporal Difference (TD) learning**."

**Write on board:**
```
Monte Carlo: wait until episode ends, then update from total return
TD Learning: update after every step using:
  V(s) ← V(s) + α [R + γV(s') - V(s)]
                    └── TD error ──┘
```

**Say:** "The TD error — that bracketed term — is the surprise: how much better or worse did things turn out than you expected? This is neuroscientifically grounded. Dopamine neurons in the brain fire exactly in proportion to TD error. RL is not just a computational abstraction; it's also a theory of learning in biological agents."

**Code Demo Note — Cell 16:**
Show the MountainCar-v0 environment. **Say:** "The MountainCar problem is a nice toy with a social science interpretation: a car trapped in a valley. It has to swing back and forth to gain momentum. The reward is -1 per timestep until it reaches the top. This is a delayed-reward problem — progress isn't immediately visible. A lot of social investments are like this: education, public health infrastructure. The agent needs temporal credit assignment."

**Ask students:** "What is the discount factor γ doing here, and what does it mean to set γ close to 1 versus close to 0?"

*Expected: γ near 1 means the agent cares about the distant future (patient); γ near 0 means myopic. Connect to social science: short-term vs. long-term political thinking.*

---

## Section 3: Deep RL — DQN, Actor-Critic, and PPO
### [0:55 – 1:25 | 30 minutes]

**Say:** "Tabular Q-learning works great when there are few states. But what happens when the state space is huge? Like, say, the pixel values of a video game screen — 210×160 pixels times 3 color channels. You can't have a table entry for every possible screen. This is where **Deep Q-Networks (DQN)** come in — you replace the Q-table with a neural network."

**Write on board:**
```
DQN innovations (Mnih et al., 2015 Nature):
1. Experience replay: store (s,a,r,s') tuples, sample randomly
2. Target network: separate, slowly-updated network to compute targets
3. ε-greedy exploration: decay ε over training
```

**Say:** "Why does experience replay matter? Two reasons. First, real experience is correlated in time — if you see a car crash, the next frame also has a car crash. Neural networks trained on correlated data overfit badly. Randomly sampling from a replay buffer breaks that correlation. Second, you reuse expensive experience — one game transition trains the network multiple times."

**Code Demo Note — Cell 20:**
Run the DQN implementation on CartPole-v1. **Say:** "CartPole is the 'Hello World' of RL. The agent must balance a pole on a cart by pushing left or right. Watch the reward curve — it should start near 10 (random policy) and climb toward 200 (near-perfect). Note how unstable early training is — this is DQN's sensitivity to hyperparameters."

**Transition to PPO:**

**Say:** "Policy gradient methods — and their modern descendent, PPO — take a different approach. Instead of learning a value function and acting greedily, they directly optimize the policy. The challenge: policy gradient updates can be too large, destabilizing training. PPO's clever fix is the **clipped objective**."

**Write on board:**
```
PPO clipped objective:
L_CLIP = E[min(r_t(θ)·Â_t,  clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)

This prevents the new policy from moving too far from the old one.
```

**Say:** "The ratio r_t(θ) tells us how much more or less probable action a is under the new policy than the old one. Clipping it between (1-ε, 1+ε) means we only trust small policy updates. This gives us the stability of trust-region methods without expensive second-order optimization."

**Code Demo Note — Cell 22:**
```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1", verbose=1, n_steps=256, n_epochs=10)
model.learn(total_timesteps=10_000)
```
**Say:** "Three lines to get a functional PPO agent. This is how production RL is done — Stable Baselines3 handles all the engineering. Your job as a social scientist is to design the environment and reward function."

**[BREAK — 10 minutes]**

---

## Section 4: RLHF, Constitutional AI, and RL for Reasoning
### [1:35 – 2:05 | 30 minutes]

**Say:** "Now we come to the most socially significant application of RL in recent AI: using it to align language models with human values. The key paper is Ouyang et al. 2022 — InstructGPT — which became the foundation for ChatGPT."

**Write on board:**
```
RLHF Pipeline (3 steps):
1. Supervised Fine-Tuning (SFT): train on human demonstrations
2. Reward Model (RM): train on human preference comparisons (A vs B)
3. RL Optimization: optimize policy to maximize RM score (with KL penalty)
```

**Say:** "The genius of step 2 is that asking humans 'which response is better?' is much easier than asking them to write the perfect response. You can get cheap, consistent preference labels at scale. The RM learns to proxy human preferences, and then PPO optimizes the LLM against this proxy."

**Ask students:** "What could go wrong with this pipeline? What are the failure modes?"

*Expected: the RM is only a proxy — Goodhart's Law; the RM may encode biases of the labelers; the policy might over-optimize the RM while drifting from actual human values (reward hacking).*

**Say:** "All of these happen in practice. The KL penalty in step 3 — comparing the new policy to the original SFT model — is specifically there to prevent the policy from collapsing into a 'reward-hacking' mode where it gets high RM scores for gibberish."

**Code Demo Note — Cells 25–47 (Module 2):**

**Say:** "In the notebook, Module 2 walks through all three RLHF steps. In Cell 29, you fine-tune a reward model using the `ultrafeedback_binarized` dataset. In Cell 44, you run PPO fine-tuning with that reward model. A key finding is on Cell 47: your 0.5B RM achieves only ~54% accuracy — barely above chance. This is a known problem. The notebook then uses a pre-trained DeBERTa reward model which does better. The lesson: reward models are hard to train well, and weak reward models produce misaligned policies."

**Transition to Constitutional AI — Cell 48:**

**Say:** "Bai et al. (2022) from Anthropic proposed an elegant fix: instead of human labelers, use the AI itself to generate preference labels according to a set of written *principles* — a constitution. This is **Constitutional AI**."

**Write on board:**
```
Constitutional AI (CAI / RLAIF):
1. Generate initial response
2. Critique: "Does this violate principle X?" → critique
3. Revise: generate a revised response using the critique
4. Use revised responses to train the RM (replacing human labels)
```

**Say:** "This is deeply interesting for social scientists. A 'constitution' is literally a normative framework. Anthropic's constitution includes principles like 'don't be harmful' and 'respect autonomy.' But who writes the constitution? Who decides which principles matter? This is exactly the question political philosophers have grappled with for centuries."

**Ask students:** "Think about the social norm analogy. In human societies, how do individuals internalize norms? Where do those norms come from?"

*Expected: socialization, family, religion, law, peer pressure, media. Note that CAI is an artificial version of this — norm internalization through self-critique and revision.*

**Transition to GRPO and DeepSeek-R1 — Module 3:**

**Say:** "The most recent development is using RL not just to align models but to make them *reason better*. DeepSeek-R1 (Guo et al., 2025) showed that you can train a model to develop chain-of-thought reasoning entirely through RL — without any human reasoning demonstrations."

**Write on board:**
```
GRPO (Group Relative Policy Optimization):
- Generate G completions for each prompt
- Reward each completion r_i
- Advantage for completion i: A_i = (r_i - mean(r)) / std(r)
  → No learned critic needed
- Update policy to favor above-average completions
```

**Say:** "The remarkable thing about DeepSeek-R1 is that it exhibited what the authors called 'aha moments' — the model spontaneously started generating longer, more careful reasoning chains during RL training, without being told to. The model discovered that thinking helps."

**Code Demo Note — Cells 56–59 (Module 3):**
**Say:** "Cell 56 loads a math reasoning dataset (NuminaMath-TIR) and Cell 57 runs GRPOTrainer. The reward is simple: if the final answer is correct, +1; otherwise 0. That's it. No human labels. Just a verifiable answer. The model learns to reason in order to get more right answers. Kim, Lai et al.'s 'Societies of Thought' paper extends this further — showing that when reasoning models like DeepSeek interact, their reasoning traces form emergent social dynamics, with agents influencing each other's conclusions."

---

## Section 5: Multi-Agent RL and the Emergence of Social Norms
### [2:05 – 2:30 | 25 minutes]

**Say:** "Now we arrive at the most directly social-scientific part of this lecture: what happens when multiple RL agents interact? Can we model the emergence of cooperation, competition, and social norms through multi-agent RL?"

**Write on board:**
```
Multi-Agent RL settings:
- Fully cooperative: all agents share one reward
- Fully competitive: zero-sum (one agent's gain = another's loss)
- Mixed: each agent has its own reward; may cooperate or defect
```

**Code Demo Note — Cells 62–64 (Module 4, Iterated Prisoner's Dilemma):**

**Say:** "The Prisoner's Dilemma is the canonical game-theoretic model of cooperation. Two agents independently choose to cooperate or defect. Mutual cooperation yields (3,3). Mutual defection yields (1,1). Unilateral defection yields (5,0). Nash equilibrium is mutual defection — but it's socially suboptimal."

**Ask students:** "What do you expect RL agents to learn in the IPD over many repeated rounds? What would you expect humans to learn?"

*Expected: students may say 'defect' since it's NE; some may say 'cooperate' since it's repeated. Both are right in different contexts.*

**Say:** "Cell 64 shows something fascinating: the cooperation rate varies dramatically with the reward parameter alpha. When alpha=0, agents play standard IPD and learn to defect. When alpha=1, agents care more about joint payoff — they learn to cooperate. This is a model of institutional design: change the payoff structure, change the emergent behavior."

**Code Demo Note — Cell 66 (Theory of Mind Agent):**
**Say:** "Cell 66 introduces a Theory of Mind agent — one that maintains hypotheses about its opponent's strategy and acts accordingly. Compare its cooperation rate to the plain Q-learning agent. The ToM agent cooperates more, because it reasons about the other agent's intentions. This is Hypothetical Minds from Cross et al. (2025): agents that model others' minds learn better policies in social environments."

**Code Demo Note — Cell 68 (Counterfactual Credit Assignment):**
**Say:** "In team settings, there's a credit assignment problem: if the team wins, who gets the credit? Cell 68 shows that counterfactual rewards — 'what would the team have gotten if I had acted differently?' — produce better cooperation than simply splitting the team reward equally."

**Ask students:** "Can you think of a real institution that uses counterfactual credit assignment?"

*Expected: performance-based pay, academic citation networks, legal liability in joint ventures. The point is that institutions are RL environments — their reward structures shape emergent behavior.*

---

## Section 6: Code Walkthrough Integration and Homework Briefing
### [2:30 – 2:55 | 25 minutes]

**Say:** "Let me now connect what we've seen in lecture to the homework structure. Module 1 (Cells 4–23) is the RL foundations — bandits, Q-learning, DQN, PPO. This is required. Module 2 (Cells 24–53) is the RLHF pipeline — this is also required. Then you choose one of Module 3 (GRPO/reasoning) or Module 4 (multi-agent RL)."

**Write on board (homework checklist):**
```
Required:
  Module 1: RL foundations (DQN or PPO on a benchmark env)
  Module 2: Reward model + PPO fine-tuning of an LM

Choose one:
  Module 3: GRPO for reasoning task — measure improvement
  Module 4: Multi-agent environment — measure emergent behavior

Final HW: Apply RL concepts to your final project domain
```

**Code Demo Note — Module 1, Section F (Cell 20):**
Walk through the DQN architecture briefly:
```python
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)
```
**Say:** "The architecture is just a feedforward network from Week 1. The magic of DQN is not the network — it's experience replay and target networks. Keep the network simple; the RL algorithm does the work."

**Ask students:** "For your final project — what would be the state, action, and reward if you wanted to use RL to model an institution or agent behavior in your domain?"

*Give students 3 minutes to think and share.*

---

## Section 7: Discussion — Can RL Design Better Social Institutions?
### [2:55 – 3:20 | 25 minutes]

**Say:** "I want to close with a bigger question that connects all of today's material. We've seen that RL agents can learn to cooperate, defect, reason, and align with values. Can we use this machinery to design better social institutions?"

**Ask students:** "The reward function in RL is the institution's incentive structure. Think about a real institution — a court system, a market, a university. What is its reward function? Who designed it? Who is it optimal for?"

*Expected: complex answers. Courts reward conviction rates (which may bias toward harsh sentencing). Markets reward profit (which may externalize harms). Universities reward publications (which may incentivize quantity over quality).*

**Write on board:**
```
Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."
RL version: Reward hacking — agents find unexpected ways to maximize R
              that violate the spirit of the design.
```

**Say:** "The DeepSeek-R1 'aha moment' is the positive version of this: an agent discovering a better way to pursue the reward than the designers anticipated. Reward hacking is the negative version: an agent discovering a way to exploit the measurement. Social institutions face exactly this tension — designing metrics and incentives that are hard to game."

**Ask students:** "GRPO trains reasoning through RL. Kim, Lai et al.'s 'Societies of Thought' shows that reasoning models generate emergent social dynamics. What are the implications for deliberative democracy — if AI agents are used to assist in collective decision-making?"

*Discuss 5–8 minutes. Key tensions: AI reasoning can be a tool for more rigorous deliberation OR a tool for sophisticated manipulation. The same RL machinery that trains a model to reason well can also train it to reason persuasively in service of a hidden reward.*

**Say:** "Constitutional AI gives us one answer: encode values explicitly in a constitution and let the AI self-critique against those values. But as we noted, the question of who writes the constitution is irreducibly political. Social scientists — particularly political theorists, ethicists, and sociologists — have crucial expertise to bring to that question. That's not a problem for AI engineers to solve alone."

---

## Closing Summary
### [3:20 – 3:30 | 10 minutes]

**Write on board:**
```
Today's arc:
Bandit → MDP → DQN/PPO → RLHF → CAI → GRPO → Multi-Agent RL
Simple feedback → Complex reasoning → Social norms
```

**Say:** "Here's the one-paragraph summary of today. Reinforcement learning is the framework for agents that improve from experience. The MDP formalizes the environment; the policy and value function formalize the agent. Deep RL scales this to complex state spaces. RLHF uses RL to align LLMs with human preferences — but this is hard because reward models are noisy proxies. Constitutional AI replaces human labels with principled self-critique. GRPO shows that RL can grow reasoning capability from scratch. And multi-agent RL shows how cooperation, defection, and social norms emerge from individual reward-maximizing behavior."

**Say:** "For next week — Week 8 — we move to multimodal agents. We've been working entirely in the text domain. But AI agents increasingly perceive the world through images, audio, and video. And that means their biases about what the world looks like become baked into perception, not just language."

**Homework reminders:**
- Complete Modules 1 and 2 of `Week_7_RL.ipynb`, plus Module 3 or 4
- Weekly memo due before class: 300–500 words connecting one of the required readings to your final project
- Lab session with Avi Oberoi: Tuesday 11am–12pm — bring your RL environment design questions
- Reading for Week 8: Yao et al. 2025 "REACT" and Guilbeault, Delecourt, Desikan 2025 "Age and Gender Distortion"

---

## Appendix: Anticipated Student Questions and Instructor Responses

**Q: What's the difference between model-based and model-free RL?**
A: Model-free RL (DQN, PPO) learns directly from experience without building an explicit model of the environment. Model-based RL (e.g., AlphaZero) learns a model of P(s'|s,a) and then plans using that model. Model-based methods are more sample-efficient; model-free methods are more robust when the environment is complex and stochastic.

**Q: Is RLHF the same as what makes ChatGPT work?**
A: Yes — InstructGPT (Ouyang et al. 2022) is the direct predecessor to ChatGPT. The key insight is that human feedback on response quality is a better training signal than maximum likelihood on a text corpus for producing helpful, harmless assistants.

**Q: Can RL be used for actual policy design in government?**
A: Researchers have experimented with this (AI Economist, TaxAI). The challenge is that real social environments are not stationary MDPs — they adapt to the policy, and the reward function is deeply contested. RL is perhaps most useful as a simulation tool for exploring institutional design before deployment.

**Q: How do you handle non-Markovian real-world environments in RL?**
A: Typical approaches are: (1) augment the state with history (recurrent networks), (2) use transformer architectures that attend over full history, (3) accept the Markovian approximation and monitor for drift.

---

## Key Papers Referenced Today

| Paper | One-line summary |
|---|---|
| Mnih et al. (2015) Nature | DQN: deep Q-networks beat human performance on Atari games |
| Ouyang et al. (2022) InstructGPT | RLHF aligns LLMs with human feedback via PPO |
| Bai et al. (2022) Constitutional AI | Replace human RLHF labels with AI self-critique via a constitution |
| Guo et al. (2025) DeepSeek-R1 | RL incentivizes chain-of-thought reasoning; emergent 'aha moments' |
| Kim, Lai et al. (2025) Societies of Thought | Reasoning models generate emergent social dynamics |
| Cross et al. (2025) Hypothetical Minds | Theory of Mind scaffolding improves multi-agent RL |
