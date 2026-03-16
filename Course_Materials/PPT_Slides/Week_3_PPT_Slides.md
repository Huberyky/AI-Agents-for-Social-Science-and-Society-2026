# Week 3: LLMs Prompted for Multi-Agent Simulation
## AI Agents for Social Science and Society 2026
### Ignite-Style Slide Deck — 30 Slides

---

## Slide 1: Could You Simulate a Society?

**Visual Description:**
A dramatic overhead photograph of a real city — dense urban grid, thousands of ant-like figures on sidewalks. Over the image, in bold white text: "7.9 billion people. Infinite interactions. Emergent order."

Below, three questions in smaller white text:
- "What would you need to know?"
- "What would you need to build?"
- "What would you be willing to assume?"

**Bullet Points / Text:**
- Every social theory is a model of society
- Every model makes simplifying assumptions
- Question: what if those assumptions could talk back?

**Instructor Notes:**
Open with the provocation. Simulation has a long history in social science — from Schelling's segregation model to agent-based models of epidemics — but what is new is that the "agents" can now hold conversations, have memories, express beliefs, and adapt in real time. This week is about what becomes possible when we replace rule-based agents with language model-powered ones — and what new problems that creates.

---

## Slide 2: Review — The Tools We Now Have

**Visual Description:**
A visual "tool belt" illustration — a leather belt with labeled tools hanging from it:
- Hammer labeled "Deep networks (Week 1)"
- Wrench labeled "Text embeddings (Week 2)"
- Torch labeled "Transformer attention (Week 2)"
- Spark plug labeled "Pre-trained LLMs (Week 2)"
- Question mark labeled "???"

Caption: "This week: we assemble these tools into something that behaves like a person."

**Bullet Points / Text:**
- We can build networks that learn from data
- We can encode language into semantic geometry
- We have transformers pre-trained on human text
- Now: can we assemble a digital person?

**Instructor Notes:**
Brief bridge from the previous two weeks. Students now understand both the mechanics of neural networks (Week 1) and the architecture of transformers and LLMs (Week 2). This week is about the practical and conceptual question of what happens when you give these systems a persona and ask them to behave like social actors. The question mark in the tool belt is the simulation capability they are about to build.

---

## Slide 3: What Is a Digital Double?

**Visual Description:**
A split-screen illustration:
Left: a real person — stock photo of a middle-aged woman, labeled with demographic attributes: "Maria, 52, Chicago, Catholic, Democrat, retired teacher, follows local news, worries about crime."
Right: a terminal/code window showing a system prompt:
```
You are Maria, a 52-year-old retired teacher in Chicago.
You are Catholic and vote Democrat.
You worry about crime in your neighborhood.
You get most of your news from local TV.

When asked questions, respond as Maria would,
drawing on her background and concerns.
```

**Bullet Points / Text:**
- A digital double: an LLM configured to represent a person
- System prompt = demographic and ideological profile
- The double responds, reasons, and holds opinions — like Maria

**Instructor Notes:**
The "digital double" concept is central to this week and to Kozlowski & Evans (2025). The idea is conceptually simple: use an LLM's instruction-following capability to instantiate a specific demographic persona, then query that persona as you would a survey respondent. The hard questions are epistemological: when does the digital double represent Maria accurately? When is it just the LLM's stereotype of who Maria is?

---

## Slide 4: Prompting Strategies — Zero-Shot

**Visual Description:**
A clean diagram showing three prompt formats side by side:

Format 1 (Zero-shot):
```
Prompt:
"Will Maria vote for the Democratic candidate
 in the 2024 presidential election?"

Response:
"Yes, Maria will likely vote Democrat."
```
Caption: "No examples. Model uses only its pre-trained knowledge."

**Bullet Points / Text:**
- Zero-shot: no examples provided, just the question
- Model draws on pre-training to answer
- Baseline performance — often surprisingly good

**Instructor Notes:**
Zero-shot prompting is the most basic prompting strategy and surprisingly capable for many tasks because LLMs have absorbed so much contextual knowledge during pre-training. For social simulation, zero-shot is a useful baseline but tends to produce generic, stereotyped responses that reflect the model's average representation of a demographic group rather than individual-level variation.

---

## Slide 5: Prompting Strategies — Few-Shot

**Visual Description:**
A prompt example showing few-shot format:
```
Prompt:
"Here are examples of how Maria responds to political questions:

Q: Should the government raise the minimum wage?
A: Yes, working families are struggling. I've seen it.

Q: What do you think about immigration?
A: We need fair laws, but families shouldn't be separated.

Now answer: Will Maria vote for the Democratic candidate
in 2024?"

Response:
"Yes, and here's why — [nuanced, Maria-specific response]"
```
Annotation: "The examples calibrate the model to Maria's voice."

**Bullet Points / Text:**
- Few-shot: provide 2–5 examples before the question
- Examples calibrate the model to a specific voice or style
- More accurate, more consistent, more human-like

**Instructor Notes:**
Few-shot prompting dramatically improves both consistency and apparent authenticity. By providing examples of how the persona has responded to previous questions, you constrain the model's output distribution toward that persona's characteristic voice. In research applications, the examples can come from real survey responses, interview transcripts, or social media posts from representative individuals.

---

## Slide 6: Prompting Strategies — Chain-of-Thought for Agents

**Visual Description:**
A chain-of-thought prompt example for social simulation:
```
System: You are Maria, 52, Chicago, retired teacher, Democrat.

User: "Should the city increase the police budget?"

Maria thinks step by step:
  1. My neighborhood has had more robberies lately.
  2. But I've also seen police treat my neighbors badly.
  3. More money doesn't automatically mean more safety.
  4. I'd support targeted community policing, not just more officers.

Maria's answer: "It depends on how the money is spent..."
```

**Bullet Points / Text:**
- CoT for agents: the persona reasons before responding
- Reveals the internal logic of the digital double
- Makes beliefs auditable, not just outputs

**Instructor Notes:**
Chain-of-thought is especially valuable in social simulation because it makes the reasoning process visible and checkable. Rather than just getting a yes/no answer, you can see the considerations the agent weighed and verify that they are appropriate for the persona. This also allows you to inject contradictions or dilemmas and study how the agent resolves them — a form of computational attitude research.

---

## Slide 7: Actor-Critic Refinement Loops

**Visual Description:**
A diagram showing the actor-critic loop:
```
┌─────────────────────────────────────────────┐
│                                             │
│  ACTOR (Maria-LLM)                          │
│  Generates response to survey question      │
│         │                                   │
│         ▼                                   │
│  CRITIC (Evaluator-LLM)                     │
│  "Is this response consistent with          │
│   Maria's profile and prior responses?"     │
│         │                                   │
│         ▼                                   │
│  FEEDBACK: "Maria wouldn't say 'optimize' — │
│   she speaks more colloquially."            │
│         │                                   │
│         └──────── Revise response ──────────┘
```

**Bullet Points / Text:**
- Actor: generates the response
- Critic: evaluates persona consistency
- Loop refines responses until criteria are met

**Instructor Notes:**
The actor-critic pattern from reinforcement learning (which we'll formalize in Week 7) can be applied here as a prompting strategy. The critic LLM acts as a consistency checker, ensuring that the digital double's responses are coherent with its defined persona across multiple turns. This is particularly useful when you want to simulate survey panels — the same "person" responding to many questions consistently over time.

---

## Slide 8: Multi-Agent Architectures — AutoGen

**Visual Description:**
A diagram of AutoGen's ConversableAgent architecture:
```
User Proxy Agent
     │
     │ "Discuss the impact of minimum wage increases."
     ▼
Agent A (Economist)          Agent B (Labor Organizer)
"Research shows mixed..."  ◄──► "Workers need living wages..."
     │                               │
     └────────────┬──────────────────┘
                  ▼
         Agent C (Moderator)
         "Let me synthesize both views..."
                  │
                  ▼
            Final Report
```
Caption: "Agents talk to each other. Outcomes emerge from conversation."

**Bullet Points / Text:**
- AutoGen: Microsoft's multi-agent conversation framework
- ConversableAgent: any LLM that can send and receive messages
- Sequential and parallel agent graphs

**Instructor Notes:**
AutoGen provides a production-ready framework for building exactly these kinds of multi-agent conversations. The key abstraction is the ConversableAgent — any LLM that can participate in a conversation loop. Students will configure multiple agents with different personas, set up the conversation graph, and observe what emerges when "the economist" and "the labor organizer" debate minimum wage policy. Week 3's notebook walks through this in Module 3.

---

## Slide 9: Concordia — Situated Simulation

**Visual Description:**
A layered architecture diagram for Concordia (DeepMind's simulation framework):
```
┌──────────────────────────────────────────────────┐
│              GAME MASTER (LLM)                   │
│  Narrates world events, enforces rules, time     │
├──────────────────────────────────────────────────┤
│         SHARED SOCIAL ENVIRONMENT                │
│   (location, time, public events, history)       │
├────────────┬─────────────┬────────────┬──────────┤
│  Agent 1   │   Agent 2   │  Agent 3   │ Agent 4  │
│  [memory]  │   [memory]  │  [memory]  │ [memory] │
│  [goals]   │   [goals]   │  [goals]   │ [goals]  │
│  [history] │   [history] │  [history] │[history] │
└────────────┴─────────────┴────────────┴──────────┘
```

**Bullet Points / Text:**
- Game Master: the world's author and referee
- Agents: persistent memory + goals + formative experiences
- Institutional constraints: laws, norms, resources

**Instructor Notes:**
Concordia goes further than AutoGen by situating agents in an explicitly modeled social environment. The Game Master is an LLM that narrates what happens in the shared world and adjudicates agent actions — analogous to the Dungeon Master in a role-playing game. The key innovation is formative memories: agents have a history that shapes who they are, not just a system prompt that describes them.

---

## Slide 10: Memory in Agents — Why It Matters

**Visual Description:**
A comparison of two agent architectures:
Left (memoryless agent):
```
[Session 1] Maria answers about minimum wage.
[Session 2] Maria is asked the same question.
            She gives a completely different answer.
Status: NOT a digital double — just an LLM.
```
Right (agent with memory):
```
[Session 1] Maria argues for minimum wage increases.
            → stored in memory vector database

[Session 2] Maria is asked about fair pay.
            → retrieves session 1 memory
            → consistent response + reference to prior view
Status: Persistent persona — approaching a digital double.
```

**Bullet Points / Text:**
- Without memory: no identity, no consistency
- Memory types: episodic (events), semantic (facts), procedural (behaviors)
- Persistent identity = prerequisite for meaningful simulation

**Instructor Notes:**
Memory is what separates a digital double from a one-shot LLM query. Without it, the "same person" gives different answers every time they are asked — the opposite of what survey methodology requires. Different memory implementations have different properties: retrieved memories can be irrelevant or interfere with each other, memory capacity limits exist, and the retrieval mechanism itself introduces biases.

---

## Slide 11: RAG — Retrieval-Augmented Generation

**Visual Description:**
A five-step pipeline diagram:
```
Step 1: DOCUMENTS
  [News articles, speeches, survey responses, policies]
         ↓
Step 2: CHUNKING
  Split into 256-token overlapping segments
         ↓
Step 3: EMBEDDING
  Each chunk → vector via text-embedding-ada-002
         ↓
Step 4: VECTOR DATABASE (FAISS/ChromaDB)
  Index all chunk vectors for similarity search
         ↓
Step 5: RETRIEVAL + GENERATION
  Query → retrieve top-k chunks → inject into prompt → generate
```

**Bullet Points / Text:**
- RAG: give the agent a memory it can search
- Grounds responses in real documents (reduces hallucination)
- Scales to millions of documents via vector search

**Instructor Notes:**
RAG is one of the most practical and impactful techniques students will learn this week. Rather than relying solely on what the LLM has memorized during pre-training, RAG allows the agent to retrieve and ground its responses in specific documents. For social science simulation, this means an agent representing a specific community can be grounded in that community's actual discourse — news coverage, policy documents, local forum posts.

---

## Slide 12: FAISS — Fast Approximate Nearest Neighbors

**Visual Description:**
Visualization of FAISS vector search:
Left: a high-dimensional space (visualized as 2D for clarity) with hundreds of document chunk vectors plotted as dots. A query vector (marked with a star) has an expanding circle showing its nearest neighbors highlighted in yellow.

Right: code snippet:
```python
import faiss
import numpy as np

# Build index
d = 1536  # embedding dimension
index = faiss.IndexFlatL2(d)
index.add(chunk_embeddings)  # add all chunks

# Search
query_vec = embed(user_query)
distances, indices = index.search(
    query_vec.reshape(1,-1), k=5  # top 5 results
)
```

**Bullet Points / Text:**
- FAISS: Facebook AI Similarity Search
- Finds k nearest neighbors in milliseconds (millions of vectors)
- L2 or cosine distance; exact or approximate search

**Instructor Notes:**
FAISS is the industrial-strength library for semantic search at scale. Students will use it in Module 4 to build a document retrieval system for their agents. The key intuition is simple: embed both your documents and your query using the same embedding model, then find which documents are geometrically closest to the query in embedding space. This is faster and more semantically sensitive than keyword search.

---

## Slide 13: Electoral Simulation — The Week 3 Capstone

**Visual Description:**
A side-by-side visualization:
Left: a map of the United States shaded by predicted Biden/Trump vote share from digital double simulation (2020 election)
Right: the actual 2020 election result map
Correlation statistics overlaid: r = [0.XX] — actual vs. simulated, by state

Below: a demographic breakdown showing which groups the simulation matched well (educated suburban voters) vs. poorly (rural working-class Midwest).

**Bullet Points / Text:**
- 2020 election: 50 digital doubles per demographic cell
- Simulate survey responses: "Who will you vote for?"
- Validate against actual voting data by county/state

**Instructor Notes:**
This is the hands-on climax of Module 2. Students will build demographic personas for representative American voters, query them with a voting intention question, and aggregate the simulated votes to the state level to compare against actual 2020 results. The places where the simulation is most wrong — the cases where digital doubles diverge from human voters — are often the most interesting research findings.

---

## Slide 14: The Promise — Argyle et al. 2023 "Silicon Sampling"

**Visual Description:**
A visualization from Argyle et al. (2023) — two distribution plots side by side:
- Left: distribution of actual survey responses on a political attitude question, segmented by demographic group (young/old, White/Black, liberal/conservative)
- Right: distribution of LLM-simulated responses with the same demographic conditioning

The distributions look strikingly similar. Caption: "GPT-3 can approximate the distribution of human opinion — conditioned on demographics."

**Bullet Points / Text:**
- "Out of One, Many": one LLM, many personas
- Demographic conditioning produces realistic distributions
- Could replace expensive surveys for some applications?

**Instructor Notes:**
Argyle et al.'s "silicon sampling" paper is the optimistic pole of this week's debate. Their finding that GPT-3, when conditioned on demographic profiles, can approximately reproduce the distribution of survey opinion on many political questions was remarkable and controversial. The key qualifier is "approximately" — and much of the rest of this week is about how large and systematic the deviations are.

---

## Slide 15: The Peril — Kozlowski & Evans 2025

**Visual Description:**
A funhouse mirror visual: on the left, a normal "reality" image showing a diverse crowd of people; on the right, the same crowd as seen in a funhouse mirror — distorted, stretched, compressed, with some figures enlarged and others diminished.

Caption: "LLMs are not a mirror of society. They are a funhouse mirror. The distortions are not random — they are systematic."

Below: a list of documented distortions:
- Over-representation of college-educated liberal views
- Compression of within-group diversity
- Amplification of stereotypic group differences
- Instability of political attitudes under rephrasing

**Bullet Points / Text:**
- LLMs over-represent liberal, educated, Western views
- Within-group diversity is compressed
- Attitudes shift with trivial prompt changes

**Instructor Notes:**
Kozlowski & Evans (2025) is the required reading that pushes back most forcefully on the promise of silicon subjects. The key finding is that LLMs do not sample from the actual distribution of human opinion — they produce a biased, smoothed, stereotype-inflated version of it. The funhouse mirror metaphor is crucial: the distortions are not random noise but systematic biases that will affect any downstream research conclusions.

---

## Slide 16: When Do LLMs Diverge From Humans?

**Visual Description:**
A scatter plot where:
- X-axis: "How well LLM matches human survey responses" (0 to 1 correlation)
- Y-axis: "Demographic group" — a list of ~15 groups from top to bottom: "White college-educated liberal" (near 1.0), "Black conservative" (low), "Rural evangelical" (low), "Hispanic moderate" (medium), etc.
- Points colored by region (urban/suburban/rural)

Caption: "The LLM works best as a 'digital double' of the people most represented in its training data."

**Bullet Points / Text:**
- Best: educated, urban, majority-group respondents
- Worst: minority, rural, non-Western, older respondents
- Training data determines who the LLM "knows"

**Instructor Notes:**
This slide makes the validation challenge concrete. The LLM is essentially modeling the people who wrote the most text on the internet and in published books — disproportionately educated, Western, liberal, and English-speaking. Any simulation study using LLM digital doubles will systematically underperform for groups underrepresented in that training corpus. Validation against real data is not optional; it is mandatory.

---

## Slide 17: Emergent Polarization in Multi-Agent Systems

**Visual Description:**
A time-lapse diagram showing a 20-agent network's opinion distribution over 100 rounds of conversation:
- Time 0: agents distributed evenly across a 5-point opinion scale
- Time 10: slight bimodal tendency emerging
- Time 50: clear bimodal distribution with most agents at extreme poles
- Time 100: near-perfect polarization — two camps

Below: a network graph showing which agents talked to which, with edge colors showing agreement (blue) vs. disagreement (red). Over time, agreement edges dominate (homophily emerges).

**Bullet Points / Text:**
- Homophily: agents prefer talking to those who agree
- Echo chambers emerge without explicit programming
- Polarization can be faster and more complete than in human experiments

**Instructor Notes:**
This finding — that multi-agent LLM simulations tend to polarize rapidly — has been replicated across several research groups and is theoretically important. It may reflect genuine dynamics of opinion formation (social influence + homophily → polarization) or it may reflect artifacts of how LLMs handle disagreement (they tend to accommodate the views of their interlocutor). Distinguishing these explanations is itself a research question.

---

## Slide 18: In Silico Sociology — COVID Polarization Study

**Visual Description:**
A study design diagram showing the Kozlowski, Kwon, Evans "In Silico Sociology" setup:
- Left: demographic profiles of actual American respondents to a COVID attitudes survey
- Center: digital doubles created from those profiles, asked about mask mandates, vaccination, government response
- Right: comparison of LLM responses to actual survey responses, showing regions of agreement and divergence

A particularly striking finding highlighted: the simulation correctly predicts the partisan divide on mask mandates but underestimates the degree of agreement on economic concerns.

**Bullet Points / Text:**
- Simulate COVID policy attitudes from demographic profiles
- Partisan divides correctly reproduced
- Economic anxiety: underestimated by simulation
- Who the LLM "misses" = substantive finding

**Instructor Notes:**
This study is an excellent example of what LLM simulation can and cannot do. The model successfully captures large-scale partisan patterns — the things that are already well-documented in the training data. What it misses is the cross-cutting economic anxiety that does not map cleanly onto the political identities the model knows. The errors are theoretically informative: they suggest where LLM training data is least representative of lived experience.

---

## Slide 19: The System Prompt — The DNA of Your Agent

**Visual Description:**
A detailed annotated system prompt example:
```
You are [NAME: James Whitfield],
a [AGE: 68-year-old] [OCCUPATION: retired coal miner]
living in [LOCATION: McDowell County, West Virginia].

You have a [EDUCATION: high school diploma].
Your family has [HISTORY: lived in coal country for 4 generations].
You feel [IDENTITY: left behind by the coastal economy].
You voted [POLITICAL: Republican in 2016 and 2020].
You [RELIGION: attend an evangelical Baptist church weekly].

Your primary concerns are: [CONCERNS: jobs, drug crisis,
decline of community, distrust of federal government].

When responding, stay in character. Use language natural
to your background. Share uncertainty when uncertain.
```
Each bracketed segment color-coded as a "gene" in the persona's DNA.

**Bullet Points / Text:**
- System prompt = the agent's identity architecture
- Every attribute shapes responses — and their interactions matter
- What you leave out is as important as what you include

**Instructor Notes:**
The system prompt is the researcher's primary instrument in simulation-based social science, and it deserves the same methodological attention as a survey instrument. Which attributes do you specify? How do you describe them? What level of granularity is appropriate? Too sparse and the agent collapses to a stereotype; too detailed and you are inventing a person rather than representing a type. These are empirical questions that require validation.

---

## Slide 20: Self-Reflection Loops

**Visual Description:**
An agent processing diagram showing a self-reflection architecture:
```
Round 1: Agent answers question
  → stores response in memory

Round 2: Agent is shown its own answer
  → "Does this response reflect your actual views?"
  → "What did you leave unsaid?"
  → Reflection stored as new memory entry

Round 3: Agent is asked a follow-up
  → draws on both original response AND reflection
  → gives a more nuanced, consistent answer
```
Caption: "Reflection improves consistency and depth — but also risk of confabulation."

**Bullet Points / Text:**
- Self-reflection: the agent evaluates its own outputs
- Improves consistency across multi-turn conversations
- Can generate post-hoc rationalizations (risk!)

**Instructor Notes:**
Self-reflection loops are borrowed from the psychological literature on introspection and are particularly useful for qualitative simulation — when you want the agent to reason about its own motivations and beliefs. The caution is that LLMs are very good at generating plausible-sounding rationalizations that bear no necessary relationship to the actual computations that produced the original response. This is the computational analog of why human introspection is often inaccurate.

---

## Slide 21: RAG in Practice — Grounding Digital Doubles

**Visual Description:**
A before/after comparison:
Left (without RAG):
```
User: "What does Maria think about the recent
       city council decision on bus fares?"
Maria: "Bus fare increases affect working
        families who depend on public transit..."
[Generic response — could be anyone]
```
Right (with RAG):
```
[Retrieved: Chicago Tribune, Jan 8, 2026:
 "City Council votes 32-18 to raise CTA fares by $0.50"]

User: Same question
Maria: "I read about the 32-18 vote last week.
        That $0.50 is real money when you ride
        every day. The 18 who voted against it
        were right..."
[Grounded in specific, real event]
```

**Bullet Points / Text:**
- RAG: injects real documents into the agent's context
- Grounds responses in specific, verifiable facts
- Reduces hallucination; increases external validity

**Instructor Notes:**
This comparison makes the value of RAG concrete. Without it, the digital double speaks in generalizations that could apply to anyone — high internal consistency but poor external validity. With RAG, the agent is responding to actual events in the world, and you can compare its responses to how real community members reacted to those same events. This dramatically improves the research validity of the simulation.

---

## Slide 22: Liebo et al. 2025 — Norms from Pattern Completion

**Visual Description:**
A diagram illustrating the paper's core argument:
Left box: "LLMs predict the most statistically likely completion of a text sequence."
Right box: "Social norms are enforced by the statistical expectation that others will follow them."

Arrow connecting them: "Both are about what is expected. LLMs may encode norms as prediction targets."

Below: an example showing a norm-following prompt:
```
"At a professional dinner, when the host raises their glass,
the guests typically..."
→ LLM: "...raise their glasses and wait for the toast."
[Correct normative behavior — or just the statistically likely text?]
```

**Bullet Points / Text:**
- LLMs learn statistical regularities — including norms
- Norms = expected behavior = high-probability text continuations
- Social regularity and statistical regularity overlap

**Instructor Notes:**
Liebo et al. (2025) make a theoretically interesting argument: that LLMs may encode social norms not because they understand them but because normative behavior is statistically prevalent in training text. This has two implications: LLMs can be useful for norm detection research (what do they predict as expected behavior?) and LLMs will reinforce existing norms rather than challenging them (they predict what is expected, not what is optimal or just).

---

## Slide 23: Xie et al. PNAS 2025 — Strategic Situations

**Visual Description:**
A game theory setup visualization — a 2x2 payoff matrix (Prisoner's Dilemma style) with the LLM's responses for each cell:
```
           Partner cooperates | Partner defects
You cooperate:  Both get 3    |   You get 0
                              |   Partner gets 5
You defect:     You get 5     |   Both get 1
                              |   Partner gets 0
```
Below: a bar chart showing how different LLM models respond to different game-theoretic framings (names changed to "helping" vs. "competing"), demonstrating sensitivity to framing.

**Bullet Points / Text:**
- LLMs can categorize and respond to strategic situations
- But framing effects are large: same game, different names → different choices
- Social norms encoded in game descriptions drive behavior

**Instructor Notes:**
Xie et al. (2025) demonstrate that LLMs can be used to simulate strategic social interactions — cooperation, defection, negotiation — with notable accuracy. But the critical finding is the framing sensitivity: the same payoff matrix described as "helping your community" vs. "outcompeting your rival" produces systematically different choices. This means that simulation results in strategic settings are highly sensitive to the precise language used to describe the scenario.

---

## Slide 24: Building the Week 3 Notebook — Module Overview

**Visual Description:**
A visual roadmap of the four notebook modules, shown as a map with waypoints:
```
START ──► MODULE 1: LLM Prompting
          (zero-shot / few-shot / CoT / actor-critic
           on election dataset)
             │
             ▼
          MODULE 2: Digital Doubles
          (build 10+ personas, simulate
           survey + open-ended responses)
             │
             ▼
          MODULE 3: AutoGen Multi-Agent
          (conversations, self-reflection,
           research collaborations)
             │
             ▼
          MODULE 4: RAG
          (vector store, retrieval pipeline,
           grounded agent) ──► FINAL PROJECT
```

**Bullet Points / Text:**
- Complete 3 of 4 modules
- Validate simulation accuracy against ground truth
- One module applied to your final project domain

**Instructor Notes:**
Walk through the module structure now and let students choose their path. Module 1 is the most accessible entry point for students less comfortable with LLMs; Module 4 (RAG) is the most technically demanding but also the most directly useful for research applications. All students should complete Module 2 (digital doubles) as it is central to this week's theme.

---

## Slide 25: Validation — The Hardest Part

**Visual Description:**
A validation framework diagram:
```
SIMULATION OUTPUT: Digital doubles' survey responses
         │
         ├──► Level 1: Face validity
         │    "Do responses sound like real people?"
         │    [Human judge rating: plausibility]
         │
         ├──► Level 2: Criterion validity
         │    "Do aggregate distributions match real surveys?"
         │    [Compare to ANES, Pew, GSS data]
         │
         ├──► Level 3: Predictive validity
         │    "Do simulated choices match actual outcomes?"
         │    [Election prediction vs. actual results]
         │
         └──► Level 4: Discriminant validity
              "Do different personas give different answers?"
              [Measure variance across demographic groups]
```

**Bullet Points / Text:**
- Validation is not optional — it is the science
- Face validity alone is insufficient and misleading
- Ground truth: real surveys, election results, behavioral data

**Instructor Notes:**
Students must validate their simulations, and this framework gives them a structured vocabulary for doing so. Face validity — "it sounds right" — is the most dangerous form of validation because LLMs are exceptionally good at producing plausible-sounding text that is systematically wrong. Always push to levels 2 and 3: do the distributions match real data? Do the predictions match real outcomes? These are the tests that matter.

---

## Slide 26: What LLMs Add to Social Simulation

**Visual Description:**
A comparison table between traditional agent-based models (ABMs) and LLM-powered agents:
```
Feature            | ABM                  | LLM Agent
───────────────────+──────────────────────+────────────────────
Agent behavior     | Hand-coded rules     | Natural language
Belief updating    | Bayesian/explicit    | In-context learning
Communication      | Signal/state         | Full conversation
Diversity          | Parameter variation  | Prompt variation
Validation         | Statistical tests    | Corpus comparison
Emergence          | Rule interactions    | Prompt interactions
Failure modes      | Known, bounded       | Unknown, unbounded
Interpretability   | Full                 | Partial (attention)
```

**Bullet Points / Text:**
- LLMs: richer behavior, less controlled, less interpretable
- ABMs: full control, less realistic behavior
- Hybrid approaches: the future of computational social science

**Instructor Notes:**
The comparison to traditional agent-based modeling is important for students with social science backgrounds. ABMs like NetLogo or Mesa give researchers complete control over agent rules and interactions, making the behavior fully interpretable and reproducible. LLM agents sacrifice this control for behavioral richness — the conversations are richer, but the reasons for specific outputs are harder to inspect. Neither approach is better; they answer different questions.

---

## Slide 27: The Ethics of Simulation — Creating People Without Consent

**Visual Description:**
A philosophical dilemma visual: two figures facing each other across a chasm.
Left figure: Researcher with notebook, labeled "Scientific value of simulation"
Right figure: Stylized person labeled "The simulated person who cannot consent"
The chasm between them is labeled: "Representation without permission."

Below: three questions:
1. "Is it ethical to simulate a person's opinions?"
2. "Who owns the digital double?"
3. "What if the simulation is used to target them?"

**Bullet Points / Text:**
- Digital doubles represent real demographic groups
- No consent mechanism exists for population-level simulation
- Potential harms: manipulation, stigmatization, impersonation

**Instructor Notes:**
This is a genuinely unresolved ethical question in the field. Unlike surveys, simulation studies do not require IRB approval, do not obtain informed consent, and do not offer participants any right to see or correct their digital representation. Yet the outputs are used to make claims about how those people think and behave. Students should engage with this seriously — it will be relevant to their own research projects.

---

## Slide 28: Silicon Subjects in Peer Review — Emerging Norms

**Visual Description:**
A split panel showing journal and conference stances on LLM simulation:
- Left: Title cards of papers using LLM simulation published in top journals: PNAS, Nature Human Behaviour, American Sociological Review, APSR
- Center: A scale showing "Promise" vs. "Skepticism" at opposite ends
- Right: Sample reviewer comment: "How do the authors demonstrate that the LLM simulation produces valid inferences about human behavior rather than inferences about LLM behavior?"

Caption: "The field is still developing norms. Your validation approach will be scrutinized."

**Bullet Points / Text:**
- LLM simulation is entering mainstream social science
- Methodological standards are still being established
- Transparency, validation, and limitations disclosure are essential

**Instructor Notes:**
The norms for LLM simulation as a social science method are actively being negotiated right now, in 2026. Students who can articulate clear validation strategies and honest limitations will be ahead of the curve both in publication and in research credibility. The peer review comment shown here is representative of the skepticism that the field still brings to these methods — and it is a healthy skepticism.

---

## Slide 29: Building a Research Pipeline with Digital Doubles

**Visual Description:**
A complete research pipeline diagram for an LLM simulation study:
```
Step 1: RESEARCH QUESTION
  "How do different demographic groups respond
   to AI-generated vs. human-written news?"

Step 2: SAMPLE DESIGN
  Define 20 demographic profiles (age × education × ideology)

Step 3: PERSONA CONSTRUCTION
  Write system prompts; validate with pilot queries

Step 4: STIMULI DESIGN
  Prepare AI-written and human-written news articles

Step 5: DATA COLLECTION
  Query each persona with each stimulus (N=400 responses)

Step 6: VALIDATION
  Compare to real survey data where available

Step 7: ANALYSIS + INTERPRETATION
  HTE models (see Week 4); qualitative coding of reasoning
```

**Bullet Points / Text:**
- Treat simulation like a survey: design matters
- Document every prompt decision (like survey instrument)
- Pre-register when possible; validate always

**Instructor Notes:**
This pipeline can serve as a template for students designing their own simulation-based final projects. The key methodological discipline is to treat the prompt as an instrument — every word in a system prompt is a design choice with consequences for the output, just as every question wording in a survey has consequences for responses. Documentation, pre-registration, and validation are not bureaucratic hurdles; they are what separates research from experimentation.

---

## Slide 30: The Funhouse Mirror — What We Learned and Why It Matters

**Visual Description:**
The funhouse mirror image returns from Slide 15, but now the distortions are labeled:
- Elongated figure labeled "Over-represented: educated liberal views"
- Compressed figure labeled "Under-represented: rural conservative views"
- Blurred figure labeled "Lost: within-group diversity"
- Extra-bright figure labeled "Amplified: stereotypic associations"

Caption: "Multi-agent LLM systems are not a mirror of society. Understanding the distortions is as important as the simulation."

**Bullet Points / Text:**
- Promise: rich, scalable, interactive social simulation
- Peril: systematic bias, stereotype amplification, false validity
- Obligation: validate, disclose, question the model's assumptions

**Instructor Notes:**
Close with the course's second sustained provocation. Simulation is not a shortcut around difficult social science — it is a method that comes with its own systematic limitations. The researcher's job is not to pretend these limitations don't exist but to understand them deeply enough to design around them, report them honestly, and build the kind of institutional knowledge that allows the field to use these tools responsibly. Next week, we add causal inference to the simulation toolkit.

---

*End of Week 3 Slide Deck — 30 Slides*
*AI Agents for Social Science and Society 2026*
*Instructor: James A. Evans | January 23, 2026*
