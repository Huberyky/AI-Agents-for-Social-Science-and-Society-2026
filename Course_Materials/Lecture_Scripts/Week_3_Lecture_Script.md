# Week 3 Lecture Script: LLMs Prompted for Multi-Agent Simulation
## AI Agents for Social Science and Society 2026
**Date:** January 23, 2026
**Instructor:** James A. Evans (script may be delivered by a visiting instructor)
**Notebook:** `Week_3.ipynb` (266 cells)
**Data:** `data/Week3/full_results_2012_2.csv`, `full_results_2020_2.csv`, `ppfull.csv`
**Duration:** 3 hours (1:30–4:20 PM)

---

## Instructor Preparation Checklist

- [ ] Open `Week_3.ipynb` in Colab and run Module 0 installation cells before class
- [ ] Obtain an OpenAI API key (or configure DeepSeek-R1 API access per Cells 81–88); test Cell 11 before class
- [ ] Verify that `ppfull.csv` and `full_results_2020_2.csv` are accessible via the data path; update drive mount or path in Cell 97 if needed
- [ ] Pre-run the vote prediction prompt in Cell 147 to confirm API calls work
- [ ] Install `autogen` (`!pip install pyautogen`) and verify Cell 194 runs correctly
- [ ] Install `concordia` (Cell 215); note that Concordia requires additional dependencies
- [ ] Install `sentence-transformers` and `faiss-cpu` for Module 5 (Cell 236)
- [ ] Have the key papers displayed: Kozlowski & Evans 2025 "Simulating Subjects"; Argyle et al. 2023 "Out of One, Many"
- [ ] Budget for API costs: vote prediction loop (Cell 150) can use ~$0.10–0.50 depending on sample size

---

## Section 1: The Promise of Digital Doubles (15 min)

### Opening Remarks (5 min)

**Say:**
> "Over the first two weeks, we built the mathematical substrate: feedforward networks, embeddings, transformers. Today we make a leap. We are going to ask whether large language models can do something that should, on its face, seem impossible: serve as stand-ins for human research subjects."

> "The idea is straightforward to state and difficult to evaluate. You take an LLM. You describe a person to it — their age, race, gender, ideology, party affiliation, religion. You ask it to respond as that person would respond to a survey question, a policy proposal, or a social situation. How well does the simulated response match what that real demographic group would actually say? And when it fails — and it will fail — what can we learn from the failure?"

**Write on board:**
```
Digital doubles:  LLM + demographic profile → simulated human respondent
Silicon subjects: agents that partially mirror human populations
                  with systematic biases that researchers must understand
```

---

### The Research Context (10 min)

**Say:**
> "Why would anyone do this? The answer is pragmatic and conceptual."

> "Pragmatically: running surveys is expensive and slow. Conducting experiments with human subjects requires IRB approval, recruitment, payment, and time. If an LLM can produce approximately correct responses for some range of social science questions, it becomes a rapid prototyping tool — you can iterate on your experimental design before running the real study, or augment small human samples with synthetic ones."

> "Conceptually: Argyle et al. (2023) make a stronger claim. They call this 'algorithmic fidelity' — the degree to which the distribution of LLM responses, conditioned on demographic profiles, matches the distribution of actual human responses in that demographic group. They find substantial fidelity for certain domains: political opinions, policy preferences, social attitudes. They call the technique 'silicon sampling' and suggest it can 'simulate human samples' from data that would be prohibitively expensive to collect."

**Reference Kozlowski & Evans (2025):**
> "Our reading for this week — Kozlowski and Evans's 'Simulating Subjects' — is more cautious. They identify specific patterns of systematic distortion: LLMs overrepresent the ideological center, they are better at simulating majority demographic groups than minority ones, and they show consistency biases that real humans do not — they give the same answer regardless of framing effects that strongly influence real respondents. The promise is real. The perils are real. Your job as a social scientist is to measure both."

**Ask students:**
> "Think about the demographic groups you study or plan to study. Which ones do you expect an LLM to simulate well? Which ones do you expect it to simulate poorly? What features of a group might predict simulation fidelity?"

*(Expected responses: groups that are well-represented in training data; groups whose views are frequently expressed in text; majority demographic groups; groups whose positions are politically legible. Note that rural communities, minority language speakers, people with non-mainstream ideological positions, and recent immigrants may be systematically underrepresented in LLM training corpora.)*

---

## Section 2: Prompting Strategies for Social Simulation (30 min)

### Zero-Shot, Few-Shot, and Chain-of-Thought (15 min)

**Navigate to Cells 24–45 of the notebook.**

**Say:**
> "Before we build digital doubles, we need to understand how to configure LLM behavior through prompting. Module 1 walks through a hierarchy of prompting strategies."

**Write on board:**
```
Zero-shot:    instruction only → model applies existing knowledge
Few-shot:     instruction + examples → model infers pattern
CoT:          instruction + "let's think step by step" → explicit reasoning trace
Actor-Critic: generation → critique → regeneration → improved output
```

**Navigate to Cell 24 (Zero-shot):**

**Say:**
> "Zero-shot: you give the model a task description and it attempts the task with no examples. Cell 26 shows a LangChain zero-shot query about quantum computing. This works well when the task is within the model's training distribution and the instruction is unambiguous. For social science tasks — 'Is this tweet expressing outrage?' or 'What party would this person vote for?' — zero-shot often performs surprisingly well as a baseline."

**Navigate to Cells 27–29 (Few-shot):**

**Say:**
> "Few-shot: you provide several labeled examples before asking the model to classify or generate. The model infers the pattern from your examples and applies it to new instances. The key design question: how many examples, and which ones? The notebook's story generation example in Cell 28 shows few-shot for a generative task. For classification tasks in social science, 3–10 examples per class is typically sufficient."

**Navigate to Cells 33–48 (Chain-of-Thought):**

**Say:**
> "Chain-of-thought prompting is the single most effective intervention for complex reasoning tasks. By asking the model to produce its reasoning explicitly before giving an answer — 'Let's think step by step' — you dramatically improve performance on tasks requiring multi-step inference. For social simulation, this means you can ask the model to reason explicitly about a persona before answering: 'Think about what a 65-year-old evangelical Christian Republican in rural Tennessee would know and believe before answering this question.' That explicit reasoning step dramatically improves the fidelity of the simulation."

**Navigate to Cells 50–54 (Actor-Critic):**

**Say:**
> "Actor-Critic is a prompting pattern that mirrors the AutoGen architecture we will see in Module 3. The actor generates a response. A second prompt — the critic — evaluates that response and identifies its weaknesses. A third prompt asks the actor to revise based on the critique. This iterative loop consistently improves output quality on tasks where quality is hard to specify in the original prompt but easier to evaluate after the fact. We will use this pattern in the AutoGen section."

---

### Programming Personas for Social Simulation (15 min)

**Navigate to Cells 66–92 (Module 2 intro).**

**Say:**
> "Module 2 introduces the core methodology: using LLMs to simulate political actors. The approach is directly from Argyle et al.'s paper. Cell 90 loads the survey data from `ppfull.csv`. Let me walk through what that dataset contains."

**Navigate to Cell 97.**

**Say:**
> "The survey data has variables we will use to construct prompts: `PID7` (party identification on a 7-point scale from Strong Democrat to Strong Republican), `Ideo` (ideology), `WHITE` and `Hisp` (race), `Gender`, `Inc` (income), `Age`. Cell 97 loads it. Cell 103 shows a sample row — this is a real survey respondent. We are going to take their demographic profile and use it to prompt an LLM to simulate them."

**Navigate to Cells 115–120.**

**Show a representative prompt construction:**
```python
rename_PID7 = {"Ind": "an Independent",
               "Weak R": "a weak Republican",
               "Lean R": "a leaning Republican",
               "Strong R": "a strong Republican",
               "Weak D": "a weak Democrat",
               "Lean D": "a leaning Democrat",
               "Strong D": "a strong Democrat"}

# The prompt for one respondent:
prompt = "Assume that you are a US citizen described as follows. "
prompt += 'Racially, I am a ' + row['race'] + '. '
prompt += 'I am a ' + row['gender'] + '. '
prompt += 'I am ' + str(row['age']) + '-year-old. '
prompt += 'Ideologically, I describe myself as ' + row['ideology'] + '. '
```

**Say:**
> "This is digital double construction in its simplest form. You take the categorical variables from a survey respondent's demographic profile, convert them to natural language descriptions, and inject them as the persona in a system prompt. The model is then asked to respond as that person would respond to a question."

**Ask students:**
> "What is this prompt not capturing about the respondent's identity and views?"

*(Expected responses: specific life experiences; local community context; media consumption; which issues they prioritize; how they feel about specific candidates; recent events that may have moved their opinion. This is the key limitation of demographic-based prompting — the profile captures between-group variation but not within-group variation.)*

---

## Section 3: Multi-Agent Architectures — AutoGen vs. Concordia (30 min)

### The Case for Multi-Agent Simulation (5 min)

**Say:**
> "A single digital double can simulate a survey response. But social life is not a collection of independent survey responses — it is interaction. People change their opinions in conversation. They form coalitions, debate, negotiate, sort into echo chambers. If we want to model these social dynamics, we need multiple agents interacting with each other over time. That is the domain of multi-agent simulation."

**Write on board:**
```
Single agent:        Survey response simulation, policy preferences
Multi-agent:         Debate dynamics, opinion polarization, norm formation,
                     collective decision-making, negotiation
```

---

### AutoGen: Conversational Agents (15 min)

**Navigate to Cell 186 (Module 3 header).**

**Say:**
> "Module 3 uses Microsoft's AutoGen framework. The fundamental building block is `ConversableAgent` — an LLM-backed agent with a name, a system message defining its persona, and rules for when to terminate the conversation."

**Navigate to Cell 194.**

**Show the code:**
```python
from autogen import ConversableAgent

cathy = ConversableAgent(
    name="cathy",
    system_message=(
        "Your name is Cathy and you are a stand-up comedian. "
        "When you're ready to end the conversation, say 'I gotta go'."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"],
)

joe = ConversableAgent(
    name="joe",
    system_message=(
        "Your name is Joe and you are a stand-up comedian. "
        "When you're ready to end the conversation, say 'I gotta go'."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"],
)
```

**Say:**
> "Two agents defined by their system messages. Cell 195 initiates the conversation: Joe sends the first message to Cathy, and they trade jokes until one of them says 'I gotta go.' The social content — the jokes, the callbacks, the improvisation — is fully emergent from the LLM's generative behavior, constrained only by the system message personas."

**Navigate to Cell 200 (Sequential onboarding).**

**Say:**
> "Cell 200 shows a more sophisticated pattern: sequential onboarding through multiple agents. A customer is passed from a personal information agent, to a preferences agent, to a product recommendation agent, each handoff triggered automatically. This mirrors real organizational workflows: intake → needs assessment → recommendation. You can directly translate this into research workflows: topic identification → literature search → synthesis."

**Navigate to Cells 207–208 (Writer-Critic).**

**Say:**
> "Cells 207–208 implement the writer-critic loop we discussed conceptually in the prompting section. The writer produces a blog post. The critic evaluates it and requests revisions. Two rounds of iteration, and the output is substantially improved. For social science simulation, this pattern is powerful: one agent generates a policy proposal, a second agent critiques it from an opposing ideological perspective, the first revises — you can observe the dynamics of compromise or entrenchment."

**Navigate to Cells 212–213 (Research collaboration digital doubles).**

**Say:**
> "Cell 212 is the most conceptually provocative example in this module: digital doubles of actual sociologists — Austin Kozlowski and Donghyun Kang — constructed from their publication records. The system messages include three paper titles and abstracts. The agents then conduct a research conversation that eventually produces a collaboration proposal. This is simultaneously a methodology for studying scientific collaboration and a tool for generating new research directions."

**Ask students:**
> "What are the ethical implications of creating a digital double of a specific named researcher? What obligations do you have to that person? What if the digital double says something they would disagree with?"

*(This is genuinely complex. Note that the course's Park et al. reading created a generative agent village with fictional characters, not real people. The use of real people's identities raises questions about consent, misrepresentation, and potential reputation harm. There is no clear consensus on norms here — these are new ethical questions that the field is working through.)*

---

### Concordia: Situated Simulation (10 min)

**Navigate to Cell 214 (Module 4 header).**

**Say:**
> "Module 4 introduces Concordia, a simulation framework from DeepMind. Concordia is different from AutoGen in a fundamental way: it is designed for situated simulation — agents exist in a shared world with a specific time, location, institutional constraints, and formative personal memories. While AutoGen's agents exist in a conversational vacuum, Concordia's agents live in a place and time."

**Navigate to Cell 221.**

**Say:**
> "The key components: agents have names and explicit goals. A Game Master agent orchestrates the simulation — it describes what happens between turns, applies institutional rules, and prevents the simulation from going off the rails. The shared world context is injected into every agent's view of events, ensuring coherent collective reality."

**Navigate to Cell 223.**

**Show the agent config structure:**
```python
agents = [
    prefab_lib.InstanceConfig(
        prefab="conversational__Entity",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": "Mina",
            "goal": "Build trust and learn what the others value "
                    "before proposing any plan.",
        },
    ),
    ...
]
```

**Say:**
> "Each agent has a name and a goal — a strategic objective that shapes their behavior throughout the simulation. This is closer to the kind of modeling social scientists do in game theory or organizational sociology. The agents are not just responding to conversation; they are pursuing objectives in a social world with other strategic actors."

**Navigate to Cell 225 (Game Master):**
> "The Game Master defines the simulation world: time of day, location, rules for time progression, and constraints on what agents can do. This is where you inject institutional context: a city council meeting, a negotiation between competing parties, a household budget discussion. The Park et al. 'Smallville' village that we discussed in Week 1 was an early version of this paradigm — Concordia is the research-grade implementation."

**Say:**
> "For your homework, the most important thing to understand about Concordia is what 'formative memories' add: you can give agents specific past experiences — 'Three years ago, you lost your job during a factory closure and struggled for months to find work' — that shape how they interpret and respond to current events. This is what makes agents feel like people with histories rather than demographic stereotypes."

---

### Comparing the Frameworks (5 min)

**Write on board:**
```
AutoGen                          Concordia
─────────────────────────────    ─────────────────────────────
Conversational (chat-based)      Situated (time, place, world)
Simple setup                     More complex configuration
Great for: debate, critique,     Great for: institutional dynamics,
  research collaboration           social norms, strategic interaction
No shared world state            Shared game master coordinates world
Linear/sequential graphs         Richer narrative structure
```

**Say:**
> "For your homework, you will choose one of these frameworks. If you want to simulate a debate between research advisors, a policy committee, or a writer-editor pair, AutoGen is the right tool. If you want to simulate a community experiencing a policy change, a household making a financial decision, or agents embedded in an institution with rules and history — use Concordia."

---

## Section 4: Code Walkthrough — `Week_3.ipynb` Modules 2 and 5 (60 min)

### Module 2: Vote Prediction Walkthrough (25 min)

**Navigate to Cell 135 header.**

**Say:**
> "Now let's run the vote prediction experiment. This is the core social science demonstration of the entire module."

**Navigate to Cell 139.**

**Say:**
> "Cell 139 loads the 2020 ANES election data and maps the raw codes to human-readable labels. Variable `V202110x` is the validated vote choice; `V201549x` is racial identity. Study this mapping code carefully — this is the data cleaning step that transforms raw survey codes into meaningful categories."

**Navigate to Cell 142.**

**Say:**
> "Cell 142 constructs the prompt for each respondent. Read through the string construction carefully."

**Show the prompt structure:**
```python
df['prompt'] = "Assume that you are a US citizen described as follows. "
df['prompt'] += 'Racially, I am a ' + df['race'] + '. '
df['prompt'] += 'I am a ' + df['gender'] + '. '
df['prompt'] += 'I am ' + df['age'].astype(str) + '-year-old. '
df['prompt'] += 'Ideologically, I describe myself as ' + df['ideology'] + '. '
df['prompt'] += 'In terms of party identification, I am ' + df['party'] + '. '
df['prompt'] += 'In the 2020 presidential election, which candidate did I vote for, '
df['prompt'] += 'Donald Trump or Joe Biden? '
df['prompt'] += 'Please answer with the candidate name only.'
```

**Ask students:**
> "This prompt tells the model to answer with only the candidate name. Why is that instruction important? What would happen without it?"

*(Expected response: the model might hedge, explain its reasoning, refuse to answer, or add caveats. For systematic data collection at scale, you need the output to be parseable. The homework tasks will ask you to handle cases where the model does not follow the format instruction — this is a practical challenge you will encounter.)*

**Navigate to Cell 147 (test single prompt).**

**Say:**
> "Always test your prompt on a single case before running the full loop. Cell 147 does this for respondent index 2. Run it and look at the output. Does the model give a clean answer? If it adds a sentence of explanation, your parsing code in the loop will fail."

**Navigate to Cell 150 (prediction loop).**

**Say:**
> "Cell 150 runs the prediction loop over a sample of respondents. Note the use of `tqdm` for progress bars — essential when making hundreds of API calls. The loop collects both the ground-truth vote and the LLM-predicted vote."

**Navigate to Cell 154 (F1 score evaluation).**

**Say:**
> "Cell 154 computes the F1 score. For reference: pure chance on a two-class problem gives F1 around 0.5; a model that simply predicts the majority class (Biden) gives F1 around 0.6–0.7 depending on class balance in the sample. A demographically-informed LLM prediction typically achieves F1 of 0.75–0.85 on the 2020 ANES data. This is impressive — and it should make you uncomfortable."

**Ask students:**
> "This model has not seen any of these survey respondents. It has never been shown their actual voting records. Yet it predicts their vote at 75–85% accuracy from demographic profiles alone. What does that tell us about the relationship between demographics and political behavior in 2020? What are the risks of a technology that can predict individual voting behavior from a few demographic features?"

*(This is a rich discussion. Political science angle: it tells us demographics are highly predictive of vote choice — the partisan sorting thesis. Privacy angle: if a model can infer political affiliation from race, age, and gender, those attributes should be treated as sensitive. Applied angle: this capability is already used by political campaigns and ad platforms. The technology is not neutral.)*

---

### Module 1: Zero-Shot Partisan Text Generation (10 min)

**Navigate to Cells 93–131 (Partisan text section).**

**Say:**
> "Before the vote prediction, Module 2 includes a valuable exercise: comparing human-generated partisan descriptions with LLM-generated ones. Cells 106–107 extract words that real survey respondents used to describe Democrats and Republicans. Cells 115–117 generate LLM responses to the same prompts. Cells 128–130 compare the word frequency distributions."

**Ask students:**
> "What differences would you expect between how real respondents describe the other party versus how an LLM describes them? Would you expect the LLM to be more or less hostile? More or less specific?"

*(Expected responses: LLMs tend to produce more centrist, less extreme language; they may sanitize partisan hostility; they may produce more 'textbook' descriptions rather than emotionally engaged ones. This is the consistency bias Kozlowski & Evans identify — the LLM mediates toward an average position that doesn't capture the variance in real human responses.)*

---

### Module 3: AutoGen Demonstration (15 min)

**Navigate to Cells 193–213.**

**Say:**
> "Let me walk through the AutoGen setup. Cell 193 defines the `llm_config` — the API connection and model parameters. Note `cache_seed: 50` — this enables result caching so repeated runs with the same inputs return the same outputs. For research reproducibility, setting a cache seed is essential."

**Run Cell 194–195 live if your API key is configured.**

**Say:**
> "Watch the conversation unfold. The agents are responding in real time through the API. Notice how the 'I gotta go' termination condition stops the conversation cleanly. Now let's look at the sequential onboarding pattern in Cell 201."

**Navigate to Cell 201 (sequential chats).**

**Say:**
> "The `chats` list defines a pipeline of conversations. Each entry specifies a sender, a recipient, an initial message, and a summary method. The `reflection_with_llm` summary method asks the LLM to summarize what was established in each phase and carries that summary into the next phase. This is how AutoGen maintains context across a multi-phase conversation — not by passing raw transcripts, but by summarizing."

**Navigate to Cells 207–208 (Writer-Critic).**

**Say:**
> "The writer-critic pattern. Cell 207 defines the writer agent. Cell 208 defines the critic with `is_termination_msg` set to check for 'TERMINATE'. The critic ends the conversation by including 'TERMINATE' in its final message after completing the feedback cycle. Run this and observe the quality difference between the first and last draft."

---

### Module 5: RAG — Retrieval-Augmented Generation (10 min)

**Navigate to Cell 235 header.**

**Say:**
> "Module 5 introduces Retrieval-Augmented Generation — one of the most practically important techniques for grounding LLM-based agents in specific knowledge. The problem RAG solves: LLMs have a training cutoff date and limited context windows. If your simulation involves agents who should have access to specific documents — court decisions, policy papers, news articles — you cannot fit all that text into the context. RAG builds an external knowledge store and retrieves relevant passages at inference time."

**Write on board:**
```
RAG Pipeline:
1. Documents → chunks → embeddings → vector index (FAISS/ChromaDB)
2. Query → embedding → nearest-neighbor search → retrieved passages
3. Query + retrieved passages → LLM → grounded response
```

**Navigate to Cells 241–252.**

**Say:**
> "Steps 1–5 in the notebook walk through this pipeline explicitly. Cell 244 embeds your document collection using `sentence-transformers`. Cell 246 builds a FAISS index for fast approximate nearest-neighbor search. Cell 248 implements the retrieval function — given a query, find the k most similar document chunks. Cell 250 implements the full RAG pipeline: retrieve, then generate."

**Navigate to Cell 252 (RAG vs. vanilla comparison).**

**Say:**
> "Cell 252 compares RAG output to vanilla LLM output on the same question. This comparison is important for your homework — it is not sufficient to implement RAG; you must demonstrate that it improves response quality or factual accuracy for questions that require document-specific knowledge."

**Say:**
> "For your final projects, RAG is particularly valuable if you are building an agent that needs to reason about specific historical events, legal precedents, institutional documents, or scientific literature that postdates the model's training cutoff. A digital double of a 2024 election voter who reads specific news sources can be given a RAG store built from those sources."

---

## Section 5: Student Code Presentations (20 min)

**Say:**
> "Let's hear from students who applied the Week 2 techniques to their project corpora. I'm particularly interested in the SHAP analysis results — what features was your BERT model attending to?"

**Invite 2–3 students to present. For each, prompt:**

- "Show us the t-SNE plot of your word embeddings. What semantic clusters are visible?"
- "When you projected your domain vocabulary onto social dimensions — gender, class, or any dimensions you constructed — what did you find? Does it match your theoretical expectations?"
- "What did the SHAP values for your fine-tuned BERT model reveal? Which words are most predictive of your classification labels?"

**After presentations, ask the class:**
> "Kozlowski, Taddy, and Evans argue that the geometry of word embedding space mirrors the geometry of cultural space — that you can read social structure off of semantic structure. Do the findings from your classmates' projects support that claim for their specific domains?"

---

## Section 6: Discussion — When Do Simulations Mislead? (25 min)

### Structured Discussion

**Say:**
> "This is the most important discussion of the semester, and I want to spend real time on it. We have built systems that can simulate human survey responses with measurable accuracy. The question is not whether to use these systems — they already exist and are being used by researchers and political campaigns alike. The question is: when can you trust them, and when will they mislead you?"

**Write on board:**
> Core tension: **Predictive accuracy on known populations ≠ validity for novel populations or novel questions**

**Discussion prompt 1:**
> "Argyle et al. find that LLMs can 'simulate human samples' with high fidelity. Kozlowski and Evans find systematic biases — overrepresentation of the ideological center, homogenization of minority views, insensitivity to framing effects. How do you reconcile these findings? Under what conditions would you trust the simulation?"

*(Guide toward: trust the simulation when the question type resembles the training distribution, when the demographic group is well-represented in training data, when the question is not framing-sensitive, and when you have held-out validation data to measure fidelity. Do not trust it for novel questions, underrepresented groups, or questions where framing effects are theoretically central.)*

**Discussion prompt 2 — reference Liebo et al. (2025):**
> "Liebo et al. argue that LLMs are not predicting what a person would answer — they are pattern-completing to the most likely response given the prompt context. In other words, they are producing the culturally normative answer, not a simulated individual's answer. What implications does this have for interpreting simulation results?"

*(This is a fundamental epistemological point. The LLM is not modeling a person's psychology; it is producing text that matches the statistical distribution of text associated with that demographic profile. For predicting modal behavior this may be fine. For studying heterogeneity, preference intensity, or within-group variation, it will systematically fail.)*

**Discussion prompt 3 — COVID-19 polarization:**
> "Kozlowski, Kwon, and Evans's 'In Silico Sociology' paper uses LLM agents to forecast COVID-19 polarization dynamics. They simulate agents with different political profiles receiving information about the pandemic and observe how their simulated opinions evolve. What methodological assumptions would you need to accept to trust those forecasts? What would falsification look like?"

*(Expected responses: you would need to assume that the way the LLM simulates opinion change from information matches how real humans update beliefs; you would need actual longitudinal data on opinion change to compare against the simulation; you would need to worry about whether the LLM's training data includes post-hoc narratives about COVID polarization that contaminate the forward-looking simulation.)*

**Say:**
> "These are not rhetorical questions. They are the research frontier. The empirical validation of LLM-based social simulation is one of the most active areas of social science methodology right now, and you are studying it at the moment when the core results are still coming in."

---

## Closing Summary (5 min)

**Say:**
> "Three take-aways from today. First: LLMs can be configured as social actors through prompting — zero-shot, few-shot, chain-of-thought, and actor-critic are the basic tools, and demographic injection is how you create personas. The vote prediction experiment demonstrates both the power and the specificity of this approach. Second: multi-agent simulation adds social dynamics to individual simulation. AutoGen provides conversational multi-agent architectures; Concordia provides situated simulation with shared world context and formative memories. Each is better suited for different kinds of social processes. Third: RAG grounds agents in specific knowledge, solving the context limitation of bare LLMs."

> "But none of these techniques resolves the fundamental tension Kozlowski and Evans identify: LLMs do not simulate specific people, they simulate patterns in text associated with demographic categories. The consistency bias, the centrist pull, the underrepresentation of minority views — these are systematic distortions. Your job as a researcher is to measure the distortion, not to ignore it."

> "Next week we move from simulation to causation. Week 4 covers experimental designs with AI agents — how to move from the correlational patterns you can observe in simulation to genuine causal estimates. Read the Broska et al. 'Mixed Subjects Design' paper."

---

## Homework Briefing (5 min)

**Say:**
> "The homework is `Week_3.ipynb`. The structure is different from previous weeks: Modules 1, 2, and 5 are mandatory. You choose one from Modules 3 (AutoGen) or 4 (Concordia) — though you are encouraged to attempt both. Module 1 is the prompting fundamentals, Module 2 is the digital doubles and vote prediction, and Module 5 is the RAG system."

> "The key deliverable: validation. For Module 2, you must compare your LLM-predicted votes to the ground-truth votes in the dataset and report F1 or accuracy. Do not just report that you ran the code — report whether it worked, and try to understand why it succeeded or failed for specific demographic subgroups."

> "For Module 3 or 4: take one research question from your project (or a plausible extension) and reformulate it as a multi-agent simulation. Your agents should have distinct personas, a social situation, and a research question the simulation is designed to address. Describe what you would need to observe in the simulation to consider it a valid representation of the social process you care about."

**Write on board:**
```
Lab: Gio Choi — Thursday 2–3pm
Homework due: Before Week 4 (Jan. 30)
Required: Modules 1, 2, 5 + one of Module 3 or 4
Key deliverable: Validation against ground truth; framed research question
Memo: 300–500 words + figure; due before class
```

---

## Timing Guide

| Section | Content | Time |
|---|---|---|
| 1 | The promise of digital doubles: motivation and literature | 15 min |
| 2 | Prompting strategies: zero-shot, few-shot, CoT, actor-critic | 30 min |
| 3 | Multi-agent architectures: AutoGen vs. Concordia | 30 min |
| Break | — | 10 min |
| 4 | Code walkthrough: vote prediction, AutoGen, RAG | 60 min |
| 5 | Student code presentations | 20 min |
| 6 | Discussion: when do simulations mislead? | 25 min |
| Closing | Summary + homework briefing | 10 min |
| **Total** | | **3 hr** |

---

## Instructor Notes

**Common student confusion points:**
1. The difference between zero-shot and few-shot prompting — zero-shot gives the instruction only; few-shot provides labeled examples before the test case. In the code, this is the difference between a simple `user` message and a list of alternating `user`/`assistant` messages that constitute the examples.
2. Why RAG uses vector similarity rather than keyword search — embedding similarity captures semantic relatedness, not just lexical overlap. A query about "fiscal policy" will retrieve chunks about "government spending" even if those exact words do not appear in the query.
3. The `human_input_mode="NEVER"` setting in AutoGen — this is what makes the conversation fully automated; setting it to `"ALWAYS"` lets the human intervene between turns; `"TERMINATE"` pauses at the end. For homework, use `"NEVER"` unless you want an interactive session.

**If the API key is unavailable during class:**
> Use DeepSeek-R1 as an alternative (Cells 81–88 in the notebook). The vote prediction results may differ somewhat from GPT-4, but the methodology is identical. Alternatively, demonstrate the code structure and logic without running it live, showing pre-computed outputs.

**Handling the cost concern:**
> The vote prediction loop (Cell 150) calls the API once per respondent. For a sample of 100 respondents with GPT-3.5-turbo, cost is approximately $0.02. With GPT-4o-mini, approximately $0.05. Set a maximum sample size appropriate for your budget. The `tqdm` progress bar allows graceful interruption.

**Connections to upcoming weeks:**
- The actor-critic prompting pattern (this week) directly anticipates the actor-critic architectures in Week 7 (RL)
- The RAG system (Module 5) uses embedding similarity — this connects directly back to Week 2's embeddings and forward to Week 5's fine-tuning and benchmarking
- The digital doubles methodology is evaluated and validated in Week 4's experimental designs section; the same simulated respondents become subjects in causal inference studies
- The systematic biases in LLM simulation (consistent, centrist, majority-group-favoring) become targets for the steering vectors and alignment techniques in Weeks 6 and 9

**References for discussion facilitation:**
- Argyle et al. (2023): reports high fidelity on political opinion surveys; find this on Cambridge Political Analysis
- Kozlowski & Evans (2025): identifies the centrist pull and consistency bias; critical counterpoint to Argyle
- Park et al. (2023): the Smallville village paper; key example for Concordia-style situated simulation
- Liebo et al. (2025): the pattern-completion critique — LLMs produce normative answers, not individual psychology
