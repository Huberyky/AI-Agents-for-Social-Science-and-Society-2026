# Week 8 Lecture Script
# AI Agents for Social Science and Society 2026
# Multi-Modal and Embodied Agents
# Date: February 27, 2026 | 1:30–4:20 PM | Room 295

---

## Instructor Notes: Before Class

- Open `Week_8_Multimodal.ipynb` in Colab and confirm GPU is available (Runtime → Change runtime type → T4 GPU)
- Pre-run Cells 0–10 to confirm CLIP and dependencies are loaded; this saves ~5 minutes in class
- Prepare 2–3 example images that are sociologically interesting (a street scene, a protest photo, a corporate headshot)
- Have the Guilbeault et al. (2025) "Age and Gender Distortion" paper open — Figure 1 is worth projecting
- Write the session's central question on the board before students arrive
- The notebook is 169 cells — you will not demo all of it; prioritize Modules 1 (Sections A–D), 2 (Section C bias), and 3 (ReAct)
- Estimated total class time: 170 minutes (10-minute break ~90 minutes in)

---

## Section 1: Opening — Perception, Bias, and the Social World
### [0:00 – 0:15 | 15 minutes]

**Write on board before class:**
```
Central question: When AI agents look at the social world,
what do they see — and what do they project onto it?
```

**Say:** "Welcome to Week 8. We've spent seven weeks working almost entirely in text. Language models read and write words. But the social world is not only words. It's faces and buildings and voices and gestures. This week we give our agents eyes and ears. We look at how they process images, audio, and video — and we spend a significant portion of class asking a harder question: what happens when perception itself is biased?"

**Ask students:** "Raise your hand if you've ever done a Google Image search for a job title — like 'nurse' or 'CEO' or 'scientist.' What did you notice about the images?"

*Expected: mostly women for nurse, mostly men for CEO. If students haven't noticed this explicitly, you can state it directly.*

**Say:** "Guilbeault, Delecourt, and Desikan (2025) systematically documented this. When you search for 'CEO,' you get images skewed toward older men. When you search for 'nurse,' you get images skewed toward younger women. But here is the worse finding: when LLMs describe those images, they *amplify* the bias. The model takes an already-skewed visual distribution and makes it more skewed in its description. Perception is not neutral — and when AI agents perceive the social world, they bring their training data's biases into every observation."

**Write on board:**
```
Today's arc:
Seeing (CNNs, CLIP, VLMs) → Hearing (spectrograms, Whisper) →
Watching (video models) → Connecting (multimodal) →
Generating (diffusion) → Acting (ReAct agents) → Embodiment
All of it filtered through: social bias in visual AI
```

---

## Section 2: Convolutional Neural Networks and Visual Perception
### [0:15 – 0:45 | 30 minutes]

**Say:** "Let's start from the ground up. How does a neural network 'see'? A raw image is just a grid of numbers — pixel values. The challenge is that meaning is not local. A single pixel tells you almost nothing. The relevant structure is at the level of edges, textures, objects, and scenes. Convolutional neural networks are engineered to extract exactly that hierarchy."

**Write on board:**
```
CNN hierarchy:
Layer 1: detect edges (horizontal, vertical, diagonal)
Layer 2: combine edges into textures, curves
Layer 3: combine curves into object parts (eyes, wheels)
Layer 4: combine parts into objects (face, car)
Layer 5: combine objects into scenes
```

**Say:** "The key operation is the **convolution**: a small filter (e.g., 3×3) slides across the image and computes a dot product at every position. This gives a *feature map* — a new grid that shows where that pattern appears in the image. With many filters at each layer, you build up a rich representation."

**Write on board:**
```
Key CNN concepts:
- Filter/kernel: small learnable pattern detector
- Stride: how far the filter moves each step
- Pooling: downsample (max or average) to reduce spatial size
- Receptive field: how large a patch of the original image a neuron 'sees'
- Depth: number of filters = number of channels in feature map
```

**Ask students:** "Why do we use pooling? What does it buy us?"

*Expected: spatial invariance (the object might be slightly shifted); dimensionality reduction; translation invariance. Accept all of these.*

**Say:** "Modern CNN architectures like ResNet, VGG, and EfficientNet are variations on this theme. ResNet's key innovation is the **residual connection**: add the input directly to the output of a block. This allows gradients to flow through hundreds of layers without vanishing. ResNet-50 has 50 layers and won the ImageNet competition in 2015."

**Transition to CLIP — Code Demo, Cells 8–9:**

**Say:** "For this course, we're less interested in training CNNs from scratch and more interested in *using* pre-trained vision models as the perceptual backbone for AI agents. The most powerful general-purpose vision model right now is **CLIP** — Contrastive Language-Image Pre-training."

**Write on board:**
```
CLIP (Radford et al. 2021):
- Train on 400M image-text pairs from the internet
- Image encoder → image embedding
- Text encoder → text embedding
- Contrastive loss: pull matching pairs together, push non-matching apart
- Result: shared embedding space for images and text
```

**Code Demo Note — Cell 9:**
```python
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Zero-shot classification
text_labels = ["a dog", "a cat", "a car", "a mountain"]
text_tokens = clip.tokenize(text_labels).to(device)
```

**Say:** "Notice what's remarkable here: we can classify an image into any category we can express in natural language. There's no fixed vocabulary. This is 'zero-shot' classification — we never trained on these specific categories. This makes CLIP extremely powerful for social science applications where your categories are theoretically defined, not off-the-shelf."

**Ask students:** "If you were studying how political candidates are visually presented in news media, how might you use CLIP?"

*Expected: embed images of candidates, use text queries like 'a confident leader,' 'a working class person,' 'a foreign threat'; measure semantic similarity; build a dataset of visual framing.*

---

## Section 3: Audio Processing — Hearing the Social World
### [0:45 – 1:05 | 20 minutes]

**Say:** "Vision gets most of the attention in AI, but audio is deeply important for social science. Tone of voice, hesitation, accent, speech patterns — these carry vast social information. Let's talk about how neural networks process sound."

**Write on board:**
```
Audio processing pipeline:
1. Waveform: raw pressure samples (16,000 samples/sec for speech)
2. STFT (Short-Time Fourier Transform): frequency content over time
3. Mel spectrogram: frequency mapped to human perceptual scale
4. Input to model: 2D image of time × frequency
```

**Say:** "The key insight is that we convert audio — a 1D signal over time — into a 2D image-like representation (the spectrogram), and then we can apply the same convolutional or transformer architectures we use for images. The Mel scale compresses high frequencies because human hearing is not linear — we distinguish small differences at low frequencies and large differences at high frequencies."

**Code Demo Note — Cells 22–26:**
Show the waveform and spectrogram plots from Cell 23. **Say:** "Look at this spectrogram. Time is on the x-axis, frequency on the y-axis, and brightness is energy. You can literally *see* speech — the horizontal stripes are voiced sounds, the vertical spikes are consonants. Whisper — OpenAI's speech recognition model — takes exactly this as input."

**Code Demo Note — Cells 24–26 (Whisper):**
```python
import whisper
model_whisper = whisper.load_model("turbo")  # 809M params

def transcribe(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model_whisper.device)
    _, probs = model_whisper.detect_language(mel)
    options = whisper.DecodingOptions()
    result = whisper.decode(model_whisper, mel, options)
    return result.text
```

**Say:** "Whisper is trained on 680,000 hours of multilingual audio from the internet. It handles 99 languages. The social science applications are enormous: analyzing oral histories, court recordings, legislative hearings, political speeches, focus groups. And importantly: it detects the language automatically, which means you can run it on multilingual corpora without pre-classification."

**Ask students:** "What ethical issues arise when you apply automatic speech recognition to sensitive social data — like therapy sessions, police interactions, or testimony from vulnerable populations?"

*Expected: accuracy disparities across accents, genders, ages; consent and privacy; the model may perpetuate the biases of its internet training data; transcription errors that distort meaning in qualitative coding.*

**[BREAK — 10 minutes]**

---

## Section 4: Video Understanding and Vision-Language Models
### [1:15 – 1:40 | 25 minutes]

**Say:** "Video is the hardest modality — it's images over time, and temporal dynamics carry meaning that static frames do not. Think about protest footage: a single frame tells you people are outdoors; the temporal sequence tells you whether the crowd is dispersing or converging."

**Write on board:**
```
Video understanding approaches:
1. 3D convolutions: extend spatial filters to time dimension
2. Two-stream: separate spatial (appearance) + temporal (optical flow) streams
3. Video transformers: TimeSformer, VideoMAE
   - Treat video as sequence of patches across space AND time
   - Masked autoencoding: reconstruct masked video patches
```

**Code Demo Note — Cells 32–34 (VideoMAE):**
**Say:** "Cell 32 downloads a video and extracts frames with OpenCV. Cell 34 loads VideoMAE — a masked autoencoder for video. Note how we sample 16 evenly-spaced frames from the video. This is a critical design choice: too few frames and you miss temporal dynamics; too many and you blow GPU memory."

**Transition to Vision-Language Models (VLMs):**

**Say:** "The really powerful recent development is combining vision and language in a single model. GPT-4V, Qwen3-VL — these models can take an image as input and answer open-ended questions about it. They use cross-modal attention mechanisms that let language tokens attend to image patch tokens."

**Code Demo Note — Cells 12–15:**
```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_vlm = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

**Say:** "In Cell 13, we give the model a food image and ask it to describe what it sees. In Cell 15, we compare this to GPT-4o's response for the same image. Note the qualitative differences: Qwen tends to be more concise; GPT-4o tends to give more detailed cultural context. For social science, this calibration matters — if you're analyzing images at scale, you need to know what your model emphasizes versus omits."

**Ask students:** "GPT-4V can describe protest images, analyze facial expressions in political advertisements, and read text in street photography. What are the research use cases? And what are the failure cases?"

*Expected uses: media analysis, protest dynamics, visual framing of social groups. Failure cases: hallucination about what's in an image; racial misidentification; applying Western cultural scripts to non-Western visual contexts.*

**Say:** "The GPT-4V System Card from OpenAI (2023) discusses some of these failure modes explicitly. The card notes that the model can make sensitive inferences about individuals from images — guessing profession, political affiliation, emotion. OpenAI adds guardrails to limit some of these. But deployed as an API, those guardrails can be bypassed with careful prompting. This is the tension between capability and safety that we'll return to in Week 9."

**Code Demo Note — Cells 40–48 (CLIP semantic search):**
**Say:** "Section D of Module 1 is perhaps the most immediately useful for social scientists. Using CLIP embeddings, you can build a semantic image search: type a query like 'a photograph of police confronting protesters' and retrieve the most semantically similar images from a corpus — even if those images were never explicitly labeled."

**Write on board:**
```python
# CLIP semantic search
query_emb = model.encode(["a protest in a city street"], convert_to_tensor=True)
hits = util.semantic_search(query_emb, img_embs, top_k=5)
```

**Say:** "Cell 47 takes this further: it visualizes both text queries and images in the same 2D space using t-SNE. Text and images cluster together if they're semantically related. 'A dog' lands near dog photos. This shared embedding space is what makes CLIP so powerful for cross-modal retrieval and analysis."

---

## Section 5: Diffusion Models and Social Bias in Generation
### [1:40 – 2:10 | 30 minutes]

**Say:** "Now let's talk about the other direction: not *perceiving* social content, but *generating* it. Diffusion models are the state of the art for image generation, and they're where social bias becomes most visibly problematic."

**Write on board:**
```
Diffusion models (DDPM):
Forward process: x_0 → x_1 → ... → x_T (add Gaussian noise at each step)
Reverse process: x_T → ... → x_0 (learn to denoise)
Training: predict noise added at each step
Generation: start from pure noise, iteratively denoise
Conditioning: use text embedding to guide denoising
```

**Code Demo Note — Cells 57–65:**
**Say:** "Cell 57 shows the forward noise process visually — an image progressively corrupted into noise. Cell 60 shows the noise schedule: early steps destroy fine detail; later steps destroy global structure. Cell 62 loads the components of Stable Diffusion: the VAE (compresses images to latent space), the U-Net (predicts noise), and the CLIP text encoder (conditions generation on text)."

**Say:** "What I want you to notice in Cell 63 is that the *same* CLIP text encoder from Section D is being used here. CLIP is the bridge between language and images — both in perception (understanding images) and generation (creating images that match text). This shared architecture means bias in CLIP propagates to bias in generation."

**Transition to the Bias Demo — Cells 75–87:**

**Say:** "This is the most important part of Module 2 for social scientists. Cell 77 sets up an experiment based directly on Guilbeault et al. (2025). We generate images for occupations — some traditionally male-associated (military officer, engineer), some traditionally female-associated (nurse, teacher) — and then use GPT-4o to automatically classify the perceived gender and age of the generated figures."

**Code Demo Note — Cell 77:**
```python
occupations = {
    "male-associated": ["military officer", "engineer", "CEO", "scientist"],
    "female-associated": ["nurse", "teacher", "librarian", "social worker"],
    "neutral": ["doctor", "lawyer", "chef", "journalist"]
}
```

**Say:** "Cell 81 is particularly elegant: it uses GPT-4o as a classifier of its own (or Stable Diffusion's) outputs. We pass the generated image to GPT-4o and ask: 'What is the apparent gender and approximate age of the person in this image?' This is automated visual coding — the same thing human research assistants would do, but at scale."

**Write on board:**
```
Key finding (Guilbeault et al. 2025):
1. Generated images skew gender by occupation (replicates search bias)
2. Generated women appear younger than generated men (age-gender intersection)
3. LLM descriptions of biased images amplify, not reduce, the bias
4. Explicit gender prompts ("a female CEO") reduce but don't eliminate bias
```

**Ask students:** "If you were a hiring manager who used an AI image tool to generate images of candidates for job descriptions, or a researcher using image generation to create stimuli for experiments — what's the practical harm?"

*Expected: reinforces stereotypes in the minds of viewers; creates stimulus materials that introduce confounds; normalizes underrepresentation; could affect how people perceive what a CEO or nurse 'looks like.' This connects to Guilbeault et al.'s earlier work on how online images amplify gender bias in LLM embeddings.*

**Ask students:** "Can you 'fix' this bias? What interventions would you consider?"

*Expected: curated training data; explicit debiasing; prompt engineering ('a professional photograph of a nurse, diverse'); adversarial debiasing at the representation layer. Discuss tradeoffs — debiasing one dimension may introduce bias on another.*

**Say:** "Cell 86 tests this directly: it generates images under three conditions — no gender specified, explicitly male, explicitly female — and compares the results. Even with explicit gender prompts, the model's background assumptions about age and physical appearance still bleed through."

---

## Section 6: REACT — Synergizing Reasoning and Acting
### [2:10 – 2:35 | 25 minutes]

**Say:** "We've talked about perceiving the world. Now let's talk about *acting* in it. Module 3 of the notebook introduces the ReAct framework — one of the most influential ideas in the AI agent literature."

**Write on board:**
```
ReAct (Yao et al., 2025) = Reasoning + Acting
Loop:
  Thought: "I need to find out who wrote this song..."
  Action: search("The Simpsons Milhouse")
  Observation: [search results]
  Thought: "Now I know Milhouse was named after Richard Nixon..."
  Action: search("Richard Nixon president number")
  Observation: [results]
  Answer: 37
```

**Say:** "The key innovation in ReAct is interleaving language model reasoning with tool use. The model is not just answering from memory — it's reasoning about what it doesn't know, deciding what to look up, and integrating retrieved information into its reasoning. This is qualitatively different from chain-of-thought prompting, where the model can only reason over what's already in its context."

**Code Demo Note — Cells 101–105:**
Show the tool definitions:
```python
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return the first paragraph."""
    ...

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for a topic",
            "parameters": {"type": "object", "properties": {"query": {...}}}
        }
    }
]
```

**Say:** "Cell 103 is the agent loop — the core 10 lines that make a ReAct agent. The model sends messages to the API; if the response contains a tool call, we execute the function and add the result to the conversation; if not, we return the final answer. This pattern generalizes to any tools you want to give the agent."

**Code Demo Note — Cell 105:**
**Say:** "The test question is directly from the ReAct paper: 'Musician and satirist Allie Goertz wrote a song about the Simpsons character Milhouse, who Matt Groening named after which President?' This requires two searches and combining information. Watch the trajectory: first search, get an observation, reason about it, second search, combine, answer. Cell 109 plots the trajectory visually — each step is labeled with the thought, action, and observation."

**Code Demo Note — Cells 110–117 (Multimodal ReAct):**
**Say:** "Section C of Module 3 extends the agent with perception tools. Now the agent can call `describe_image(url, question)` or `transcribe_audio(path)` as tools. Cell 114 shows a remarkable thing: the agent is given an image of an ant, it calls the vision tool to identify it, then calls the search tool to look it up, and combines both in its final answer. Vision and language are fully interoperable in the tool loop."

**Ask students:** "For your final project — what tools would you want to give a ReAct agent? What data sources would it need to look things up? What perception tools would it need?"

*Give students 3 minutes to think and share. This is good scaffolding for the homework.*

---

## Section 7: Embodied Agents and the Grounding Problem
### [2:35 – 2:55 | 20 minutes]

**Say:** "The final module in the notebook takes us from virtual perception to physical embodiment. Embodied AI agents must not only reason about the world but act in it — robots, smart home systems, autonomous vehicles. The central challenge is the **grounding problem**: how do you map high-level natural language goals onto low-level physical actions?"

**Write on board:**
```
Embodied planning pipeline (TaPA, Song et al. 2023):
1. Perceive scene → detect objects (Grounding DINO)
2. Get task instruction ("Can you make me coffee?")
3. Ground plan: only reference objects that were detected
4. Execute sequence of primitives: "pick up mug", "place under coffee machine"

Failure mode: ungrounded plans reference objects that don't exist in the scene
```

**Code Demo Note — Cells 135–148:**
**Say:** "Section B of Module 4 uses Grounding DINO — an open-vocabulary object detector — to inventory a scene. The model returns bounding boxes and labels for everything it can identify. Cell 138 shows this on a kitchen scene. Cell 144 then passes that object list to GPT-4o to generate a task plan."

**Code Demo Note — Cell 145:**
**Say:** "The critical comparison is in Cell 146: grounded planning (which only references detected objects) versus ungrounded planning (which generates plans from LLM priors alone). The ungrounded plan might say 'use the coffee machine' even if no coffee machine was detected. Cell 157 validates this: we run both plans through a simulated room state tracker and count how many steps fail because of missing objects."

**Say:** "This is the sim-to-real gap: in simulation everything works; in reality, the agent trips over the cat. Social science has its own version of this. Simulations of social behavior — like the digital doubles we built in Week 3 — are trained on text about humans, not humans themselves. The grounding problem for social AI is: how do you make agents that are grounded in actual human experience, not just text descriptions of that experience?"

**Ask students:** "Ludwig and Mullainathan (2024) argue that ML is most valuable not for prediction but for *hypothesis generation* — using a model to reveal patterns in data that suggest new theories. How does embodied perception change what kinds of hypotheses are possible in social science?"

*Expected: you can generate hypotheses from visual and audio data that were never textually described — neighborhood appearance and economic outcomes, acoustic patterns in deliberative vs. contentious meetings, body language in political negotiations.*

---

## Section 8: Code Walkthrough and Homework Briefing
### [2:55 – 3:20 | 25 minutes]

**Say:** "The homework is structured around four modules, all required. Let me walk through what each involves."

**Write on board:**
```
Week 8 Homework (all modules):
Module 1: Sound, Image & Video Basics
  - CLIP zero-shot on your own images
  - Whisper transcription on domain-relevant audio
  - VideoMAE or Qwen3-VL on video data

Module 2: Conditional Generation
  - Stable Diffusion / SDXL text-to-image
  - Bias audit: generate images for 10+ social categories
  - Use GPT-4o to classify generated images (auto-coding)

Module 3: Multimodal Agents
  - Build a ReAct agent with custom tools
  - Add at least one perception tool (vision or audio)
  - Test on domain-relevant multi-hop questions

Module 4: Social Analysis
  - Apply multimodal analysis to a social dataset from your project
  - Build CLIP semantic search on your image corpus
  - Connect to your final project framing
```

**Code Demo Note — Final connection (Cell 43):**
Show the semantic image search function:
```python
def search(query, k=3):
    query_emb = model_st.encode([query], convert_to_tensor=True)
    hits = util.semantic_search(query_emb, img_embs, top_k=k)
    ...
```

**Say:** "This 10-line function is one of the most practical tools you'll take from this course. If you have an image corpus — news photos, social media images, Google Street View, protest photographs — you can search it semantically without any manual labeling. Think about what that enables: 'find all images showing police presence' or 'find images of economic hardship' or 'find images where women are in leadership roles.' This is AI-assisted content analysis at scale."

**Ask students:** "For your final project, what image or audio corpus might you want to analyze? What text queries would operationalize your theoretical concepts?"

*Give 3 minutes for pair discussion, then have 2–3 students share.*

---

## Section 9: Discussion — Bodies, Sensors, and Social Robots
### [3:20 – 3:40 | 20 minutes]

**Say:** "I want to close with a broader question. Embodied AI is becoming real: delivery robots, care companions for the elderly, social robots in schools. These agents don't just process language — they perceive physical social environments and act in them. What does this mean?"

**Write on board:**
```
Three questions for discussion:
1. What new social science questions become answerable with multimodal AI?
2. How do visual and auditory biases in AI affect vulnerable populations?
3. What does it mean for an AI to have a body — in simulation or reality?
```

**Ask students:** "Research by Ekman on facial expressions claimed that emotions are universally displayed and recognized across cultures. Recent work — including Gendron et al. — challenges this. If you deploy a facial expression recognition system trained on Western populations globally, what's the risk?"

*Expected: misidentification of emotional states; false positives in threat detection systems; automation of racially biased policing; misreading of cultural emotional norms as individual affect.*

**Say:** "The Street View research by Gebru et al. and others showed you can predict neighborhood income, voting patterns, and demographic composition from street-level imagery. This is powerful — but it also encodes biases about what 'affluence' looks like. An algorithm trained to identify 'good neighborhoods' from street appearance will reflect and potentially reinforce historical patterns of investment and neglect."

**Say:** "The core tension of this week is that multimodal AI dramatically expands what social science can measure — more data, new data types, previously inaccessible archives. But every expansion of measurement also expands the reach of the biases baked into the models. Being a responsible practitioner means developing the technical literacy to audit those biases, not just exploit the capabilities."

---

## Closing Summary
### [3:40 – 3:50 | 10 minutes]

**Write on board:**
```
Today's key ideas:
1. CNNs extract hierarchical visual features; transfer learning makes this accessible
2. CLIP creates a shared image-text embedding space for zero-shot analysis
3. Audio → Mel spectrogram → transformer (Whisper for ASR)
4. VLMs (GPT-4V, Qwen3-VL) enable open-ended visual reasoning
5. Diffusion models generate images but encode and amplify social biases
6. ReAct agents interleave reasoning and tool use (including perception)
7. Embodied agents must solve the grounding problem: plans constrained by scene reality
8. All of this: perception is not neutral — biases in training data become biases in perception
```

**Say:** "Next week — Week 9 — is the culmination of the course. We've built agents that learn, perceive, reason, and act. Week 9 asks: how do we make sure they're aligned with human values? How do we detect when they're not? And what does their growing presence mean for human creativity and science? We'll use the Qwen2.5-7B model running locally on Colab — the most technically intensive notebook of the course."

**Homework reminders:**
- Complete all four modules of `Week_8_Multimodal.ipynb`
- Weekly memo: connect one of today's papers to your final project — particularly the bias findings
- Lab session with Avi Oberoi: Tuesday 11am–12pm — bring your multimodal tool design questions
- Reading for Week 9: Hubinger et al. 2024 "Sleeper Agents" and Gabriel et al. 2025 "We Need a New Ethics for AI Agents" — both are essential

---

## Appendix: Anticipated Student Questions and Instructor Responses

**Q: What's the difference between CLIP and GPT-4V?**
A: CLIP gives you embeddings — numerical representations in a shared image-text space — ideal for search and classification. GPT-4V generates text descriptions and answers questions about images; it's more flexible but slower and more expensive. For large-scale corpus analysis, CLIP is often the right tool. For complex reasoning about specific images, GPT-4V.

**Q: Can I use CLIP to analyze my own image corpus for my final project?**
A: Yes — this is explicitly what Module 4 is asking you to do. Make sure you have the images accessible (local files or public URLs). The SentenceTransformers CLIP wrapper makes this very straightforward.

**Q: How do diffusion models differ from GANs?**
A: GANs (Generative Adversarial Networks) pit a generator against a discriminator in a minimax game. Training is notoriously unstable. Diffusion models use a fixed forward noise process and learn to reverse it — more stable training, better coverage of the data distribution. The tradeoff: diffusion models are slower to sample from (hundreds of denoising steps vs. one GAN pass).

**Q: Why does the Guilbeault et al. paper matter for my research if I'm not studying gender?**
A: The principle generalizes. Any AI system trained on internet data will encode whatever biases exist in that data — racial, socioeconomic, political, cultural. Before using a model to analyze or generate social content in your domain, you should ask: what demographic or cultural groups are underrepresented in the training data? What patterns from those groups might be incorrectly generalized?

**Q: Is there a way to do multimodal RAG — like retrieval-augmented generation but with images?**
A: Yes — and the notebook briefly mentions it. You embed your image corpus with CLIP, use semantic search to retrieve relevant images given a query, and pass those images plus text to a VLM. This is exactly the same RAG architecture from Week 3, but with images as the retrieved documents.

---

## Key Papers Referenced Today

| Paper | One-line summary |
|---|---|
| Yao et al. (2025) REACT | Interleaving reasoning and acting in LLM agents via tool calls |
| OpenAI (2023) GPT-4V System Card | Capabilities, limitations, and safety evaluations of GPT-4V |
| Song et al. (2023) TaPA | Task planning with LLMs grounded in detected scene objects |
| Guilbeault, Delecourt, Desikan (2025) | Age and gender distortion in AI-generated images of occupations |
| Ludwig & Mullainathan (2024) | ML as a tool for hypothesis generation, not just prediction |
| Gebru et al. (2017) | Street-level imagery predicts neighborhood socioeconomic status |
