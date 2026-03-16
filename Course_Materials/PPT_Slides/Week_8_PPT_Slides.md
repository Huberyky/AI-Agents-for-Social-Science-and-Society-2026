# Week 8: Multi-Modal and Embodied Agents
## PPT Slide Deck — 30 Slides

---

### Slide 1: The Social World is Not Just Text
**Visual Description:** Split panel: left shows dense text corpus (tweets, papers, books); right shows collage of images (street scenes, faces, memes), audio waveforms, and video frames. Arrow from each pointing to a central "AI Agent" brain icon.

**Bullet Points:**
- 90% of internet data is non-textual
- Images, audio, and video carry cultural meaning language cannot
- Social science needs agents that perceive the full sensory world

**Instructor Notes:** Open by asking: what does a political meme communicate that its caption alone does not? This motivates multimodal AI as a social science necessity, not just a technical curiosity.

---

### Slide 2: Roadmap for Today
**Visual Description:** Vertical timeline with icons: Camera (CNNs) → Microphone (Audio) → Video camera (Video) → Merged icon (VLMs) → Robot arm (Embodiment). Each with a small clock showing ~25 min.

**Bullet Points:**
- CNNs: seeing spatial patterns
- Audio & Video: temporal perception
- Vision-Language Models: bridging modalities
- Embodied agents: acting in physical/virtual worlds

**Instructor Notes:** Today is the most technically diverse session of the course. Emphasize the unifying thread: each modality gives agents a new "sense" through which to perceive and reason about the social world.

---

### Slide 3: How Humans vs. CNNs See an Image
**Visual Description:** Side-by-side: human brain diagram with "holistic, top-down" label; CNN diagram showing convolution → activation → pooling layers with feature map visualizations (edges → textures → objects → scenes).

**Bullet Points:**
- Humans: top-down, conceptual, context-driven
- CNNs: bottom-up, hierarchical feature extraction
- Deeper layers = more abstract representations

**Instructor Notes:** The key insight is that CNNs learn a hierarchy of features — not programmed by hand, but discovered through gradient descent. This parallels what we saw in Week 1 for tabular data but extended to spatial structure.

---

### Slide 4: The Convolution Operation
**Visual Description:** Animated diagram showing 3×3 kernel sliding over a 6×6 input matrix. Element-wise multiplication shown with numbers. Output feature map on right. Formula: `(I * K)[i,j] = Σ Σ I[i+m, j+n] · K[m,n]`.

**Bullet Points:**
- Kernel = learned feature detector (edge, curve, texture)
- Stride: step size of kernel movement
- Padding: handling boundaries
- Output size: (W - F + 2P) / S + 1

**Instructor Notes:** Walk through one convolution step on the board with actual numbers. Ask students: what does a kernel that is `[[1,0,-1],[2,0,-2],[1,0,-1]]` detect? (Vertical edges — the Sobel filter.)

---

### Slide 5: CNN Architecture: From Pixels to Predictions
**Visual Description:** Full CNN pipeline: Input image (28×28×1) → Conv(32 filters) → ReLU → MaxPool → Conv(64 filters) → ReLU → MaxPool → Flatten → FC(128) → FC(10) → Softmax. Dimension annotations at each stage.

**Bullet Points:**
- Convolutional layers: spatial feature extraction
- Pooling: spatial downsampling + translation invariance
- Fully-connected head: classification/regression
- Parameters: mostly in FC layers, efficiency in conv layers

**Instructor Notes:** Point out the dramatic dimensionality changes at each stage. A 224×224×3 ImageNet image enters; 1000 class probabilities exit. The conv layers compress while preserving what matters spatially.

---

### Slide 6: Transfer Learning — Standing on ImageNet's Shoulders
**Visual Description:** Two-panel diagram. Left: ResNet-50 trained on ImageNet, layers labeled Early (edges/textures), Middle (parts), Late (objects). Right: same network with final layers replaced; arrow labeled "fine-tune on social images." Examples of transferred features shown.

**Bullet Points:**
- ImageNet: 1.2M images, 1000 categories — features generalize broadly
- Feature extraction: freeze all but final layer
- Fine-tuning: unfreeze top N layers for domain adaptation
- Social applications: faces, scenes, documents, memes

**Instructor Notes:** Show the celebrated visualization by Zeiler & Fergus of what ImageNet CNN layers detect. Early layers are universal (edges, gabors); later layers are task-specific. This is why transfer learning works for social images.

---

### Slide 7: Social Bias in Visual AI — Guilbeault et al. 2025
**Visual Description:** Four-panel result figure. Panel 1: Google Images for "CEO" — predominantly male faces. Panel 2: same query via GPT-4V description — even more skewed. Panel 3: distribution of age in online media vs. LLM descriptions. Panel 4: bar chart showing amplification ratio by demographic group.

**Bullet Points:**
- LLMs amplify existing visual gender bias by 3–5×
- Age distortion: online media under-represents elderly
- LLMs further compress age diversity in descriptions
- AI "sees" the world through a demographically skewed lens

**Instructor Notes:** This is a critical paper for social scientists. The concern is not just that training data is biased (we know this) but that models systematically *amplify* those biases when generating new descriptions or images. Ask: what does this mean for AI-generated research materials?

---

### Slide 8: Processing Audio — From Waveform to Features
**Visual Description:** Three-step pipeline: (1) Raw waveform oscilloscope plot; (2) Short-Time Fourier Transform → spectrogram (time × frequency heatmap); (3) Mel filterbank → Mel spectrogram with perceptually-spaced frequency axis. Python code: `librosa.feature.melspectrogram(y=audio, sr=sr)`.

**Bullet Points:**
- Audio = 1D time-series sampled at 16–44kHz
- STFT: converts windowed segments to frequency domain
- Mel scale: logarithmic frequency spacing matching human hearing
- Log-Mel spectrogram: standard input for audio deep learning

**Instructor Notes:** Play a 5-second audio clip and show its mel spectrogram live. Ask students to identify features: speech looks like horizontal bands; music shows harmonic structure; ambient noise is diffuse.

---

### Slide 9: Deep Learning for Audio — Whisper and Beyond
**Visual Description:** Whisper architecture diagram: Audio → Log-Mel Spectrogram → CNN encoder → Transformer encoder → Transformer decoder → Text output. Icons for applications: speech-to-text, speaker ID, emotion detection, language identification.

**Bullet Points:**
- Whisper (OpenAI): 680K hours of multilingual audio training
- Zero-shot transcription across 99 languages
- Emotion from voice: valence, arousal, dominance
- Social applications: interview transcription, protest audio analysis

**Instructor Notes:** Demonstrate Whisper transcription live. Discuss social science value: oral history interviews, public hearings, legislative debates — all become analyzable text with attribution. Ask: what nuance is lost when voice is converted to text?

---

### Slide 10: Video Understanding — Temporal Dynamics
**Visual Description:** Four-quadrant diagram: Q1 shows video as tensor (T×H×W×C); Q2 shows optical flow arrows overlaid on video frame; Q3 shows 3D convolution kernel with time dimension; Q4 shows TimeSformer architecture with divided space-time attention.

**Bullet Points:**
- Video = sequence of frames + temporal dynamics
- Optical flow: pixel-level motion vectors between frames
- 3D CNNs: extend spatial convolution to time dimension
- Video Transformers (TimeSformer, VideoMAE): self-attention over frame patches

**Instructor Notes:** Emphasize what video adds over images: causality, social interaction, gesture, gaze. For social science: crowd dynamics, protest behavior, courtroom proceedings, consumer behavior in stores — all require temporal understanding.

---

### Slide 11: CLIP — Bridging Vision and Language
**Visual Description:** CLIP architecture: left tower = image encoder (ViT); right tower = text encoder (Transformer). Both project to same embedding space. Contrastive loss shown with positive pairs (image-caption) vs. negatives. Bottom: zero-shot classification pipeline.

**Bullet Points:**
- Trained on 400M image-text pairs from the web
- Objective: maximize similarity of matching image-text pairs
- Zero-shot: classify images by comparing to text descriptions
- Emergent capabilities: visual reasoning, cultural concept detection

**Instructor Notes:** Show zero-shot CLIP classification: feed an image of a protest and compare embeddings with "peaceful demonstration" vs. "violent riot." The results reveal what visual patterns the model associates with each label — and where those associations might be biased.

---

### Slide 12: GPT-4V — Seeing and Reasoning Together
**Visual Description:** GPT-4V system card diagram: image tokens (patches via ViT) + text tokens both fed into transformer decoder. Example prompt: [photo of a political poster] "What demographic is this advertisement targeting and what rhetorical strategies does it use?" with model response.

**Bullet Points:**
- ViT patches image into tokens; cross-attention with text
- Can answer questions, describe scenes, identify text in images
- Chain-of-thought reasoning applies to visual inputs
- Social science: multimodal content analysis at scale

**Instructor Notes:** Show a live demo: upload a political meme or advertisement and ask GPT-4V to analyze its rhetorical strategies. This replaces hours of manual content coding with seconds of automated analysis — but requires the same validity checks as any automated method.

---

### Slide 13: REACT — Reason + Act with Multimodal Perception
**Visual Description:** ReAct loop diagram: Observation (image/text from environment) → Thought (chain-of-thought reasoning) → Action (tool call / API / movement command) → New Observation → loop. Example: agent looking at a street scene → thinks about what to do → calls Google Maps API → moves.

**Bullet Points:**
- Yao et al. 2025: synergizing reasoning and acting
- Perception → reasoning → action as unified loop
- Grounds language model outputs in real-world feedback
- Enables verification: agent can check its own conclusions

**Instructor Notes:** ReAct is the architecture underlying many modern AI assistants. The key social science implication: agents that reason and act can make *mistakes* that compound — an agent that misidentifies a building can make a sequence of wrong inferences. Error propagation in agentic systems is a critical validity concern.

---

### Slide 14: Embodied Task Planning with LLMs
**Visual Description:** Three-layer diagram: Top = Natural Language Goal ("Fetch the red cup from the kitchen shelf"). Middle = LLM planner generating action sequence in code: `navigate_to("kitchen"), grasp("red_cup"), return_to_base()`. Bottom = Robot executing actions with feedback arrows back to planner.

**Bullet Points:**
- LLMs as high-level planners; robot APIs as low-level executors
- Inner monologue: agent reasons about its own actions
- Grounding: mapping language to physical actions requires spatial understanding
- Sim-to-real gap: agents trained in simulation may fail in physical environments

**Instructor Notes:** Connect to social science: virtual embodied agents can explore simulated environments (buildings, cities) and report observations like ethnographers. Physical robots can conduct naturalistic behavioral studies in situ. Both require careful calibration between linguistic and physical capabilities.

---

### Slide 15: Diffusion Models — Generation by Denoising
**Visual Description:** Forward and reverse diffusion process diagram. Forward: clean image → gradually add Gaussian noise over T steps → pure noise. Reverse: train neural network to predict noise at each step → generate image by iteratively denoising from random noise. Loss function shown: `L = E[||ε - ε_θ(x_t, t)||²]`.

**Bullet Points:**
- Forward process: corrupt image with Gaussian noise over T steps
- Reverse process: neural network learns to denoise step-by-step
- Conditioning: text prompt guides denoising direction
- Stable Diffusion: latent diffusion for efficiency

**Instructor Notes:** Diffusion models generate images by literally learning to reverse entropy. From a social science perspective, they're interesting both as tools (generate stimuli for experiments) and as subjects (what images do they generate for "CEO" vs. "janitor"?).

---

### Slide 16: Conditional Generation and Social Bias
**Visual Description:** 3×3 grid of AI-generated images. Row labels: "Doctor", "Nurse", "Engineer". Columns: DALL-E output, Stable Diffusion output, Midjourney output. Each cell shows generated face with demographic annotation below. Bar chart showing % female / % non-white per occupation.

**Bullet Points:**
- Generated images reproduce occupational gender stereotypes
- Bias can be measured: prompt same occupation, count demographics
- Mitigations: prompt engineering, fine-tuning on balanced data
- Implication: AI-generated experimental stimuli may have systematic biases

**Instructor Notes:** This connects directly to Week 5's discussion of bias in training data. Diffusion models are trained on internet images, which over-represent certain demographics in certain roles. Researchers using AI-generated images as stimuli must audit for demographic bias before deploying.

---

### Slide 17: Using Street-Level Imagery for Social Analysis
**Visual Description:** Map of Chicago with color-coded overlay. Left: Google Street View images sampled every 100m. Center: CNN processing each image, predicting "physical disorder index." Right: choropleth map showing spatial distribution overlaid with crime statistics. Scatter plot showing correlation.

**Bullet Points:**
- Niemi et al. 2017: Google Street View + deep learning → neighborhood demographics
- Gebru et al. 2017: car types in street imagery predict income/voting
- Physical environment signals social conditions at city scale
- Enables longitudinal tracking of neighborhood change

**Instructor Notes:** This is a powerful example of ML as a hypothesis generation tool (Ludwig & Mullainathan 2024). The algorithm notices patterns in images that human coders would take years to label. Ask: what validity concerns arise when images become social science data?

---

### Slide 18: Facial Expression — Universal or Cultural?
**Visual Description:** World map with facial expression images sampled from different world regions. Six expression categories shown (Ekman's basic emotions). Bar chart comparing recognition rates across cultures. Caption reference: "Sixteen facial expressions occur in similar contexts worldwide" (Cowen et al. 2020).

**Bullet Points:**
- Ekman (1969): 6 universal basic emotions detectable cross-culturally
- Cowen et al. (2020): 28 distinct expressions occur in similar contexts worldwide
- Culture modulates display rules, not underlying signals
- Deep learning enables large-scale cross-cultural affect research

**Instructor Notes:** Discuss the history: Ekman's universality thesis was controversial for decades. Deep learning on massive cross-cultural datasets (YouTube videos coded for context) provided new empirical traction. This is a model for how computational tools advance long-running debates in social science.

---

### Slide 19: Building a Multimodal Research Pipeline
**Visual Description:** End-to-end pipeline diagram: Data Collection (scrape images + text + audio) → Preprocessing (resize, normalize, transcribe) → Feature Extraction (CNN embeddings, CLIP embeddings, Whisper transcripts) → Fusion (concatenate, cross-attention, or late fusion) → Analysis (classification, clustering, regression) → Interpretation (SHAP, saliency maps, qualitative validation).

**Bullet Points:**
- Multimodal fusion: early (feature-level), late (decision-level), or cross-attention
- CLIP embeddings serve as a common semantic space for image+text
- Saliency maps reveal which pixels drive predictions
- Always validate: show the model's "evidence" to domain experts

**Instructor Notes:** Walk through the pipeline as the architecture for a final project. Students integrating images into their projects should think about where in this pipeline they're intervening. The fusion strategy is a key design choice with significant validity implications.

---

### Slide 20: Hypothesis Generation from Images — Ludwig & Mullainathan 2024
**Visual Description:** Figure from paper showing ML discovering housing features associated with mobility outcomes (upward income mobility in childhood neighborhoods). Left: before ML — researchers hypothesized based on theory. Right: after ML — algorithm surfaces surprising feature importance rankings. Highlighted surprising finding shown.

**Bullet Points:**
- ML can discover features humans didn't think to measure
- "Machine learning as a tool for hypothesis generation" — not just testing
- Discovered: visual cues of neighborhood cohesion predict mobility beyond income
- Process: train model → interpret important features → formulate causal hypotheses → test

**Instructor Notes:** This is a profound methodological shift. We're not just using ML to classify data efficiently — we're using it as a discovery tool that identifies which features of the social world deserve theoretical attention. The risk: overfitting to sample-specific features.

---

### Slide 21: Notebook Demo — CNNs for Social Image Analysis
**Visual Description:** Jupyter notebook screenshot (mock) showing: (1) `torchvision.models.resnet50(pretrained=True)` with final layer replaced; (2) training loop on a custom social images dataset; (3) confusion matrix output; (4) Grad-CAM saliency visualization showing which pixels activated for each prediction.

**Bullet Points:**
- Week_8_Multimodal.ipynb: Modules 1–4
- ResNet-50 fine-tuned on domain-specific image dataset
- Grad-CAM: gradient-weighted class activation mapping
- Qualitative check: do highlighted regions make sociological sense?

**Instructor Notes:** Walk through the notebook's actual code structure. Emphasize Grad-CAM as the interpretability layer — students should always visualize what the model "sees" before trusting its predictions on a new social image dataset.

---

### Slide 22: Audio Pipeline Demo — Transcription and Sentiment
**Visual Description:** Pipeline screenshot: audio file → Whisper API call (`openai.Audio.transcribe`) → transcript text → sentiment model → polarity time series plot showing how sentiment evolves over a speech or interview.

**Bullet Points:**
```python
import openai
transcript = openai.Audio.transcribe(
    model="whisper-1",
    file=open("interview.mp3", "rb")
)
```
- Transcription fidelity: evaluate on ground-truth subset
- Speaker diarization: who said what?
- Downstream: topic modeling, sentiment, argument mining

**Instructor Notes:** Demo this live if possible. The combination of Whisper + sentiment analysis + topic modeling turns unstructured audio archives into analyzable structured datasets. Oral history projects, legislative hearings, and court recordings are immediately tractable.

---

### Slide 23: Vision-Language Agent Demo — GPT-4V Analysis
**Visual Description:** Screenshot of GPT-4V API call with multimodal input. Python code showing base64 image encoding and structured prompt. Example: image of a political rally + prompt "Identify: (1) estimated crowd size, (2) visible demographics, (3) visible signs or symbols, (4) mood/affect of crowd."

**Bullet Points:**
```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "text", "text": "Analyze this political rally..."}
        ]
    }]
)
```
- Structured extraction from images at scale
- Validate: compare to manual coding on 10% of sample
- Limitation: confidently wrong on ambiguous images

**Instructor Notes:** Show a live example. The key social science move is converting this from a one-off description to a systematic coding scheme applied to thousands of images — effectively replacing months of RA labor with hours of API calls, pending validation.

---

### Slide 24: Multimodal Social Simulation
**Visual Description:** Architecture diagram: multimodal agent perceiving both text and images. Left: agent receives image (social media post with meme image) + text caption. Agent uses VLM to understand image, combines with text embedding, then generates a response as a persona (e.g., a politically conservative 45-year-old). Output: simulated social media reply.

**Bullet Points:**
- Digital doubles that see, not just read
- Memes communicate through image+text interaction
- Social media behavior is fundamentally multimodal
- Research question: does visual content change simulated political response?

**Instructor Notes:** Connect to Week 3: our digital doubles until now were text-only. Adding vision changes what social scenarios they can navigate. A meme is fundamentally a multimodal artifact — text-only agents cannot engage with it as humans do.

---

### Slide 25: Embodied AI for Social Science — Virtual Ethnography
**Visual Description:** Screenshot of a 3D virtual environment (like those in Concordia or a game engine). An AI agent avatar navigating a virtual neighborhood, with thought bubbles showing its observations: "Three people gathered near the corner store. Notices broken window. Checks time: 11pm. Assessment: potential social disorder indicator." Log of observations shown.

**Bullet Points:**
- Virtual environments as field sites
- Agents as ethnographers: observe, note, interpret
- Systematic coverage: no sampling fatigue, consistent observation criteria
- Validity question: do virtual neighborhoods mirror real social dynamics?

**Instructor Notes:** This is a speculative but increasingly viable research design. Virtual worlds (games, simulations) already contain rich social behavior. AI ethnographers that can systematically observe and report would dramatically scale qualitative fieldwork.

---

### Slide 26: Multimodal Bias Audit Framework
**Visual Description:** 2×2 matrix. X-axis: "Input Modality" (text vs. image). Y-axis: "Bias Type" (demographic representation vs. stereotyping). Each cell filled with example: text-demographic (training corpus underrepresents minority voices), image-demographic (search results skew white), text-stereotyping (gendered occupational associations), image-stereotyping (generated images of doctors are male).

**Bullet Points:**
- Every modality can carry distinct bias signatures
- Demographic representation bias ≠ stereotyping bias
- Audit checklist: measure both before deploying multimodal agents
- Mitigation: balanced fine-tuning, prompt engineering, post-hoc filtering

**Instructor Notes:** Give students this as a practical audit framework for their final projects. Any project incorporating images, audio, or video needs to systematically check both dimensions of this matrix before drawing conclusions from model outputs.

---

### Slide 27: Evaluation — When Does Multimodal Analysis Work?
**Visual Description:** Recall-precision curves for four tasks: (1) facial expression recognition in laboratory vs. naturalistic video; (2) political sentiment from text only vs. text+image; (3) crowd estimation from aerial photos; (4) building type classification from street view. Each curve with confidence interval showing performance degradation as data moves from lab to naturalistic settings.

**Bullet Points:**
- Lab-to-field degradation: controlled conditions ≠ real-world performance
- Multimodal fusion helps when modalities are complementary
- Fusion hurts when one modality is noisy or misaligned with label
- Always report performance on held-out naturalistic test set

**Instructor Notes:** The key take-away: don't assume that because a model works on a benchmark, it works in your research context. Always evaluate on a small hand-labeled sample from your actual data before applying at scale.

---

### Slide 28: Scope Conditions and Limitations
**Visual Description:** Warning sign graphic with four panels. Panel 1: "Images decontextualized from culture" (a thumbs-up means different things in different countries). Panel 2: "Audio quality varies" (street recordings vs. studio). Panel 3: "Video compression artifacts" (YouTube 360p vs. raw footage). Panel 4: "Temporal drift" (visual norms change; 2010 trained model ≠ 2026).

**Bullet Points:**
- Cultural context: visual meaning is not universal
- Quality degradation: real-world data is messier than training data
- Temporal shift: models trained on past data may not reflect present
- Consent and ethics: whose images are used? For what purpose?

**Instructor Notes:** Every multimodal social science application must reckon with these limitations. The cultural context point is especially important: CLIP was trained predominantly on English-captioned images, which introduces anglophone cultural biases into its visual representations.

---

### Slide 29: Connecting to Final Projects
**Visual Description:** Mind map with "Multimodal Data" at center. Branches to: Political Science (rally images + speeches), Sociology (neighborhood photos + census), Communication (social media posts + memes), Economics (product images + reviews + prices), Public Health (urban environment + health outcomes), Cultural Studies (art + text descriptions + sales).

**Bullet Points:**
- Images: protest imagery, product photos, satellite data, street views
- Audio: interviews, speeches, social media videos (transcribed)
- Video: protest footage, legislative sessions, social interactions
- Fusion: combine with Week 2 text features for richer representations

**Instructor Notes:** Walk through each branch with a concrete research question. Students should be thinking about which modalities are available and relevant for their final projects. Many valuable social datasets are already multimodal — researchers have just historically ignored the non-text components.

---

### Slide 30: Synthesis — The Perceptive Social Agent
**Visual Description:** Full course arc diagram: Week 1 (numeric data) → Week 2 (text) → Week 3 (simulated social actors) → Week 4 (causal inference) → Week 5 (domain adaptation) → Week 6 (interpretable) → Week 7 (learning from feedback) → Week 8 (perceptive, multimodal) → Week 9 (aligned with values). Each week shown as expanding the agent's capabilities.

**Bullet Points:**
- Multimodal agents perceive the world more richly than text-only agents
- But richer perception = richer opportunities for bias amplification
- Social scientists must audit, validate, and interpret multimodal outputs
- Next week: are these agents aligned with human values?

**Instructor Notes:** Close by previewing Week 9. The question for next week is not "can agents perceive and act?" — clearly they can. The question is: "do their perceptions and actions align with human values, and how would we even know?" That is the hardest question in the field.
