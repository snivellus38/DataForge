# DataForge 2026: Pathway Track

## Explain the Frontier
Al research moves faster than the material used to teach it. Ideas often appear across many papers before there is a clear, reusable way to learn them. The NeurIPS 2026 Education Track was built around this gap: it asks contributors to distill emerging Al ideas into accessible resources such as interactive demos, notebooks, code, short videos, slides, lecture notes, animations, and visual explanations, along with the intended audience, prerequisites, learning objectives, and recent papers that show the concept is active now.

This Pathway track brings that challenge to DataForge. Choose one approved concept. Build an explainer that makes one difficult technical idea genuinely understandable. Connect it to Dragon Hatchling (BDH) or BDH-CQ. The learner should not just read or watch. The learner should interact with the system, change something meaningful, and observe the concept behaving.

Every submission must also contain a substantial, technically correct section on BDH or BDH-CQ.

The goal is simple: make one frontier Al concept click. If you're aiming for the star, you might as well use this opportunity to use this as a submission for NeurIPS 2026 Education Track (check their deadline here). The deadline is close but India's top student builders have a habit of making ambitious timelines look possible.

## About Pathway
Pathway is a frontier Al company working on architectures and models for memory, latent reasoning, continual learning, and long-horizon reasoning. Its central research program is Dragon Hatchling, or BDH: a brain-inspired Post Transformer architecture family. BDH-CQ is a later system in the same family that lets the model learn from demonstrations and reason without a written chain of thought.

## Approved topics
These topics were pulled from concept patterns across recent NeurIPS papers and related Al venues: ideas that are emerging, resurging, or repeatedly used by researchers right now. A valid concept can be a new technical advance, or an older idea returned in an important new form, but a generic introduction to a well-established topic isn't enough. Explain why it matters now, and show its use in recent research.

Choose one topic from the list below. A team may combine two closely related topics only when the final artifact still has one central claim and one coherent learning journey.

### Memory and learning

**State-Space Models (SSMS)**
A long sequence is compressed into an evolving state instead of being preserved token by token. Note: do not classify BDH as an SSM in the Mamba sense; BDH-GPU is a separate GPU-friendly formulation built from ReLU-low-rank transformations with linear attention.

**Linear Attention**
Attention reorganised so memory can be updated incrementally instead of comparing every pair of tokens. Note: BDH-CQ's paper relates its contextual memory to attention, fast-weight memory and linear-attention views of contextual association, with the special case where state accumulates additively per demonstration.

**Comparing Linear Attention Variants**
a survey of the recent linear attention variants developed and adopted in prominent LLM architectures, and what each one trades away.

**Synaptic Plasticity as Short-Term Memory**
recent activity temporarily strengthens connections, turning the network's wiring into working memory. This is BDH's mechanism in its conceptual neuron-synapse model, where attention becomes synaptic memory through Hebbian writes.

**Test-Time Adaptation**
Optimization versus Context, how a system acquires an unseen task's rule before answering it. HRM and TRM take the optimization route on ARC: demonstration pairs are augmented into training samples, each puzzle gets a learned identity embedding, predictions are voted over augmentations, and so an unseen task requires a backward pass before evaluation. BDH-CQ is the counterexample, no evaluation-task demonstrations in training and no parameter updates at inference, with adaptation happening in recurrent state rather than in weights.

**Associative Memory and Fast Weights**
Rapidly changing connections store relationships and retrieve them when a similar cue returns.

**In-Context Learning with Recurrent Memory**
A fixed-size state is repeatedly read, updated and carried forward as new information arrives.

**Continual and Online Learning in Language Models**
How deployed models adapt without erasing earlier capabilities. Study catastrophic forgetting and the stability-plasticity trade-off, distinguishing durable parameter learning from temporary adaptation through context or recurrent memory, with BDH as a recent case study in architecture-native session memory.

**Key-Value Caching, Limitations, and Alternate Approaches**
How Transformers retain previous tokens through a growing key-value cache, and why this creates memory and context-length constraints. Compare eviction, compression, retrieval, linear attention, state-space and fixed-size recurrent alternatives. Include BDH as an integrated approach whose key architectural differences must be understood together, not as isolated modifications. See also Adrian Kosowski's comment on the Hugging Face paper page.

**Parametric Memory in LLMs**
Where memory lives: external stores, specialised memory layers, fixed "slow" parameters, or parameter-shaped "fast" memory updated during inference. Use BDH to explore how memory, adaptation and reasoning can share one computational fabric, including the scientific problem of consolidating useful fast state into durable slow weights.

### Reasoning and generalisation

**Recurrent Latent-Space Reasoning**
Thinking by repeatedly refining a hidden state instead of narrating every intermediate step.

**Inference-Time Scaling**
How additional inference compute can be allocated through more generated reasoning tokens or more recurrent updates in latent space. Compare their accuracy, cost, latency and observability trade-offs, using BDH-CQ's low-, medium- and high-effort results as evidence that latent test-time compute can improve performance without verbalising intermediate reasoning.

**Skill Acquisition from Demonstrations**
How a system infers and applies a new rule from sparse demonstrations, as tested in ARC-style tasks, rather than merely reusing a capability learned during training. Make sure to use the BDH CQ example.

**State Interpretability**
Inspecting what a model remembers and computes now-not only what its fixed parameters contain.

**Demonstration Coverage and Extrapolation**
How far a model can generalise beyond the complexity, length or structure shown in its examples. Relates to topics such as generalization over time explained through BDH materials.

**Cost-Accuracy Pareto Frontiers (particularly timely)**
A system lies on the cost-accuracy Pareto frontier when no alternative is both cheaper and more accurate. Define intelligence per dollar as verified task performance divided by cost per task. Compare BDH-CQ with other reasoning systems and examine how architectural choices-including latent reasoning, sparse computation and test-time scaling-are shifting this frontier.

**Evaluating Mathematical and Abstract Reasoning**
How different benchmarks probe different forms of reasoning: for example, GSM8K tests multi-step grade-school mathematics, while ARC-AGI tests the acquisition and application of unfamiliar abstract transformations. Compare what success and failure on each benchmark actually establish, without treating either as a complete measure of general reasoning. Include published BDH or BDH-CQ evidence where available.

**Alternatives to Chain-of-Thought Reasoning**
How models can reason without serialising every intermediate step into natural-language tokens. Compare recurrent latent computation, continuous-thought models, iterative hidden-state refinement, programmatic reasoning and other alternatives, using BDH-CQ as a case study among other approaches. Examine their trade-offs in accuracy, inference cost, latency, supervision and observability.

**Long-Horizon Evolving State**
How models maintain, update and selectively forget a compact internal state across extended sequences or tasks. Compare growing token histories with recurrent memory, examining persistence, interference, state capacity and recovery. Use BDH's evolving associative state and reported long-context results as case studies, without equating within-session memory with durable cross-session learning.

**Reasoning Under Complex Constraints**
How models solve problems in which many interacting rules must remain simultaneously satisfied and locally plausible choices can create global contradictions. Compare representative approaches to representing, updating and verifying constraints, using BDH's reported Sudoku Extreme result as a case study. Show how controlled constraint tasks make reasoning dynamics, generalisation and failure patterns measurable, and connect these insights to the broader study of logical intelligence.

**Games, Puzzles and Simulated Environments as Al Evaluation Testbeds**
How controlled environments help evaluate spatial reasoning, planning, memory, rule learning, causal prediction and world-model formation. Draw on arcade-style game benchmarks, unfamiliar-game tests, Sudoku, ARC-AGI and other verifiable environments to examine whether models learn reusable dynamics or exploit task-specific patterns and scaffolding. Include BDH's reported Sudoku and BDH-CQ's ARC-AGI results as recent case studies within the wider research landscape of game-based Al evaluation.

### Brain-inspired computation, sparsity and interpretability

**Sparse Non-Negative Activations:**
only a small set of units activates at any moment, and those activations are non-negative. In reported BDH runs roughly 5% of neurons are active, and activity varies with predictability rather than following a fixed sparsity budget.

**Monosemantic Synapses**
a connection that responds selectively to one recognisable concept rather than many unrelated ideas. In BDH this is reported at the level of synapses encoding semantic concepts rather than tokens. Determining which activations, connections, parameters or computational pathways can remain inactive.

**Scale-Free and Heavy-Tailed Neural Connectivity**
most units have few connections while a small number become highly connected hubs, producing a power-law degree distribution in trained BDH models.

**Local Neural Computation**
Complex global behaviour emerging from many components that apply only local interaction rules.

### Architectural directions

**Post-Transformer Architectures (particularly timely)**
Sequence models that rethink attention, recurrence, state and memory rather than merely scaling Transformers. Explain their design motivations; compare leading architectures, organizations and production efforts; identify where they outperform or lag Transformers; and assess their maturity out of 10 and largest remaining gaps. For BDH and BDH-CQ, cover recent developments with appropriate caveats, including reported early pretraining-scaling experiments from 1B to 600B parameters, their GPU-oriented formulation, and development and deployment integration with Amazon SageMaker HyperPod and existing AWS workflows. Keep the explainer scientific, comparative and visually coherent.

**Dragon Hatchling (BDH)**
a brain-inspired Post-Transformer architecture in which reasoning and memory share one computational fabric, with attention reformulated as synaptic memory that updates as the model reads.

**The Equations of Reasoning**
How attention, memory and iterative inference can emerge from a compact set of local rules governing neurons and synapses. Use BDH's formalism to connect microscopic graph dynamics with macroscopic reasoning, model scaling and longer inference. See Pathway's blog on The Equations of Reasoning.

## Your Mission
Build one self-contained educational experience for a clearly stated audience. A strong submission lets a learner do all of the following:
* Understand one precise technical claim.
* Manipulate a real concept variable.
* Observe an immediate, meaningful consequence.
* Compare the system output with ground truth, a reference, or a clearly defined expectation.
* Explain the concept back in their own words.
* Understand where the concept appears in BDH or BDH-CQ, or how it differs from the corresponding mechanism in BDH.
* Recognize at least one limitation, failure case, or common misconception.

Do not try to explain an entire field. A focused artifact that creates one reliable moment of understanding beats a dashboard with many views and no central lesson.

## The one-sentence claim
Bare minimum, before writing code, write one falsifiable sentence that your explainer will teach.
Examples:
* A fixed-shape recurrent state can process a sequence of unbounded duration without allocating a new memory slot for every token, but it can still forget through interference.
* A model can perform repeated latent computation without producing a verbal chain of thought.

The interaction must let the learner reproduce, test, or challenge the sentence. If the artifact cannot in principle reveal that the sentence is wrong, the claim is probably too vague.

## The BDH module (must-have)
Somewhere in the journey, connect your concept to BDH or BDH-CQ. Don't tack it on at the end. Weave it into the flow, with its own real learning objective, and ground it in something concrete: an equation, a diagram, a live experiment, or a clearly labeled precomputed result.
Say plainly which system you mean, and why the concept shows up there. Point to what's actually changing, whether that's activity, recurrent state, memory, or trained parameters etc. Work from primary sources, such as the Dragon Hatchling paper or the BDH-CQ technical report, rather than secondhand summaries. Teams are not expected to run an unavailable BDH or BDH-CQ checkpoint or independently reproduce unpublished systems. A valid BDH module may use published equations, architecture diagrams, public source code, documented evaluations or clearly labelled precomputed results. Any toy model or independent reimplementation must be identified as such and must not be presented as an official BDH model.

## What teams may build
Your submission may take the form of:
* An interactive visual essay.
* A public web demonstration.
* An explorable model or dataset.
* A runnable notebook with interactive controls.
* A small educational code library with a working demonstration.
* An interactive simulation.
* A visual walkthrough driven by real computation.
* A combination of these formats.

At least one part must let the learner change a meaningful input, parameter, state, example or assumption and observe the consequence.
Short videos, diagrams, animations, slides or lecture notes may support the resource, but a static artifact by itself does not meet the requirements of this track.
Expensive experiments may be performed beforehand and presented through clearly labelled precomputed results. Illustrative animations are welcome when they are identified as illustrations rather than presented as live model behaviour.

## Design standards for the explainer
* **One claim.** Every chart, control, animation, and paragraph should serve the central claim. Cut anything that doesn't.
* **Substrate.** The concept should behave inside the artifact before the learner even acts. A real tiny model, a replay of a real run, or a live toy system all count. A scripted animation doesn't.
* **Visible state.** Shrink the model, sequence, or dataset until the learner can see the variables that matter.
* **Truth beside estimate.** Put the expected answer next to the model output. The gap is often the lesson.
* **Catchy.** Skip the blank canvas and the Run button. Open with a preset already running.
* **Fast feedback.** Controls should respond in under a second. Precompute expensive results and ship them as data.
* **Few controls.** Every control should map to one real variable. Cut decorative sliders.
* **Guide, then sandbox.** Walk the learner through the insight first. Open it up after.
* **No hidden limits.** Caps and approximations are fine. State them.

## Examples worth looking at
These resources demonstrate clarity, interaction and technical honesty. They are references, not templates that teams must be constrained by. Feel free to review the examples most relevant to your idea, identify themes and implementation patterns worth drawing inspiration from, and use them for your own planning or as reference context for Al tools. They can help shape your explainer if a pedagogical approach does not come to mind within your given time constraints.

* **Model internals and complete systems:** Transformer Explainer, LLM Visualization, CraftGPT, BertViz, The Annotated Transformer, Spreadsheets Are All You Need, AttentionViz, Tiktokenizer and CNN Explainer.
* **Technical concepts made manipulable:** TensorFlow Playground, GAN Lab, Colorful Vectors, How to Use t-SNE Effectively, An Interactive Tutorial on Numerical Optimization, MLU-Explain: Neural Networks, Seeing Theory, Why Momentum Really Works, Grokking, Double Descent, Neural Networks and Deep Learning, Chapter 4, ConvNetJS and Neural Networks, Manifolds and Topology.
* **Interpretability:** Mapping LLMs with Sparse Autoencoders, Neuronpedia and On the Biology of a Large Language Model.
* **Design guidance and further examples:** Communicating with Interactive Articles, Piotr Migdal's Interactive Machine Learning list, Hugging Face's collection of interactive tools and VISXAI.
* **Primary BDH resources:** The Dragon Hatchling paper, BDH-CQ technical report, Why BDH Uses a Brain-Inspired Architecture, From Attention to Synapses: Deriving BDH, The Equations of Reasoning and the official (toy) BDH implementation.

For broader format guidance, see the NeurIPS 2026 Education Track and its Call for Educational Resources.

Teams that combine a knack for technical development, teaching, Al-assisted coding, reading or publishing Al research, and frontend or interaction design are likely to have an edge. These are useful strengths, not mandatory roles or credentials. One person may cover several areas, and a small team with a narrow, well-executed idea can outperform a larger team attempting to explain too much.

Mentorship from senior students, alumni, researchers or people with publication experience is encouraged. Mentors may advise, review and challenge the work, but their involvement must be disclosed and the registered team must build, understand and defend the submission.

## Al assistance and technical ownership
Al-assisted coding, writing, design, and research are allowed, but teams must understand and defend every component. All Al-generated, reused, or forked work must be disclosed in the README. Forks should add a meaningful contribution. The same applies to the blog, be ready to explain every claim, sentence, and citation.

## How judging works
Total score: 100 points, across seven criteria.
* **Technical correctness and depth (25 points):** Judges assess whether the concept, equations, code, measurements, visual mappings, and limitations are accurate. Incorrect claims receive a major penalty.
* **Technical ownership and live defense (15 points):** Judges assess whether the team understands every major component, can trace the system, can predict the result of changes, and can distinguish real behavior from precomputation or animation.
* **Learning effectiveness (15 points):** Judges assess the one-sentence claim, audience definition, prerequisites, learning objectives, guided narrative, sixty-second test, and whether a learner can explain the concept after using the artifact.
* **Interactive substrate and honesty (15 points):** Judges assess whether the concept truly behaves in the artifact, whether controls map to concept variables, whether the important state is visible, whether truth is shown beside estimate, and whether feedback is fast.
* **BDH or BDH CQ integration and evidence discipline (10 points):** Judges assess whether the BDH module is substantive, technically correct, connected to the selected concept, properly sourced, and explicit about evidence level and limitations.
* **Craft, robustness, accessibility, and provenance (10 points):** Judges assess visual clarity, writing, mobile usability, loading behavior, link stability, source quality, reproducibility, credits, and licenses.
* **One-page concept summary (10 points):** Judges assess whether the summary is technically correct, self-contained and accessible to an average data scientist. They will also assess the quality of its architectural comparison, evidence classification, strengths-and-limitations analysis, competitive context, maturity assessment, primary sourcing, and treatment of BDH and BDH-CQ.

## What to submit
Submit one package containing an explorable learning artifact and its supporting materials.
The artifact itself can be an interactive visual essay, a public web demo, a runnable notebook with interactive controls, a small educational code library, or a hybrid of these. A static article, poster, slide deck, or recorded video by itself is not enough. At least one element must let the learner alter an input, parameter, state, example, or assumption and observe the result. Any animation must be connected to real computation or clearly labeled as a teaching simplification.

The package must include:
* A public artifact URL that opens without sign-in.
* A public source code repository.
* The blog as a pdf file (Talked about below)
* A complete README.
* Clear setup instructions for any notebook or local component.
* At least three recent primary papers from 2022 to 2026 that use, extend, test, or rely on the selected concept, with citations beside technical claims.
* A source and license record for code, data, weights, graphics, fonts, and reused components.
* An Al assistance, code, data, asset, and license disclosure.

The README must explain the claim, the intended learner and prerequisites, the learning objectives, the architecture of the artifact, the role of every major component, which parts are live, precomputed, synthetic, or animated, how to reproduce the results, and credits and licenses.

## Accuracy
Do not overclaim or misrepresent any research. Verify claims against primary sources rather than memory or secondhand summaries, and disclose whether a claim is formal or publicly demonstrated.

## The one-page concept summary
Every team must submit a readable one-page concept summary as a PDF. Approximately 500-950 words are recommended.
This is not a blog post, promotional article, project diary, extended abstract or collection of paper summaries. It should work as a self-contained, authoritative briefing for an average data scientist encountering the concept and your submission for the first time. A good test of your summary would be to use it as a prompt and see the kind of outputs it gives.

A useful quality test is to give the summary, without additional context, to a person or capable Al system. It should enable them to explain your central claim, the underlying mechanism, the roles of BDH and BDH-CQ, the evidence supporting your explanation, and its limitations without introducing major inaccuracies. In other words, the summary should be precise enough to preserve important technical distinctions and accessible enough to make the reader curious to learn more.

The summary could situate the concept rather than explain it in isolation. It may:
* Explain the problem or design pressure that motivated the concept and how its mechanism differs from a conventional Transformer or the nearest established alternative.
* State what the concept changes technically, why that change might matter, and what trade-offs it introduces.
* Identify two or three representative architectures, systems or research efforts in the area and compare them on dimensions that genuinely matter, such as accuracy, memory use, inference cost, latency, adaptability, interpretability or deployability.
* Explain where the concept has demonstrated an advantage over incumbents-and where it still performs worse, has weaker evidence or remains untested.
* Ground the discussion in at least one notable evaluation, independent reproduction, real deployment or significant partnership. Label the evidence correctly: a benchmark is not a deployment, a commercial partnership is not an independent evaluation, and a result reported by its developer is not the same as an external reproduction.
* Explain the roles of BDH and BDH-CQ accurately and naturally. Their relevance need not be equal. If one has no direct role in the selected concept, say so rather than inventing a connection.
* Identify the most important limitation, unanswered question or missing piece of evidence.
* Cite the primary sources beside the claims they support and tell the reader where to continue learning.

A compact comparison table is encouraged when it communicates the landscape more clearly than prose.

## No Al slop
The one-page summary will be judged strictly for information density and intellectual ownership. Avoid padded sentences, undefined buzzwords, unsupported claims, invented references or numbers, lists of papers without synthesis, mechanism-free prose, repeated adjectives and a BDH mention added only to satisfy the rules.

Every sentence should contribute a definition, mechanism, piece of evidence, limitation or necessary connection. Teams must be able to explain and defend every sentence and citation.

The summary has succeeded if an average data scientist can read it once, explain the concept and the submission's central contribution, ask an informed follow-up question, and know where to continue learning.

## Weak, strong, and exceptional submissions
* **Weak:** generic overviews, paper-summarizing chatbots, static decks, animations passed off as real computation, unchanged forks, a bolted-on BDH mention, unsourced claims, or code the team can't explain.
* **Strong:** one clear claim, a real substrate, honest labeling, a BDH module that teaches, and a disclosed limitation.
* **Exceptional:** a reusable substrate, a claim a learner can reproduce in under a minute, or a resource people keep using after the hackathon.

---

## 🔗 Mentioned Links & Resources

While explicit URLs were not provided in the original text, the following resources, links, and papers are specifically mentioned and referenced as reading/interactive materials:

**Events & Guidelines**
* NeurIPS 2026 Education Track (Call for Educational Resources)

**Primary BDH Resources**
* The Dragon Hatchling paper
* BDH-CQ technical report
* *Why BDH Uses a Brain-Inspired Architecture*
* *From Attention to Synapses: Deriving BDH*
* *The Equations of Reasoning* (Pathway's blog)
* The official (toy) BDH implementation

**Other References & Examples**
* Adrian Kosowski's comment on the Hugging Face paper page (regarding Key-Value Caching)
* **Model internals:** Transformer Explainer, LLM Visualization, CraftGPT, BertViz, The Annotated Transformer, Spreadsheets Are All You Need, AttentionViz, Tiktokenizer, CNN Explainer
* **Manipulable concepts:** TensorFlow Playground, GAN Lab, Colorful Vectors, How to Use t-SNE Effectively, An Interactive Tutorial on Numerical Optimization, MLU-Explain: Neural Networks, Seeing Theory, Why Momentum Really Works, Grokking, Double Descent, Neural Networks and Deep Learning (Chapter 4), ConvNetJS, Neural Networks, Manifolds and Topology
* **Interpretability:** Mapping LLMs with Sparse Autoencoders, Neuronpedia, On the Biology of a Large Language Model
* **Design guidance:** Communicating with Interactive Articles, Piotr Migdal's Interactive Machine Learning list, Hugging Face's collection of interactive tools, VISXAI
