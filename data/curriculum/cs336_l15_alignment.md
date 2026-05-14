# CS336 Lecture 15 — RLHF / Alignment

**Source:** Stanford CS336 Spring 2025.

## The Class Thus Far

We've now covered pre-training, which gets you to GPT3. But how do we get to InstructGPT?

Instruction following is a remarkable form of control.

**And what about safety and content moderation?** Deployment to many users requires stronger control over outputs.

**Goal today:** enable better, tighter controls over LM output. Pretraining data isn't quite what we want (but it scales).

Can we collect data of behaviors we do want and train the LM?

1. What does that data look like?
2. How do we best make use of that data?
3. Do we need scale for this?

## Where Today's Lecture Fits In

Standard approach: imitation (SFT) followed by reinforcement ('RL' HF) [Ouyang 2022].

## Part 1: The 'Supervised Finetuning' Part

### What Are the Ingredients in SFT?

- The training data
- The method

### Training Data

Let's talk about two more details about instruction tuning datasets:

1. What's actually inside these datasets?
2. What matters in building 'high performance' instruction tuning data?

**Looking inside some instruction-tuning data:**

Three datasets: Oasst, FLAN, Alpaca.

**FLAN:** Includes tasks like email subject line generation, classification, summarization. Teaches models to follow diverse instruction formats.

**Alpaca:** Instruction-following data generated with GPT-3.5. Examples include health tips, algorithm explanations, coding tasks.

**OpenAssistant:** Multi-turn conversations with nuanced answers, including references.

### What Did We Notice Across the Datasets?

These datasets vary in many ways:

- Length and bullet points (style variations)
- References, other complex knowledge

Less visible, but important aspects: scale, safety.

### Style Variations in Data and Models

Models vary a lot in response length. When evaluating by preferences, style matters. Very strong length effects in both humans and GPT-based evaluations [Wang+ 2023, Dubois+ 2023].

These factors are (mostly) not that relevant for other benchmark performances.

### References, Complex Knowledge, and Factuality

Consider an example from OpenAssistant asking about 'monopsony' in economics with citations.

What is this example teaching the model?

1. Teaching the model about specific references.
2. Teaching the model to output citations when asked to do so.

But by what mechanism? Does the model know about these citations?

### Knowledge Extraction and Alignment

Folklore: fine-tuning a model on 'facts it doesn't know' makes it hallucinate [Schulman 2023, Gekhman 2023].

**Takeaways on knowledge extraction and alignment:**

1. You may not want to fine-tune on tail knowledge, even if that's the LM use case.
2. In principle, 'RL' style correctness feedback could help.
3. Knowledge storage and extraction in LMs is messy and nuanced.

### Safety

LMs are widely deployed to end-users, and need some safety controls. Concerns include misinformation [Goldstein+ 2023] and scams/spam [Kang+ 2023].

**Safety-tuning:** A bit of instruction tuning can drastically change safety profiles. The challenge is really to balance this with over-refusals.

**Safety-tuning with just a little data:** Significant improvements to safety with ~500 samples. Adding 500 Alpaca-style examples makes models follow safety guidelines.

### Putting It Together – SFT Data

1. Instruction fine-tuning (SFT) works best when we are just extracting pretraining behaviors, not adding new ones.
2. Adding (factually correct!) data can sometimes hurt.
3. Small amounts of the right kinds of behavior (safety, instruction-following, style) make a big difference, but there is a long tail that benefits from more data.

### How to Fine-Tune

Just do gradient descent.

In many academic settings: this is basically it. But if you have tons of compute and data and want to scale up instruction tuning:

**Turning instruction tuning into pretraining:** The following (increasingly popular) idea:

1. Pre-train on web/pretraining data.
2. Mix in instruction-tuning data into pre-training.
3. Do an actual (but short) instruction-tuning round.

Lets you scale up instruction tuning without catastrophic forgetting.

**'Midtraining' / 'Two-phase training':** The recipe is common knowledge among many LLM companies (but not documented). Widely used by most models today. Publicized in recent Chinese-derived LMs (MiniCPM, JetMoE).

## Part 2: The 'RL' Part

### From Imitation to Optimization

**Imitation (SFT):**

Fit p̂(y|x) ≈ p*(y|x) for some reference distribution p*(y|x).

- Pure generative modeling perspective.
- Requires samples from reference policy.

**Optimization (RLHF):**

Find p̂(y|x) such that max_p E_p[R(y, x)] for a reward R(y, x).

- Maximize some reward function that we can measure.
- LMs are policies, not a model of some distribution.

### Why Optimize? Costs

Might be easier to get scalar feedback rather than optimal policy.

Even for a tiny and simple 7B model:

| | Base model | Supervised learning | Pairwise feedback | RL | Evaluation |
|---|---|---|---|---|---|
| Compute cost | $300k | $100 | $100 | $100 | $0 |
| Annotation cost | $0 | $25k | $4k | $0 | $50 |

SFT data can be really expensive, and there may be tasks that are much easier for experts to verify than solve. Most frontier model labs spend millions on post-training data.

### Why Optimize? G-V Gap

People don't always write the thing that they prefer in LM outputs [Zhang et al 2023 – "Benchmarking Large Language Models for News Summarization"].

### Overview of RLHF

Three aspects to cover:

- **Data:** How do people collect RLHF data? What are some things to worry about?
- **How do we RLHF?** PPO and DPO.
- **What are some side-effects of RLHF?**

## RLHF Data

What types of pairwise feedback. How do we get (good) pairwise feedback?

**Standard 'pairwise feedback' setup.**

**RLHF and data – InstructGPT guideline:** Scale + Upwork – 40 workers.

### RLHF and Data – Crowdsourcing

Complexities of crowdsourcing:

- Hard to get really high-quality, verifiable annotators.
- Hard to get them to really check correctness.
- Have to be careful about GPT4 use.

**Crowdsourcing ethics:** Data collection at scale can have significant ethical issues.

**Demographics:** The annotator distribution for RLHF can significantly shift its behaviors [Santurkar+ 2023].

**Annotators matter (a lot):** This is true even for many annotators [Hosking, Blunsom, Bartolo 2024].

### RLHF and Data – LM-Generated

GPT4 is a surprisingly good pairwise feedback system:

- Near-perfect rank correlation at the system level.
- Agreement near human inter-annotator levels.

At the lower end of the cost+quality spectrum: AI feedback often used for RLHF. Examples: Ultrafeedback, Zephyr 7B, Tulu3. Used in e.g., OlMo, Zephyr, etc.

**Self-training:** From Constitutional AI [Bai et al]. Models can generate and refine their own feedback.

### RLHF and Style – Length Effects

Length effects are a very significant outcome of RLHF [Chen et al 2024, Singhal et al 2024].

## How Do We RLHF?

We now have a (high quality) pairwise feedback data collection pipeline. How do we adapt the model to make use of pairwise feedback?

- **Part 1: PPO** – the original and very finicky approach (the brief version)
- **Part 2: DPO** – the new, very accessible approach

### PPO in Language Modeling

From InstructGPT. More details and background from Stiennon "Learning to summarize from human feedback."

**PPO – at a conceptual level:**

- Attempt 1: Policy gradients (variances are too high): ∇_θ E_{p_θ}[R(z)] = E_{p_θ}[R(z) ∇_θ log p_θ(z)]
- Attempt 2: TRPO (Linearize the problem around the current policy)
- Attempt 3: PPO (Clip the ratios at some eps)

### Can We Get Rid of PPO?

Some reasonable ideas people thought about:

- Train the model with a control token (SFT on the pairs, prepend [GOOD] to chosen, [BAD] to not chosen).
- Train the model on only preferred output.
- Train a reward model, get LM outputs, train on the preferred output.
- Train a reward model, get 1024 LM outputs, take the best one.

## DPO – RLHF Without Tears?

Try to simplify PPO by:

- Getting rid of the reward model.
- Getting rid of any on-policy stuff (rollouts, outer loops etc).

Instead:

- Take gradient steps on log-loss of good stuff.
- Take negative gradient steps on bad stuff (appropriately weighted).

### DPO – Derivation from the RLHF Formula

Our goal is to optimize. Assume that the policy π is the set of all policies (nonparametric assumption). The maximizer in closed form allows us to solve for the 'implied reward.' (This is the equivalence also used in the Kimi-think paper.)

### DPO Derivation (Part 2)

We can now optimize the implied reward as a reward model via the Stiennon objective. This gives the DPO objective.

**The key steps:**

1. Make a nonparametric assumption (links π_θ and r in closed form).
2. Parametrize reward r via the policy.
3. Optimize the reward using supervised losses (which in turn, optimizes the policy).

Conceptually: this is MLE on the pairwise rewards, under nonparametric assumption + alternative parametrization.

### DPO Updates and Components

In some sense, reduces to "positive gradient on good, negative gradient on bad." Scaled by 'prediction error' of the implied reward model.

**Results:** Compared to PPO implementation, DPO achieves same performance (on simulation) with no pain!

**DPO works.** From Chris Manning: most 'top open-source' RLHF models are DPO'd.

### Variants

Lots of variants, but two of note from the Tulu 3 paper:

- **SimPO** (no reference model)
- **Length normalized DPO**

### But PPO Does Too (and Sometimes Better)

The trickiness of RL-related empirical work: lots of results are highly contingent on the specifics of the experiment setup.

## Things to Watch Out For in RLHF

### Overoptimization / Overfitting on the Reward

Across many different RLHF-style optimizers, optimizing for reward overfits past a point. Holds true for human preference (left), noisy LM preference (mid) but not noiseless LM preference (right).

### Mode Collapse / Entropy

RLHF makes models no longer 'probabilistic models' – no calibration by default.

## Recap of the Lecture

**RLHF recap:**

1. RLHF data collection is (also) hard! Many confounding factors.
2. RLHF algorithms are a bit more complex than SFT – especially PPO.
3. Be mindful of the impact of (over)optimizing for rewards.
