**Evaluation**: given a **fixed model**, how "**good**" is it?

Takeaways

- There is no one true evaluation; choose the evaluation depending on what you're trying to measure.

- Always look at the individual instances and the predictions.

- There are many aspects to consider: capabilities, safety, costs, realism.

- Clearly state the rules of the game (methods versus models/systems).

## Benchmark scores

Recent language models are evaluated on similar, but not entirely identical, benchmarks (MMLU, MATH, etc.).

What are these benchmarks?

What do these numbers mean?

Pay close attention to the costs!

Maybe a model is good if people choose to use it (and pay for it)...

## Vibes

A crisis...

You might think evaluation is a mechanical process (take existing model, throw prompts at it, average some numbers)...

Actually, evaluation is a profound and rich topic...

...and it determines the future of language models.

What's the point of evaluation?

There is no one true evaluation; it depends on what question you're trying to answer.

1. User or company wants to make a purchase decision (model A or model B) for their use case (e.g., customer service chatbots).

2. Researchers want to measure the raw capabilities of a model (e.g., intelligence).

3. We want to understand the benefits + harms of a model (for business and policy reasons).

4. Model developers want to get feedback to improve the model.

In each case, there is an abstract **goal** that needs to be translated into a concrete evaluation.

Framework

1. What are the **inputs**?

2. How do **call** the language model?

3. How do you evaluate the **outputs**?

4. How to **interpret** the results?

What are the inputs?

1. What use cases are **covered**?

2. Do we have representation of **difficult** inputs in the tail?

3. Are the inputs **adapted** to the model (e.g., multi-turn)?

How do you call the language model?

1. How do you prompt the language model?

2. Does the language model use chain-of-thought, tools, RAG, etc.?

3. Are we evaluating the language model or an agentic system (model developer wants former, user wants latter)?

How do you evaluate the outputs?

1. Are the reference outputs used for evaluation error-free?

2. What metrics do you use (e.g., pass@k)?

3. How do you factor in cost (e.g., inference + training)?

4. How do you factor in asymmetric errors (e.g., hallucinations in a medical setting)?

5. How do you handle open-ended generation (no ground truth)?

How do you inteprret the metrics?

1. How do you interpret a number (e.g., 91%) - is it ready for deployment?

2. How do we assess generalization in the face of train-test overlap?

3. Are we evaluating the final model or the method?

Summary: lots of questions to think through when doing evaluation

Recall: that a language model is a probability distribution **p(x)** over sequences of tokens.

Perplexity (1/p(D))^(1/|D|) measures whether p assigns high probability to some dataset D.

In pre-training, you minimize perplexity on the training set.

The obvious thing is to measure perplexity on the test set.

Standard datasets: Penn Treebank (WSJ), WikiText-103 (Wikipedia), One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news)

Papers trained on a dataset (training split) and evaluated on the same dataset (test split)

Pure CNNs+LSTMs on the One Billion Word Benchmark (perplexity 51.3 -> 30.0) 

GPT-2 trained on WebText (40GB text, websites linked from Reddit), zero-shot on standard datasets

This is out-of-distribution evaluation (but idea is that training covers a lot)

Works better on small datasets (transfer is helpful), but not larger datasets (1BW)

Since GPT-2 and GPT-3, language modeling papers have shifted more towards downstream task accuracy.

But reasons why perplexity is still useful:

- Smoother than downstream task accuracy (for fitting scaling laws)

- Is universal (why we use it for training) whereas task accuracy might miss some nuances

- Note: can measure conditional perplexity on downstream task too (used for scaling laws) 

Warning (if you're running a leaderboard): evaluator needs to trust the language model

For task accuracy, can just take output generated from a blackbox model and compute the desired metrics

For perplexity, need LM to generate probabilities and trust that they sum to 1 (even worse with UNKs back in the day)

The perplexity maximalist view:

- Your true distribution is t, model is p

- Best possible perplexity is H(t) obtained iff p = t

- If have t, then solve all the tasks

- So by pushing down on perplexity, will eventually reach AGI

- Caveat: this might not be the most efficient way to get there (pushing down on parts of the distribution that don't matter)

Things that are spiritually perplexity:

Similar idea: cloze tasks like LAMBADA 

HellaSwag 

### Massive Multitask Language Understanding (MMLU)

- 57 subjects (e.g., math, US history, law, morality), multiple-choice

- "collected by graduate and undergraduate students from freely available sources online"

- Really about testing knowledge, not language understanding

- Evaluated on GPT-3 using few-shot prompting

### MMLU-Pro

- Removed noisy/trivial questions from MMLU

- Expanded 4 choices to 10 choices

- Evaluated using chain of thought (gives model more of a chance)

- Accuracy of models drop by 16% to 33% (not as saturated)

### Graduate-Level Google-Proof Q&A (GPQA)

- Questions written by 61 PhD contractors from Upwork

- PhD experts achieve 65% accuracy

- Non-experts achieve 34% over 30 minutes with access to Google

- GPT-4 achieves 39%

### Humanity's Last Exam

- 2500 questions: multimodal, many subjects, multiple-choice + short-answer

- Awarded $500K prize pool + co-authorship to question creators

- Filtered by frontier LLMs, multiple stages of review

So far, we've been evaluating on fairly structured tasks.

Instruction following (as popularized by ChatGPT): just follow the instructions.

Challenge: how to evaluate an open-ended response?

### Chatbot Arena

How it works:

- Random person from the Internet types in prompt

- They get response from two random (anonymized) models

- They rate which one is better

- ELO scores are computed based on the pairwise comparisons

- Features: live (not static) inputs, can accomodate new models

### Instruction-Following Eval (IFEval)

- Add simple synthetic constraints to instructions

- Constraints can be automatically verified, but not the semantics of the response

- Fairly simple instructions, constraints are a bit artificial

### AlpacaEval

- 805 instructions from various sources

- Metric: win rate against GPT-4 preview as judged by GPT-4 preview (potential bias)

### WildBench

- Sourced 1024 examples from 1M human-chatbot conversations

- Uses GPT-4 turbo as a judge with a checklist (like CoT for judging) + GPT-4 as a judge

- Well-correlated (0.95) with Chatbot Arena (seems to be the de facto sanity check for benchmarks)

Consider tasks that require tool use (e.g., running code) and iterating over a period of time

Agent = language model + agent scaffolding (logic for deciding how to use the LM)

### SWEBench

- 2294 tasks across 12 Python repositories

- Given codebase + issue description, submit a PR

- Evaluation metric: unit tests

### CyBench

- 40 Capture the Flag (CTF) tasks

- Use first-solve time as a measure of difficulty

### MLEBench

- 75 Kaggle competitions (require training models, processing data, etc.)

All of the tasks so far require linguistic and world knowledge

Can we isolate reasoning from knowledge?

Arguably, reasoning captures a more pure form of intelligence (isn't just about memorizing facts)

Introduced in 2019 by Francois Chollet

ARC-AGI-1

ARC-AGI-2: harder

What does safety mean for AI?

### HarmBench

- Based on 510 harmful behaviors that violate laws or norms

### AIR-Bench

- Based on regulatory frameworks and company policies

- Taxonomized into 314 risk categories, 5694 prompts

### Jailbreaking

- Language models are trained to refuse harmful instructions

- Greedy Coordinate Gradient (GCG) automatically optimizes prompts to bypass safety 

- Transfers from open-weight models (Llama) to closed models (GPT-4)

### Pre-deployment testing

- US Safety Institute + UK AI Safety Institute working together

- Company gives safety institutes access to model before release (currently voluntary)

- Safety institutes run evaluations and produce a report to company

### But what is safety?

- Many aspects of safety are strongly contextual (politics, law, social norms - which vary across countries)

- Naively, one might think safety is about refusal and is at odds with capability, but there's more...

- Hallucinations in a medical setting makes systems more capable and more safe

Two aspects of a model that reduce safety: capabilities + propensity

- A system could be capable of doing something, but refuse to do it

- For API models, propensity matters

- For open weight models, capability matters (since can easily fine-tune safety away)

**Dual-use**: capable cybersecurity agents (do well on CyBench) can be used to hack into a system or to do penetration testing

CyBench is used by the safety institute as a safety evaluation, but is it really a capability evaluation?

Language models are used heavily in practice:

However, most existing benchmarks (e.g., MMLU) are far away from real-world use.

Live traffic from real people contain garbage, that's not always what we want either.

Two types of prompts:

1. Quizzing: User knows the answer and trying to test the system (think standardized exams).

2. Asking: User doesn't know the answer is trying to use the system to get it.

Asking is more realistic and produces value for the user.

### Clio (Anthropic)

- Use language models to analyze real user data

- Share general patterns of what people are asking

### MedHELM

- Previous medical benchmarks were based on standardized exams

- 121 clinical tasks sourced from 29 clinicians, mixture of private and public datasets

Unfortunately, realism and privacy are sometimes at odds with each other.

How do we know our evaluations are valid?

### Train-test overlap

- Machine learning 101: don't train on your test set

- Pre-foundation models (ImageNet, SQuAD): well-defined train-test splits

- Nowadays: train on the Internet and don't tell people about your data

Route 1: try to infer train-test overlap from model

- Exploit exchangeability of data points

Route 2: encourage reporting norms (e.g., people report confidence intervals)

- Model providers should report train-test overlap 

### Dataset quality

- Fixed up SWE-Bench to produce SWE-Bench Verified 

- Create Platinum versions of benchmarks

What are we even evaluating?

In other words, what are the rules of a game?

Pre-foundation models, we evaluated **methods** (standardized train-test splits).

Today, we're evaluating **models/systems** (anything goes).

There are some exceptions...

nanogpt speedrun: fixed data, compute time to get to a particular validation loss

DataComp-LM: given a raw dataset, get the best accuracy using standard training pipeline 

Evaluating methods encourage algorithmic innovation from researchers.

Evaluating models/systems is useful for downstream users.

Either way, we need to define the rules of the game!
