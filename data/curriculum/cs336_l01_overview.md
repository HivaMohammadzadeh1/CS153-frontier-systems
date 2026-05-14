Next time: PyTorch building blocks, resource accounting

## CS336: Language Models From Scratch (Spring 2025)

This is the second offering of CS336.

Stanford edition has grown by 50%.

Lectures will be posted on YouTube and be made available to the whole world.

## Why did we make this course?

Let's ask GPT-4 

Problem: researchers are becoming **disconnected** from the underlying technology.

8 years ago, researchers would implement and train their own models.

6 years ago, researchers would download a model (e.g., BERT) and fine-tune it.

Today, researchers just prompt a proprietary model (e.g., GPT-4/Claude/Gemini).

Moving up levels of abstractions boosts productivity, but

- These abstractions are leaky (in contrast to programming languages or operating systems).

- There is still fundamental research to be done that require tearing up the stack.

**Full understanding** of this technology is necessary for **fundamental research**.

This course: **understanding via building**

But there's one small problem...

## The industrialization of language models

GPT-4 supposedly has 1.8T parameters. 

GPT-4 supposedly cost $100M to train. 

xAI builds cluster with 200,000 H100s to train Grok. 

Stargate (OpenAI, NVIDIA, Oracle) invests $500B over 4 years. 

Also, there are no public details on how frontier models are built.

From the GPT-4 technical report 

:

## More is different

Frontier models are out of reach for us.

But building small language models (<1B parameters in this class) might not be representative of large language models.

Example 1: fraction of FLOPs spent in attention versus MLP changes with scale. 

Example 2: emergence of behavior with scale 

## What can we learn in this class that transfers to frontier models?

There are three types of knowledge:

- **Mechanics**: how things work (what a Transformer is, how model parallelism leverages GPUs)

- **Mindset**: squeezing the most out of the hardware, taking scale seriously (scaling laws)

- **Intuitions**: which data and modeling decisions yield good accuracy

We can teach mechanics and mindset (these do transfer).

We can only partially teach intuitions (do not necessarily transfer across scales).

## Intuitions? 🤷

Some design decisions are simply not (yet) justifiable and just come from experimentation.

Example: Noam Shazeer paper that introduced SwiGLU 

## The bitter lesson

Wrong interpretation: scale is all that matters, algorithms don't matter.

Right interpretation: algorithms that scale is what matters.

### accuracy = efficiency x resources

In fact, efficiency is way more important at larger scale (can't afford to be wasteful).

 showed 44x algorithmic efficiency on ImageNet between 2012 and 2019

Framing: what is the best model one can build given a certain compute and data budget?

In other words, **maximize efficiency**!

## Pre-neural (before 2010s)

- Language model to measure the entropy of English 

- Lots of work on n-gram language models (for machine translation, speech recognition) 

## Neural ingredients (2010s)

- First neural language model 

- Sequence-to-sequence modeling (for machine translation) 

- Adam optimizer 

- Attention mechanism (for machine translation) 

- Transformer architecture (for machine translation) 

- Mixture of experts 

- Model parallelism 

## Early foundation models (late 2010s)

- ELMo: pretraining with LSTMs, fine-tuning helps tasks 

- BERT: pretraining with Transformer, fine-tuning helps tasks 

- Google's T5 (11B): cast everything as text-to-text 

## Embracing scaling, more closed

- OpenAI's GPT-2 (1.5B): fluent text, first signs of zero-shot, staged release 

- Scaling laws: provide hope / predictability for scaling 

- OpenAI's GPT-3 (175B): in-context learning, closed 

- Google's PaLM (540B): massive scale, undertrained 

- DeepMind's Chinchilla (70B): compute-optimal scaling laws 

## Open models

- EleutherAI's open datasets (The Pile) and models (GPT-J) 

- Meta's OPT (175B): GPT-3 replication, lots of hardware issues 

- Hugging Face / BigScience's BLOOM: focused on data sourcing 

- Meta's Llama models 

- Alibaba's Qwen models 

- DeepSeek's models 

- AI2's OLMo 2 

## Levels of openness

- Closed models (e.g., GPT-4o): API access only 

- Open-weight models (e.g., DeepSeek): weights available, paper with architecture details, some training details, no data details 

- Open-source models (e.g., OLMo): weights and data available, paper with most details (but not necessarily the rationale, failed experiments) 

## Today's frontier models

- OpenAI's o3 

- Anthropic's Claude Sonnet 3.7 

- xAI's Grok 3 

- Google's Gemini 2.5 

- Meta's Llama 3.3 

- DeepSeek's r1 

- Alibaba's Qwen 2.5 Max 

- Tencent's Hunyuan-T1 

This is an *executable lecture*, a program whose execution delivers the content of a lecture.

Executable lectures make it possible to:

- view and run code (since everything is code!),

- see the hierarchical structure of the lecture, and

- jump to definitions and concepts: 

All information online: 

This is a 5-unit class.

Comment from Spring 2024 course evaluation: *The entire assignment was approximately the same amount of work as all 5 assignments from CS 224n plus the final project. And that's just the first homework assignment.*

## Why you should take this course

- You have an obsessive need to understand how things work.

- You want to build up your research engineering muscles.

## Why you should not take this course

- You actually want to get research done this quarter.<br>(Talk to your advisor.)

- You are interested in learning about the hottest new techniques in AI (e.g., multimodality, RAG, etc.).<br>(You should take a seminar class for that.)

- You want to get good results on your own application domain.<br>(You should just prompt or fine-tune an existing model.)

## How you can follow along at home

- All lecture materials and assignments will be posted online, so feel free to follow on your own.

- Lectures are recorded via [CGOE, formally SCPD](https://cgoe.stanford.edu/) and be made available on YouTube (with some lag).

- We plan to offer this class again next year.

## Assignments

- 5 assignments (basics, systems, scaling laws, data, alignment).

- No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness.

- Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and speed).

- Leaderboard for some assignments (minimize perplexity given training budget).

- AI tools (e.g., CoPilot, Cursor) can take away from learning, so use at your own risk.

## Cluster

- Thanks to Together AI for providing a compute cluster. 🙏

- Please read [the guide](https://docs.google.com/document/d/1BSSig7zInyjDKcbNGftVxubiHlwJ-ZqahQewIzBmBOo/edit) on how to use the cluster.

- Start your assignments early, since the cluster will fill up close to the deadline!

## It's all about efficiency

Resources: data + hardware (compute, memory, communication bandwidth)

How do you train the best model given a fixed set of resources?

Example: given a Common Crawl dump and 32 H100s for 2 weeks, what should you do?

Design decisions:

## Overview of the course

## Efficiency drives design decisions

Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware.

- Data processing: avoid wasting precious compute updating on bad / irrelevant data

- Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures.

- Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)

- Training: we can get away with a single epoch!

- Scaling laws: use less compute on smaller models to do hyperparameter tuning

- Alignment: if tune model more to desired use cases, require smaller base models

Tomorrow, we will become data-constrained...

Goal: get a basic version of the full pipeline working

Components: tokenization, model architecture, training

## Tokenization

Tokenizers convert between strings and sequences of integers (tokens)

Intuition: break up string into popular segments

This course: Byte-Pair Encoding (BPE) tokenizer 

Tokenizer-free approaches: 

Use bytes directly, promising, but have not yet been scaled up to the frontier.

## Architecture

Starting point: original Transformer 

Variants:

- Activation functions: ReLU, SwiGLU 

- Positional encodings: sinusoidal, RoPE 

- Normalization: LayerNorm, RMSNorm 

- Placement of normalization: pre-norm versus post-norm 

- MLP: dense, mixture of experts 

- Attention: full, sliding window, linear 

- Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA) 

- State-space models: Hyena 

## Training

- Optimizer (e.g., AdamW, Muon, SOAP) 

- Learning rate schedule (e.g., cosine, WSD) 

- Batch size (e..g, critical batch size) 

- Regularization (e.g., dropout, weight decay)

- Hyperparameters (number of heads, hidden dimension): grid search

## Assignment 1

- Implement BPE tokenizer

- Implement Transformer, cross-entropy loss, AdamW optimizer, training loop

- Train on TinyStories and OpenWebText

- Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100 

Goal: squeeze the most out of the hardware

Components: kernels, parallelism, inference

## Kernels

What a GPU (A100) looks like:

Analogy: warehouse : DRAM :: factory : SRAM

Trick: organize computation to maximize utilization of GPUs by minimizing data movement

Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens

## Parallelism

What if we have multiple GPUs (8 A100s)?

Data movement between GPUs is even slower, but same 'minimize data movement' principle holds

Use collective operations (e.g., gather, reduce, all-reduce)

Shard (parameters, activations, gradients, optimizer states) across GPUs

How to split computation: {data,tensor,pipeline,sequence} parallelism

## Inference

Goal: generate tokens given a prompt (needed to actually use models!)

Inference is also needed for reinforcement learning, test-time compute, evaluation

Globally, inference compute (every use) exceeds training compute (one-time cost)

Two phases: prefill and decode

Prefill (similar to training): tokens are given, can process all at once (compute-bound)

Decode: need to generate one token at a time (memory-bound)

Methods to speed up decoding:

- Use cheaper model (via model pruning, quantization, distillation)

- Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)

- Systems optimizations: KV caching, batching

## Assignment 2

- Implement a fused RMSNorm kernel in Triton

- Implement distributed data parallel training

- Implement optimizer state sharding

- Benchmark and profile the implementations

Goal: do experiments at small scale, predict hyperparameters/loss at large scale

Question: given a FLOPs budget ($C$), use a bigger model ($N$) or train on more tokens ($D$)?

Compute-optimal scaling laws: 

TL;DR: $D^* = 20 N^*$ (e.g., 1.4B parameter model should be trained on 28B tokens)

But this doesn't take into account inference costs!

## Assignment 3

- We define a training API (hyperparameters -> loss) based on previous runs

- Submit "training jobs" (under a FLOPs budget) and gather data points

- Fit a scaling law to the data points

- Submit predictions for scaled up hyperparameters

- Leaderboard: minimize loss given FLOPs budget

Question: What capabilities do we want the model to have?

Multilingual? Code? Math?

## Evaluation

- Perplexity: textbook evaluation for language models

- Standardized testing (e.g., MMLU, HellaSwag, GSM8K)

- Instruction following (e.g., AlpacaEval, IFEval, WildBench)

- Scaling test-time compute: chain-of-thought, ensembling

- LM-as-a-judge: evaluate generative tasks

- Full system: RAG, agents

## Data curation

- Data does not just fall from the sky.

- Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.

- Appeal to fair use to train on copyright data? 

- Might have to license data (e.g., Google with Reddit data) 

- Formats: HTML, PDF, directories (not text!)

## Data processing

- Transformation: convert HTML/PDF to text (preserve content, some structure, rewriting)

- Filtering: keep high quality data, remove harmful content (via classifiers)

- Deduplication: save compute, avoid memorization; use Bloom filters or MinHash

## Assignment 4

- Convert Common Crawl HTML to text

- Train classifiers to filter for quality and harmful content

- Deduplication using MinHash

- Leaderboard: minimize perplexity given token budget

It's a wasteland out there!  Need to really process the data.

So far, a **base model** is raw potential, very good at completing the next token.

Alignment makes the model actually useful.

Goals of alignment:

- Get the language model to follow instructions

- Tune the style (format, length, tone, etc.)

- Incorporate safety (e.g., refusals to answer harmful questions)

Two phases:

## Assignment 5

- Implement supervised fine-tuning

- Implement Direct Preference Optimization (DPO)

- Implement Group Relative Preference Optimization (GRPO)

## Supervised finetuning (SFT)

Instruction data: (prompt, response) pairs

Data often involves human annotation.

Intuition: base model already has the skills, just need few examples to surface them. 

Supervised learning: fine-tune model to maximize p(response | prompt).

Now we have a preliminary instruction following model.

Let's make it better without expensive annotation.

## Preference data

Data: generate multiple responses using model (e.g., [A, B]) to a given prompt.

User provides preferences (e.g., A < B or A > B).

## Verifiers

- Formal verifiers (e.g., for code, math)

- Learned verifiers: train against an LM-as-a-judge

## Algorithms

- Proximal Policy Optimization (PPO) from reinforcement learning 

- Direct Policy Optimization (DPO): for preference data, simpler 

- Group Relative Preference Optimization (GRPO): remove value function 

This unit was inspired by Andrej Karpathy's video on tokenization; check it out! 

## Summary

- Tokenizer: strings <-> tokens (indices)

- Character-based, byte-based, word-based tokenization highly suboptimal

- BPE is an effective heuristic that looks at corpus statistics

- Tokenization is a necessary evil, maybe one day we'll just do it from bytes...

Raw text is generally represented as Unicode strings.

A language model places a probability distribution over sequences of tokens (usually represented by integer indices).

So we need a procedure that *encodes* strings into tokens.

We also need a procedure that *decodes* tokens back into strings.

A 

 is a class that implements the encode and decode methods.

The **vocabulary size** is number of possible tokens (integers).

To get a feel for how tokenizers work, play with this 

## Observations

- A word and its preceding space are part of the same token (e.g., " world").

- A word at the beginning and in the middle are represented differently (e.g., "hello hello").

- Numbers are tokenized into every few digits.

Here's the GPT-2 tokenizer from OpenAI (tiktoken) in action.

Check that encode() and decode() roundtrip:

## Character-based tokenization

A Unicode string is a sequence of Unicode characters.

Each character can be converted into a code point (integer) via `ord`.

It can be converted back via `chr`.

Now let's build a `Tokenizer` and make sure it round-trips:

There are approximately 150K Unicode characters. 

Problem 1: this is a very large vocabulary.

Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.

## Byte-based tokenization

Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.

The most common Unicode encoding is 

Some Unicode characters are represented by one byte:

Others take multiple bytes:

Now let's build a `Tokenizer` and make sure it round-trips:

The vocabulary is nice and small: a byte can represent 256 values.

What about the compression rate?

The compression ratio is terrible, which means the sequences will be too long.

Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...

## Word-based tokenization

Another approach (closer to what was done classically in NLP) is to split strings into words.

This regular expression keeps all alphanumeric characters together (words).

Here is a fancier version:

To turn this into a `Tokenizer`, we need to map these segments into integers.

Then, we can build a mapping from each segment into an integer.

But there are problems:

- The number of words is huge (like for Unicode characters).

- Many words are rare and the model won't learn much about them.

- This doesn't obviously provide a fixed vocabulary size.

New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.

## Byte Pair Encoding (BPE)

The BPE algorithm was introduced by Philip Gage in 1994 for data compression. 

It was adapted to NLP for neural machine translation. 

(Previously, papers had been using word-based tokenization.)

BPE was then used by GPT-2. 

Basic idea: *train* the tokenizer on raw text to automatically determine the vocabulary.

Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.

The GPT-2 paper used word-based tokenization to break up the text into inital segments and run the original BPE algorithm on each segment.

Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.

## Training the tokenizer

## Using the tokenizer

Now, given a new text, we can encode it.

In Assignment 1, you will go beyond this in the following ways:

- encode() currently loops over all merges. Only loop over merges that matter.

- Detect and preserve special tokens (e.g., <|endoftext|>).

- Use pre-tokenization (e.g., the GPT-2 tokenizer regex).

- Try to make the implementation as fast as possible.

Start with the list of bytes of `string`.

Count the number of occurrences of each pair of tokens

Find the most common pair.

Merge that pair.
