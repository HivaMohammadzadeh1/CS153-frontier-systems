# CS336 Lecture 9 — Scaling Laws: Basics

**Source:** Stanford CS336 Spring 2025.

## Taking Scaling Seriously

Imagine the following scenario: your friend has given you ten thousand H100s for a month, and asked you to build a good open source LM.

What do you do?

- Put together your infra team and distributed training framework (A2)
- Put together a great pretraining dataset (A4)
- Run a big model (but which one??) ← we are here.

**Scaling isn't easy.** Wide or deep? How many heads? Which nonlinearity?

We could cargo cult things from existing LMs… but how do these get optimized in the first place?

Today: simple, predictive 'laws' for behaviors of LMs.

- Old and unpleasant: tune hyperparameters on big models
- New (over?) optimism: tune on small models, extrapolate to large ones

## Part 1: Scaling Laws – History and Background

- Data scaling as empirical sample complexities
- Initial forays into understanding neural scaling with data

### Sample Complexity and Rates

Theorists have thought about 'scaling' for a long time. These are upper bounds, not actual, realized loss values.

**Earliest (data) scaling law paper: 1993.**

**Log-linear scaling with data [Banko and Brill '01].**

**Early tests of functional forms:** Kolachina et al 2012 – power law relation between data and downstream performance.

**Earliest 'large scale neural' scaling work: Hestness 2017.** Predictable scaling on many tasks (MT, LM, Speech) and hypothesized scaling shape. Very ahead of its time. Topics covered: "Emergence", scaling by compute, speed = accuracy.

## Part 2: Neural (LLM) Scaling Behaviors

1. Data vs performance – "Are there simple rules that determine how data affects performance?"
2. Data vs model size – Do we train on more data or bigger models?
3. Hyper-parameters vs performance – "How should we set hyperparameters on the big model?"

**Scaling laws – power law relationships for many factors.** These scaling laws hold on many different kinds of phenomena, even in non-standard settings (when train ≠ test) [Kaplan+ 2020].

## Data vs Performance

**What's a data scaling law?**

Data scaling laws: simple formula that maps dataset size (n) to error.

What do we expect? Monotonic, logistic-like curves [Hestness+ 2017].

**Data scaling laws for language models:**

Loss and dataset size is linear on a log-log plot ("scale-free" or "power law"). For language modeling, from Kaplan+ 2020.

## Conceptual Foundations of Data Scaling Laws

Q: Why do scaling laws show up?

We know error should be monotone. But why is it a power law / linear in log-log?

A (?): Estimation error naturally decays polynomially.

**Toy example: mean estimation.**

Input: x_1…x_n ~ N(μ, σ²). Task: estimate the average.

What's the error? E[(μ̂ - μ)²] = σ²/n

This is a 'scaling law': log(Error) = -log(n) + 2log(σ). More generally, any polynomial rate 1/n^α is a scaling law.

**Scaling law exponents: an intriguing mystery.** Similar arguments show most 'classical' models (regression, etc) have scaling ~1/n, meaning we should see y = -x + C. But what we find in neural scaling laws (machine translation, speech, language modeling) is very different from classical predictions.

**Detour: scaling laws for (nonparametric) learning.** In d-dimensions, estimation error scales as n^(-1/d). This means scaling slope depends on the intrinsic dimensionality of the data.

**Intrinsic dimensionality theory of data scaling laws [Bahri 2021]:**

1. Scaling laws arise due to polynomial rates of learning (1/n^α).
2. Some argue the slope α is closely connected to the intrinsic dimensionality of the data.

But estimators of intrinsic dimension are sketchy, and this is not airtight.

## Other Data Scaling Laws

**Dataset composition affects performance.** A: Data composition affects the offset, not the slope [Kaplan+ 2021, Hashimoto 2021].

**Scaling laws under data repetition:** In practice, we have finite data – how does repeating examples affect scaling?

- D' = Effective data
- Ud = Unique tokens
- Rd* = Constant
- Rd = Repetition

**Data selection scaling:** Given that repeated data is less valuable, data selection should then be adaptive to scale.

**Recap: data scaling laws.**

- Remarkably linear relationship between log-data size and log-error.
- Holds across domains and models.
- Theory understanding: similar to generalization bounds, mean estimation example.
- Applications: data collection / curation.

## Scaling Laws for Model Engineering

Our motivation: how can we efficiently design huge LMs?

- LSTMs vs Transformers
- Adam vs SGD

How should we allocate our limited resources?

- Train models longer vs train bigger models?
- Collect more data vs get more GPUs?

Scaling laws provide a simple procedure to answer these.

### Hyperparameter Questions

We'll consider some choices in the context of the classic Kaplan scaling paper.

#### 1. Architecture: Transformers vs LSTMs

Q: Are transformers better than LSTMs? Brute force way: spend tens of millions to train a LSTM GPT-3. Scaling law way: run small models, extrapolate [Kaplan+ 2021, Tay et al cross-architecture scaling].

#### 2. Optimizer Choice

What about Adam vs SGD? [Hestness+ 2017].

#### 3. Depth/Width: Number of Layers

- 1 vs 2 layers makes a huge difference.
- More layers have diminishing returns below 10^7 params.

**Not all parameters are made equal.** Embedding layer parameters don't behave the same. Related: recent papers on scaling laws for mixtures of experts.

**Other transformer hypers:** Do hyperparameters like the aspect ratio depend on scale?

#### 4. Batch Size: Critical Batch Size

Batch size has strong diminishing returns past a certain point. Critical batch = min number of examples for target loss / min number of steps for target loss.

The smaller the loss target, the bigger the batch.

**Q: As we increase both compute and model size, how should we scale training?**

- Huge batches, same number of steps, or fixed batches, more steps?
- Good news for data parallel processing.

#### 5. Learning Rates: muP and Scale-Aware LR Choices

If we naively scale up, optimal learning rate depends on scale [Yang et al 2022, Yao et al 2024]. We need scaling-aware initialization and learning rate scaling.

**Caution: scaling behaviors can differ downstream.** Thus far: scaling is predictable and depends mainly on parameters. Catch: downstream scaling can often be much less predictable [Tay et al 2023].

### Some Surprising Takeaways

The effect of hyperparameters on big LMs can be predicted before training!

- Optimizer choice
- Model depth
- Architecture choice

**The scaling law based design procedure:**

1. Train a few smaller models.
2. Establish a scaling law (e.g., Adam vs SGD scaling law).
3. Select optimal hyperparam based on the scaling law prediction.

## One Important Use of Scaling Laws: Data vs Model Size

Q: Do we need more data or bigger models?

Clearly, lots of data is wasted on small models.

**Joint data-model scaling laws:**

From Rosenfeld+ 2020: Error = n^(-α) + m^(-β) + C

From Kaplan+ 2020: Error = m^(-α) + n^(-1) β

Provides surprisingly good fits to model-data joint error. Trading off data size and model size: optimize n^(-α) + m^(-β) + C with your costs.

### Compute Tradeoffs

Q: what about other resources? Compute vs performance?

For a fixed compute budget: big model that's undertrained vs small model that's well trained?

Scaling laws let us navigate this tradeoff [Brown+ 2020, Kaplan+ 2021].

### Caution – 'Optimal' Scaling Laws Are Hard to Get

Rosenfeld, Kaplan both predict relationship of data, model, and performance. Chinchilla [Hoffman et al] argues these fits are quite off. Main difference: accounting for LR schedules.

**Chinchilla in depth – 3 methods:**

The chinchilla authors suggest 3 ways of fitting scaling laws:

1. **Minimum over runs:** Similar to the FLOPS figure on Kaplan – the minimum over the union of all training curves is a power law.
2. **IsoFLOPS:** Pick a range of FLOP budgets, vary the total parameter count, take the min over these convex shapes. The minima form a power law.
3. **Joint fits:** Run a bunch of models on the size-data grid. Use least squares to fit a joint scaling law.

They mostly (minus method 3) suggest similar constants.

**Fun addendum – errors in chinchilla method 3:** Note that method 3 was likely flawed in the original paper. Some authors did data forensics, recovered the raw data, and re-did the fit and got results more consistent with methods 1 and 2 [Besiroglu et al 2024].

### Important Note – Train-Optimal May Not Be What You Want

Chinchilla aims to tell you what gives the best model for fixed training compute. But most of the compute in a real deployment is inference. So we should 'over' train:

- GPT3: 2 tokens/param
- Chinchilla: 20 tokens/param
- LLaMA 65B: 22 tokens/param
- Llama 2 70B: 29 tokens/param
- Mistral 7B: 110 tokens/param
- Llama 3 70B: 215 tokens/param

The more usage we expect, the more it becomes worth it to pay the upfront cost.

Methods like IsoFLOPS are pretty easy to execute, and have been replicated for other model types (e.g., diffusion models) [Gulrajani+ 2023].

## Scaling Laws for Models and Compute

Log-linearity extends to model parameters and compute! Lets us set the following based on small models:

- Pick optimizer
- Pick architecture and model sizes

Also lets us make smart resource tradeoffs:

- Big models vs more data?

## Recap: Scaling Laws – Surprising and Useful!

- Data scaling: understand how data affects models, clean theory.
- Model scaling: dramatically reduce costs for training.
- Scaling as prediction: understand what problems can be 'brute forced'.
