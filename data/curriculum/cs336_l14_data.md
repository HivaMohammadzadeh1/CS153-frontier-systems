Last lecture: overview of datasets used for training language models

- Live service (GitHub) → dump/crawl (GH Archive) → processed data (The Stack)

- Processing: HTML to text, language/quality/toxicity filtering, deduplication

This lecture: deep dive into the mechanics

- Algorithms for filtering (e.g., classifiers)

- Applications of filtering (e.g., language, quality, toxicity)

- Deduplication (e.g., Bloom filters, MinHash, LSH)

### Summary

- Algorithmic tools: n-gram models (KenLM), classifiers (fastText), importance resampling (DSIR)

- Applications: language identification, quality filtering, toxicity filtering

- Deduplication: hashing scales to large datasets for fuzzy matching

- Now you have the tools (mechanics), just have to spend time with data (intuitions)

Algorithmic building block:

- Given some **target data** T and lots of **raw data** R, find subset T' of R similar to T.

Desiderata for filtering algorithm:

- Generalize from the target data (want T and T' to be different)

- Extremely fast (have to run it on R, which is huge)

Survey paper on data selection 

**n-gram model with Kneser-Ney smoothing** 

- KenLM: fast implementation originally for machine translation 

- Common language model used for data filtering

- Extremely simple / fast - just count and normalize

### Concepts

Maximum likelihood estimation of n-gram language model:

- n = 3: p(in | the cat) = count(the cat in) / count(the cat)

Problem: sparse counts (count of many n-grams is 0 for large n)

Solution: Use Kneser-Ney smoothing to handle unseen n-grams 

- p(in | the cat) depends on p(in | cat) too

### CCNet

- Items are paragraphs of text

- Sort paragraphs by increasing perplexity

- Keep the top 1/3

- Was used in LLaMA

Summary: Kneser-Ney n-gram language models (with KenLM implementation) is fast but crude

fastText classifier 

- Task: text classification (e.g., sentiment classification)

- Goal was to train a fast classifier for text classification

- They found it was as good as much slower neural network classifiers

### Baseline: bag of words (not what they did)

Problem: V*K parameters (could be huge)

### fastText classifier: bag of word embeddings

Only H*(V + K) parameters

Implementation:

- Parallelized, asynchronous SGD

- Learning rate: linear interpolation from [some number] to 0 

### Bag of n-grams

Problem: number of bigrams can get large (and also be unbounded)

Solution: hashing trick

- For quality filtering, we have K = 2 classes (good versus bad)

- In that case, fastText is just a linear classifier (H = K = 2)

In general, can use any classifier (e.g., BERT, Llama), it's just slower

Data Selection for Language Models via Importance Resampling (DSIR) 

Setup:

- Target dataset D_p (small)

- Proposal (raw) dataset D_q (large)

Take 1:

- Fit target distribution p to D_p

- Fit proposal distribution q to D_q

- Do importance resampling with p, q, and raw samples D_q

Problem: target data D_p is too small to estimate a good model

Take 2: use hashed n-grams

Result: DSIR slightly better than heuristic classification (fastText) on the [GLUE](https://gluebenchmark.com/) benchmark

Comparison with fastText:

- Modeling distributions is a more principled approach capturing diversity

- Similar computation complexity

- Both can be improved by better modeling

Setup:

- Target distribution p (want samples from here)

- Proposal distribution q (have samples from here)

Implementations: KenLM, fastText, DSIR

### General framework

Given target T and raw R, find subset of R similar to T

1. Estimate some model based on R and T and derive a scoring function

2. Keep examples in R based on their score

### Instantiations of the framework

Generative model of T (KenLM):

1. score(x) = p_T(x)

2. Keep examples x with score(x) >= threshold (stochastically)

Discriminative classifier (fastText):

1. score(x) = p(T | x)

2. Keep examples x with score(x) >= threshold (stochastically)

Importance resampling (DSIR):

1. score(x) = p_T(x) / p_R(x)

2. Resample examples x with probability proportional to score(x)

The same data filtering machinery can be used for different filtering tasks.

Language identification: find text of a specific language (e.g., English)

Why not just go multilingual?

- Data: difficult to do curation / processing of high-quality data in any given language

- Compute: in computed-limited regime, less compute/tokens dedicated to any given language

Models differ on multilinguality:

- English was only 30% of BLOOM (was undertrained), English performance suffered 

- Most frontier models (GPT-4, Claude, Gemini, Llama, Qwen) are heavily multilingual (sufficiently trained)

fastText language identification 

- Off-the-shelf classifier

- Supports 176 languages

- Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)

Example: Dolma keeps pages with p(English) >= 0.5 

Caveats:

- Difficult for short sequences

- Difficult for low-resource languages

- Could accidentally filter out dialects of English

- Hard for similar languages (Malay and Indonesian)

- Ill-defined for code-switching (e.g., Spanish + English)

OpenMathText 

- Goal: curate large corpus of mathematical text from CommonCrawl

- Use rules to filter (e.g., contains latex commands)

- KenLM trained on ProofPile, keep if perplexity < 15000

- Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math

Result: produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data

- Some deliberately do not use model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)

- Some use model-based filtering (GPT-3, LLaMA, DCLM) [becoming the norm]

**GPT-3** 

- Positives: samples from {Wikipedia, WebText2, Books1, Books2}

- Negatives: samples from CommonCrawl

Train linear classifier based on word features 

Keep documents stochastically based on score

** LLaMA/RedPajama** 

- Positives: samples from pages **referenced** by Wikipedia

- Negatives: samples from CommonCrawl

- Keep documents that are classified positive

**phi-1** 

Philosophy: really high quality data (textbooks) to train a small model (1.5B)

Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data

Train random forest classifier on T using output embedding from pretrained codegen model

Select data from R that is classified positive by the classifier

Result on [HumanEval](https://huggingface.co/datasets/openai_humaneval):

- Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)

- Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better!

Toxicity filtering in Dolma 

Dataset: Jigsaw Toxic Comments dataset (2018) 

- Project goal: help people have better discussions online 

- Data: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}

Trained 2 fastText classifiers

- hate: positive = {unlabeled, obscene}, negative = all else

- NSFW: positive = {obscene}, negative = all else

Two types of duplicates:

- Exact duplicates (mirror sites, GitHub forks) 

- Near duplicates: same text differing by a few tokens

Examples of near duplicates:

- Terms of service and licenses 

- Formulaic writing (copy/pasted or generated from a template) 

- Minor formatting differences in copy/pasting

Product description repeated 61,036 times in C4

'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!

Deduplication training data makes language models better 

- Train more efficiently (because have fewer tokens)

- Avoid memorization (can mitigate copyright, privacy concerns)

Design space:

1. What is an item (sentence, paragraph, document)?

2. How to match (exact match, existence of common subitem, fraction of common subitems)?

3. What action to take (remove all, remove all but one)?

Key challenge:

- Deduplication is fundamentally about comparing items to other items

- Need linear time algorithms to scale

- Hash function h maps item to a hash value (integer or string)

- Hash value much smaller than item

- Hash collision: h(x) = h(y) for x ≠ y

Tradeoff between efficiency and collision resistance 

- Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin)

- DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables)

We will use MurmurHash:

**Simple example**

1. Item: string

2. How to match: exact match

3. Action: remove all but one

- Pro: simple, clear semantics, high precision

- Con: does not deduplicate near duplicates

- This code is written in a MapReduce way, can easily parallelize and scale

**C4** 

1. Item: 3-sentence spans

2. How to match: use exact match

3. Action: remove all but one

Warning: when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent

Goal: efficient, approximate data structure for testing set membership

Features of Bloom filters

- Memory efficient

- Can update, but can't delete

- If return 'no', definitely 'no'

- If return 'yes', most likely 'yes', but small probability of 'no'

- Can drive the false positive rate down exponentially with more time/compute

First, make the range of hash function small (small number of bins).

Problem: false positives for small bins

Naive solution: increase the number of bins

Error probability is O(1/num_bins), decreases polynomially with memory

Better solution: use more hash functions

Reduced the false positive rate!

Assume independence of hash functions and items 

Consider a test input (not in the set) that would hash into a given test bin (say, i).

Now consider putting items into the Bloom filter and seeing if it hits i.

Optimal value of k (given fixed m / n ratio) [results in f ~ 0.5]

Resulting false positive rate (improved)

Tradeoff between compute (k), memory (m), and false positive rate (f) 

Example: Dolma

- Set false positive rate to 1e-15

- Perform on items = paragraphs

Let's now look at approximate set membership.

First we need a similarity measure.

### Jaccard similarity

Definition: Jaccard(A, B) = |A intersect B| / |A union B|

Definition: two documents are **near duplicates** if their Jaccard similarity >= threshold

Algorithmic challenge: find near duplicates in linear time

### MinHash

MinHash: a random hash function h so that Pr[h(A) = h(B)] = Jaccard(A, B)

Normally, you want different items to hash to different hashes

...but here, you want collision probability to depend on similarity

Characteristic matrix representation:

item | A | B

1    | 1 | 1

2    | 1 | 1

3    | 1 | 1

4    | 1 | 0

5    | 0 | 1

Random hash function induces a permutation over items

Look at which item is first in A and which item is first in B.

Each item has the same probability as being first (min)

- If 1, 2, 3 is first, then first in A = first in B.

- If 4, 5 is first, then first in A ≠ first in B.

Now we can hash our items, but a collision doesn't tell us Jaccard(A, B) > threshold.

Locality sensitive hashing (LSH) 

Suppose we hash examples just one MinHash function

P[A and B collide] = Jaccard(A, B)

On average, more similar items will collide, but very stochastic...

Goal: have A and B collide if Jaccard(A, B) > threshold

We have to somehow sharpen the probabilities...

Solution: use n hash functions

Break up into b bands of r hash functions each (n = b * r)

Hash functions:

h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12

Key: A and B collide if for *some* band, *all* its hash functions return same value

As we will see, the and-or structure of the bands sharpens the threshold

Given Jaccard(A, B), what is the probability that A and B collide?

**Example**

Increasing r sharpens the threshold and moves the curve to the right (harder to match)

Increasing b moves the curve to the left (easier to match)

Example setting 

: n = 9000, b = 20, r = 450

What is the threshold (where the phase transition happens)?

Probability that a fixed band matches:

Probability that A and B collide (≈ 1-1/e):
