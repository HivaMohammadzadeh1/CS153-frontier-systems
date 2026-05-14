# Tokenization (Area A — Fundamentals)

Tokenization splits raw text into integer ids the model can consume. Byte-pair encoding (BPE) iteratively merges the most frequent symbol pairs, balancing vocabulary size against sequence length. Smaller vocabularies mean longer sequences and slower training; larger vocabularies cost embedding parameters. A misconception is that the tokenizer is interchangeable across models — pretraining data and tokenizer are coupled; swapping the tokenizer usually requires retraining embeddings.
