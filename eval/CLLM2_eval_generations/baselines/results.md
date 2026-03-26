## Fixed Sliding Window Size:

// create a table
| Window Size | Avg iters / token | Avg tokens / iter |
| --- | --- | --- |
| 32 | 0.2374 ｜ 4.212 |
| 64 | 0.2351 | 4.253 |
| 128 | 0.2353 | 4.249 |
| 256 | 0.2364 | 4.230 |
| 512 | 0.2364 | 4.230 |


## Oracle Sliding Window Replay: window = k1 + k2 + 4
=== Diffusion Decoding Profiling — EOS-only ===
Examples (eos): 162 / 162   Total wall time: 189.1203s
Avg new tokens / prompt: 206.9074
Avg calls / prompt: 53.1049
Avg iterations / call: 0.9766
Avg iterations / token: 0.2566
Avg toks/sec: 185.3233

## Oracle Sliding Window Replay: window = k1 + k2 + 24
=== Diffusion Decoding Profiling — EOS-only ===
Examples (eos): 162 / 162   Total wall time: 176.9897s
Avg new tokens / prompt: 208.7963
Avg calls / prompt: 49.8580
Avg iterations / call: 0.9749
Avg iterations / token: 0.2388
Avg toks/sec: 200.4383

# Fixed Window 64
=== Diffusion Decoding Profiling — EOS-only ===
Examples (eos): 162 / 162   Total wall time: 188.7613s
Avg new tokens / prompt: 208.1358
Avg calls / prompt: 4.7469
Avg iterations / call: 10.5268
Avg iterations / token: 0.2525
Avg toks/sec: 188.0731

# Sliding Window 32
=== Diffusion Decoding Profiling — EOS-only ===
Examples (eos): 162 / 162   Total wall time: 175.4090s
Avg new tokens / prompt: 208.0988
Avg calls / prompt: 49.5679
Avg iterations / call: 0.9747
Avg iterations / token: 0.2374
Avg toks/sec: 202.1719

# Sliding Window 128
=== Diffusion Decoding Profiling — EOS-only ===
Examples (eos): 162 / 162   Total wall time: 198.2260s
Avg new tokens / prompt: 209.0062
Avg calls / prompt: 49.1358
Avg iterations / call: 0.9746
Avg iterations / token: 0.2353
Avg toks/sec: 179.4263