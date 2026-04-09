# Mean Prompt Confidence Scoring

Custom extension to vLLM that computes GPU-side mean prompt confidence and returns it via the OpenAI-compatible completions API. Used by the RSA routing system to make confidence-based routing decisions between draft and target models.

## What it computes

```
mean_prompt_confidence = -mean(top-k logprobs across prompt positions)
```

For each prompt token position, the model computes the top-k log probabilities. The mean of all these values is negated to produce a positive confidence score. Higher values indicate the model is more "confident" about the prompt.

## Quick start

### Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B \
    --confidence-micro-chunk 131072
```

`--confidence-micro-chunk` (default: 131072) controls how many tokens are processed per GPU micro-chunk in confidence-only mode. Smaller values reduce peak GPU memory; larger values reduce kernel launch overhead.

### Client

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    prompt="What is 2+2?",
    max_tokens=1,
    extra_body={
        "prompt_confidence_only": True,
        "confidence_start_idx": 0,
    },
)

# Access the confidence score
choice = response.choices[0]
print(choice.mean_prompt_confidence)
```

## API reference

### Request fields (added to `/v1/completions`)

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt_confidence_only` | `bool` | `false` | When true, compute only the mean confidence scalar. Skips building per-token prompt logprob dictionaries for lower memory usage. |
| `confidence_start_idx` | `int \| list[int] \| null` | `null` | Only accumulate tokens at absolute position >= this index. Useful for computing confidence over just the response portion of a prompt. Automatically enables `prompt_confidence_only`. When a list is provided, each element applies to the corresponding prompt in a batched request. |

### Response fields (added to each choice)

| Field | Type | Description |
|---|---|---|
| `mean_prompt_confidence` | `float \| null` | The computed confidence score. Only present when `prompt_confidence_only` is true or `confidence_start_idx` is set. |

### Engine argument

| Argument | Type | Default | Description |
|---|---|---|---|
| `--confidence-micro-chunk` | `int` | `131072` | Tokens per micro-chunk in confidence-only GPU path. |

## Batched requests

When sending multiple prompts in a single completions request, `confidence_start_idx` can be a list to set a different start index per prompt:

```python
response = client.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    prompt=["prompt one", "prompt two", "prompt three"],
    max_tokens=1,
    extra_body={
        "confidence_start_idx": [5, 10, 3],
    },
)
```

The list length must match the number of prompts.

## How it works

### Data flow

```
CompletionRequest (API)
  -> SamplingParams (prompt_confidence_only, confidence_start_idx)
    -> GPUModelRunner._get_prompt_logprobs_dict()
      -> ModelRunnerOutput.mean_prompt_confidence_dict
        -> Scheduler -> EngineCoreOutput.mean_prompt_confidence
          -> LogprobsProcessor -> OutputProcessor
            -> RequestOutput.mean_prompt_confidence
              -> CompletionResponseChoice (API response)
```

### Two GPU paths

1. **Confidence-only path** (`prompt_confidence_only=true`): Processes prompt tokens in micro-chunks, computes `log_softmax` + `topk` in-place, and accumulates sum/count without allocating full logprob tensors. Peak memory is proportional to `confidence_micro_chunk * vocab_size` rather than `prompt_length * vocab_size`.

2. **Standard path with confidence** (when `prompt_logprobs` is also requested): The normal prompt logprobs computation runs as usual, and confidence is accumulated as a side effect from the top-k logprob values.

Both paths support chunked prefill: accumulators persist across chunks and finalize when prefill completes.

### Auto-configuration

- Setting `confidence_start_idx` automatically enables `prompt_confidence_only`.
- If `prompt_confidence_only` is true but `prompt_logprobs` is not set, it defaults to `prompt_logprobs=20` (top-20 logprobs used for the confidence mean).
