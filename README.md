# WebGames

This is a collection of challenges for general-purpose web-browsing AI agents.

They're designed to be:

- easy for humans to complete
- hard for AI agents to complete
- fast and simple to run
  - just client-side state and a single-page JavaScript app
- easy to evaluate
  - each task provides a unique password on successful completion

_Read the annoucement blog on the Convergence website: [https://convergence.ai/introducing-webgames/](https://convergence.ai/introducing-webgames/)_

## Try it now

🎮 [webgames.convergence.ai](https://webgames.convergence.ai)

## Run locally

```
cd webgames
pnpm install
pnpm run dev
```

## Download tasks

Tasks are available as a dataset on [Hugging Face](https://huggingface.co/datasets/convergence-ai/webgames).

Alternatively, you can download them from the webgames website:

1. Go to [webgames.convergence.ai?showDownloads=true](https://webgames.convergence.ai?showDownloads=true)
2. Click the download buttons in the top-right corner (csv or jsonl available)
3. Verify your agent solutions using `solution in messages[-1]` or equivalent, or use the Inspect AI eval scaffolding in the eval folder.
