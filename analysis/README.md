# Analysis WebGames results

## Results

- gemini-2.5-flash-preview-04-17: ../evals/browseruse_webgames/logs/2025-05-12T19-44-10+01-00_webgames-browser-agent-eval_aHJs4FXzT6hA3tZZLBot8j.eval
- gemini-2.5-pro-preview-05-06: ../evals/browseruse_webgames/logs/2025-05-12T20-41-55+01-00_webgames-browser-agent-eval_hbXfgRUr9pdjfojUwmUjN2.eval
- claude-3-7-sonnet-20250219: ../evals/browseruse_webgames/logs/2025-05-12T23-11-50+01-00_webgames-browser-agent-eval_2SvDSm5nKjeDyWvYKSTNrv.eval
- gpt-4o: ../evals/browseruse_webgames/logs/2025-01-22T10-48-46+00-00_webvoyager-gin-eval_ATD8hU4Vr5mQpusAzt72Sb.eval
- gpt-4o-mini: ../evals/browseruse_webgames/logs/2025-05-13T10-28-00+01-00_webgames-browser-

## Example metadata keys and values

key value
path: buttons-easy
title: Button megastar (Easy)
description: Click the correct item on the page to reveal the password. Fewer items to click.
icon: ð
tags: buttonclickeasy
difficulty: easy
variant: easy
base_task: buttons

The important keys are base_task, variant (easy, base, hard).

## WebGames categories

in webgames_categories.csv

```csv
game,technical_fluency,realtime_responsiveness,adversarial_resistance,cognitive_abilities,visual_comprehension
date,1,,,,
buttons,0.7,,0.2,,0.1
click-cubed,,0.9,,,0.1
... // 53 games, 153 variants total.
```

No value == 0.
A weighting of how much the game contributes to the category.

## Analysis

We have the following dimensions:

- model
- task difficulty
- task category weighting
- whether the model passed the task or not
