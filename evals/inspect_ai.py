from typing import Any, Optional

from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState

TASK_PROMPT = "Your task is: {description}. You must go to {homepage} and obtain the password for the game. To prove you have completed the task you must obtain and report the password for the game. If you do not have the password, you have not managed to complete the task."
WEBGAMES_BASE_URL = "https://webgames.convergence.ai"


def _record_to_sample(record: dict[str, Any]) -> Sample:
    id = record["id"]
    title = record["title"]
    description = record["description"]
    password = record["password"]
    path = record["path"]
    homepage = f"{WEBGAMES_BASE_URL}/{path}"

    task_input = TASK_PROMPT.format(description=description, homepage=homepage)

    return Sample(
        input=task_input,
        target=password,
        metadata={
            "id": id,
            "title": title,
            "path": path,
            "homepage": homepage,
            "description": description,
        },
    )


def get_webgames_dataset(limit: Optional[int] = None, shuffle: bool = False) -> Dataset:
    return hf_dataset(
        "convergence-ai/webgames",
        split="train",
        sample_fields=_record_to_sample,
        limit=limit,
        shuffle=shuffle,
    )


@scorer(metrics=[accuracy(), stderr()])
def webgames_scorer():
    async def score(state: TaskState, target: Target):
        answer = state.output.completion
        correct = target.text in answer
        return Score(value=CORRECT if correct else INCORRECT, answer=answer)

    return score
