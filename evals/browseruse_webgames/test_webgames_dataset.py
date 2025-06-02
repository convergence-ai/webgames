import json
import os

import pytest  # For potential future use with fixtures or advanced features

from .browseruse_webgames import MemoryDataset, Sample, load_webgames_dataset

# Content for a dummy webgames_tasks.jsonl file for testing
DUMMY_TASKS_CONTENT = """
{"id":"date","title":"Today's date","description":"Enter today's date","path":"date","password":"DATE_MASTER_2024","tags":["form","date"]}
{"id":"buttons","title":"Button megastar","description":"Click everything on the page to reveal the password","path":"buttons","password":"ClickityClickBoom2024","tags":["button","click"]}
{"id":"malformed","title":"Malformed Task"}
{"id":"no_id_task","password":"PASSWORD_ONLY"}
{"id":"no_password_task","id":"ID_ONLY"}
"""

DUMMY_FILE_PATH = "dummy_webgames_tasks_for_test.jsonl"


def setup_dummy_file():
    "Creates a dummy jsonl file for testing."
    with open(DUMMY_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(DUMMY_TASKS_CONTENT)


def teardown_dummy_file():
    "Removes the dummy jsonl file after testing."
    if os.path.exists(DUMMY_FILE_PATH):
        os.remove(DUMMY_FILE_PATH)


def test_load_webgames_dataset():
    "Tests the load_webgames_dataset function."
    setup_dummy_file()
    try:
        dataset = load_webgames_dataset(DUMMY_FILE_PATH)

        assert isinstance(dataset, MemoryDataset)
        assert dataset.name == "WebGamesTasks"
        # It should skip the 3 malformed tasks and load 2 valid ones
        assert len(dataset) == 2, f"Expected 2 samples, but got {len(dataset)}"

        # Check first sample (date task)
        sample1 = dataset[0]
        assert isinstance(sample1, Sample)
        assert sample1.id == "date"
        expected_input_1 = (
            "go to https://webgames.convergence.ai/date and complete the challenge "
            "as described. On successful completion, the task will give you a password. "
            "Tell me what the password is."
        )
        assert sample1.input == expected_input_1
        assert sample1.target == "DATE_MASTER_2024"
        assert sample1.metadata == {
            "title": "Today's date",
            "description": "Enter today's date",
            "path": "date",
            "tags": ["form", "date"],
        }

        # Check second sample (buttons task)
        sample2 = dataset[1]
        assert isinstance(sample2, Sample)
        assert sample2.id == "buttons"
        expected_input_2 = (
            "go to https://webgames.convergence.ai/buttons and complete the challenge "
            "as described. On successful completion, the task will give you a password. "
            "Tell me what the password is."
        )
        assert sample2.input == expected_input_2
        assert sample2.target == "ClickityClickBoom2024"
        assert sample2.metadata == {
            "title": "Button megastar",
            "description": "Click everything on the page to reveal the password",
            "path": "buttons",
            "tags": ["button", "click"],
        }

    finally:
        teardown_dummy_file()


# To run this test, navigate to the directory containing this file and run:
# pytest
# or specifically:
# pytest test_webgames_dataset.py
