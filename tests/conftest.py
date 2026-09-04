"""
Shared test fixtures and synthetic dataset generators.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import unittest


def make_sample_qa_records(count: int = 50) -> List[Dict[str, Any]]:
    return [
        {
            "id": i,
            "instruction": f"Solve problem #{i}",
            "input": f"Given value {i * 2}",
            "output": f"The solution is {i * 2 + 10}",
            "system_prompt": "You are a helpful math tutor.",
        }
        for i in range(count)
    ]


def make_sample_chat_records(count: int = 40) -> List[Dict[str, Any]]:
    return [
        {
            "id": i,
            "conversations": [
                {"from": "human", "value": f"Hello, question #{i}?"},
                {"from": "gpt", "value": f"Here is the answer to question #{i}."},
            ],
        }
        for i in range(count)
    ]


def make_sample_dpo_records(count: int = 30) -> List[Dict[str, Any]]:
    return [
        {
            "id": i,
            "prompt": f"Write a summary of event #{i}",
            "chosen": f"This is a high-quality, comprehensive summary of #{i}.",
            "rejected": f"Short bad summary #{i}.",
        }
        for i in range(count)
    ]
