"""
Unit tests for MLX format schemas and column mapping logic.
"""

import unittest
from hf2mlx.formats import (
    ColumnMapping,
    MLXFormat,
    auto_detect_mapping,
    format_record,
    format_template,
    validate_mapping,
)


class TestFormats(unittest.TestCase):

    def test_text_format_single_column(self):
        record = {"raw_text": "Apple Silicon with MLX is fast."}
        mapping = ColumnMapping(text_col="raw_text")
        out = format_record(record, MLXFormat.TEXT, mapping)
        self.assertEqual(out, {"text": "Apple Silicon with MLX is fast."})

    def test_text_format_template(self):
        record = {
            "instruction": "Explain quantum computing",
            "response": "It uses qubits.",
        }
        mapping = ColumnMapping(text_template="Q: {instruction}\nA: {response}")
        out = format_record(record, MLXFormat.TEXT, mapping)
        self.assertEqual(out, {"text": "Q: Explain quantum computing\nA: It uses qubits."})

    def test_chat_format_separate_columns(self):
        record = {
            "sys": "You are an assistant.",
            "user_query": "What is MLX?",
            "model_reply": "MLX is an array framework from Apple silicon research.",
        }
        mapping = ColumnMapping(
            system_col="sys",
            user_col="user_query",
            assistant_col="model_reply",
        )
        out = format_record(record, MLXFormat.CHAT, mapping)
        expected = {
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "What is MLX?"},
                {"role": "assistant", "content": "MLX is an array framework from Apple silicon research."},
            ]
        }
        self.assertEqual(out, expected)

    def test_chat_format_conversations_list(self):
        record = {
            "dialogue": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello there!"},
            ]
        }
        mapping = ColumnMapping(
            messages_col="dialogue",
            role_key="from",
            content_key="value",
        )
        out = format_record(record, MLXFormat.CHAT, mapping)
        self.assertEqual(
            out,
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello there!"},
                ]
            },
        )

    def test_prompt_completion_format(self):
        record = {
            "prompt": "Translate to Spanish: Hello",
            "completion": "Hola",
        }
        mapping = ColumnMapping(prompt_col="prompt", completion_col="completion")
        out = format_record(record, MLXFormat.PROMPT_COMPLETION, mapping)
        self.assertEqual(out, {"prompt": "Translate to Spanish: Hello", "completion": "Hola"})

    def test_dpo_format(self):
        record = {
            "prompt": "Write python code",
            "chosen": "print('hello')",
            "rejected": "echo 'hello'",
        }
        mapping = ColumnMapping(
            dpo_prompt_col="prompt",
            chosen_col="chosen",
            rejected_col="rejected",
        )
        out = format_record(record, MLXFormat.DPO, mapping)
        self.assertEqual(
            out,
            {"prompt": "Write python code", "chosen": "print('hello')", "rejected": "echo 'hello'"},
        )

    def test_validate_mapping(self):
        columns = ["instruction", "output"]
        mapping = ColumnMapping(prompt_col="instruction", completion_col="non_existent")
        errs = validate_mapping(MLXFormat.PROMPT_COMPLETION, mapping, columns)
        self.assertTrue(any("non_existent" in e for e in errs))

    def test_auto_detect_mapping(self):
        columns = ["instruction", "output"]
        mapping = auto_detect_mapping(MLXFormat.PROMPT_COMPLETION, columns)
        self.assertEqual(mapping.prompt_col, "instruction")
        self.assertEqual(mapping.completion_col, "output")


    def test_parse_column_list(self):
        from hf2mlx.formats import parse_column_list
        self.assertEqual(parse_column_list(None), [])
        self.assertEqual(parse_column_list(""), [])
        self.assertEqual(parse_column_list("prompt"), ["prompt"])
        self.assertEqual(parse_column_list("instruction + input"), ["instruction", "input"])
        self.assertEqual(parse_column_list("instruction, input"), ["instruction", "input"])
        self.assertEqual(parse_column_list("a + b + c"), ["a", "b", "c"])
        # Exact match priority if column actually has plus or comma
        self.assertEqual(parse_column_list("a + b", available_columns=["a + b"]), ["a + b"])

    def test_get_column_value(self):
        from hf2mlx.formats import get_column_value
        rec = {"instruction": "Solve X", "input": "X=5", "empty": "", "none": None}
        self.assertEqual(get_column_value(rec, "instruction"), "Solve X")
        self.assertEqual(get_column_value(rec, "instruction + input"), "Solve X\n\nX=5")
        # Ignores empty or None values without adding extra newlines
        self.assertEqual(get_column_value(rec, "instruction + empty"), "Solve X")
        self.assertEqual(get_column_value(rec, "instruction + none"), "Solve X")

    def test_prompt_completion_concatenation(self):
        record = {
            "instruction": "Write a story about space.",
            "input": "Keep it under 100 words.",
            "output": "Once upon a time in a galaxy...",
        }
        mapping = ColumnMapping(
            prompt_col="instruction + input",
            completion_col="output",
        )
        errs = validate_mapping(MLXFormat.PROMPT_COMPLETION, mapping, ["instruction", "input", "output"])
        self.assertEqual(errs, [])
        out = format_record(record, MLXFormat.PROMPT_COMPLETION, mapping)
        self.assertEqual(
            out,
            {
                "prompt": "Write a story about space.\n\nKeep it under 100 words.",
                "completion": "Once upon a time in a galaxy...",
            },
        )

    def test_validate_mapping_multi_column_error(self):
        mapping = ColumnMapping(prompt_col="instruction + missing_field", completion_col="output")
        errs = validate_mapping(MLXFormat.PROMPT_COMPLETION, mapping, ["instruction", "output"])
        self.assertTrue(any("missing_field" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
