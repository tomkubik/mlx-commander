# MLX-Commander 🚀

A fast, persistent dual-panel TUI (Norton Commander style) & CLI converter for preparing Hugging Face datasets into Apple Silicon MLX fine-tuning formats (`mlx-lm`).

Built entirely with Python's standard library `curses` with zero mandatory dependencies and zero pre-compiled binaries.

---

## 🌟 Key Features

- **Persistent Multi-Panel TUI (Norton Commander style)**: Full keyboard navigation (`Tab` to switch panels, `↑`/`↓` to navigate, `Enter` to edit/open dropdowns, `F2` for Finder, `F5` to convert).
- **Multi-File Selection & Dataset Merging**: Select multiple dataset files at once (e.g. combining pre-split `train.jsonl` and `test.jsonl`). Verifies that all files have identical column schemas and merges them so you can randomize fresh Train / Validation / Test sets from scratch with a custom seed.
- **Multi-Column Concatenation**: Tap `Space` to multi-select and order columns from the original dataset (e.g. `instruction + input`) to concatenate them seamlessly with `\n\n`.
- **Live Reactive Preview**: Sample records format in real time as you change target formats or adjust column mappings.
- **Instantaneous ESC Response**: Curses escape delay configured to 25ms (< 1 frame), making modal dismissal instantaneous while preserving arrow and function keys.
- **Native macOS Cocoa Finder Picker**: Seamlessly select dataset folders or files via native macOS dialogs (compiled on the fly in `/tmp` with zero checked-in binaries).
- **Supported MLX Formats**:
  1. **Text Format**: `{"text": "..."}` — Causal LM / pre-training (single column, concatenated columns, or custom template).
  2. **Chat / Messages Format**: `{"messages": [{"role": "system|user|assistant", "content": "..."}]}` — Supports message lists (standard role/content or ShareGPT `from`/`value`), or separate role columns.
  3. **Prompt & Completion Format**: `{"prompt": "...", "completion": "..."}` — Q&A / instruction fine-tuning (`mlx_lm.lora --mask-prompt` compatible).
  4. **DPO / Preference Format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}` — Direct Preference Optimization.
- **Flexible Data Loader**: Parquet (`.parquet`), Arrow (`.arrow`), Hugging Face `save_to_disk` directories, JSONL (`.jsonl`), JSON arrays (`.json`), and CSV (`.csv`).
- **Deterministic Splits & Random Seed**: Customizable Train / Validation / Test percentages with 100% reproducible shuffling via random seed.
- **Ready-to-Use `mlx_lm.lora` Command**: Generates the exact training command ready to copy-paste.
- **CLI Wizard & Headless Modes**: Run line-by-line via `--wizard` or fully automated via headless CLI flags.

---

## 📦 Quick Start

### 1. Launch MLX-Commander (Default)

Launch the interactive dashboard using any of these equivalent commands:

```bash
# Recommended for ZIP downloads (no chmod needed, no -m parameter):
python3 run.py

# Or execute current folder directly:
python3 .

# Or as a Python package module:
python3 -m mlx_commander

# Or via executable runner:
./mlx-commander
```

You can also pass arguments directly (e.g. pre-loading a dataset or multiple files):
```bash
python3 run.py -d /path/to/my_hf_dataset
# Or combine multiple files:
python3 run.py -d train.jsonl test.jsonl
```

### 2. Line-by-Line Wizard Mode

For SSH sessions or non-curses environments:

```bash
./mlx-commander --wizard
```

### 3. Direct Command-Line Conversion (Automated / Headless)

You can pass all options via flags for direct scripted conversions:

```bash
./mlx-commander \
  --dataset /path/to/my_hf_dataset \
  --format prompt_completion \
  --prompt-col instruction \
  --completion-col output \
  --output ./mlx_data \
  --train 80 \
  --valid 10 \
  --test 10 \
  --seed 42
```

---

## 🖥️ Command Line Reference

```
usage: mlx-commander [-h] [-v] [-d DATASET [DATASET ...]]
                     [-f {text,chat,prompt_completion,dpo}]
                     [-o OUTPUT] [--train TRAIN] [--valid VALID] [--test TEST]
                     [--seed SEED] [--keep-splits] [--mapping MAPPING]
                     [--text-col TEXT_COL] [--text-template TEXT_TEMPLATE]
                     [--prompt-col PROMPT_COL] [--completion-col COMPLETION_COL]
                     [--messages-col MESSAGES_COL] [--user-col USER_COL]
                     [--assistant-col ASSISTANT_COL] [--system-col SYSTEM_COL]
                     [--chosen-col CHOSEN_COL] [--rejected-col REJECTED_COL]
                     [--commander] [--wizard]
```

### Key Flags:

| Flag | Description |
|---|---|
| `-d`, `--dataset` | Path to HF dataset directory or file on disk (`.arrow`, `.parquet`, `.jsonl`, `.json`, `.csv`). |
| `-f`, `--format` | Target MLX format (`text`, `chat`, `prompt_completion`, `dpo`). |
| `-o`, `--output` | Destination directory where `train.jsonl`, `valid.jsonl`, and `test.jsonl` are saved. |
| `--train` | Percentage of data for training (e.g. `80.0`). |
| `--valid` | Percentage of data for validation (e.g. `10.0`). |
| `--test` | Percentage of data for test (e.g. `10.0`, or `0` to omit). |
| `--seed` | Integer random seed for reproducible random shuffling. |
| `--keep-splits` | Preserve existing dataset splits without re-splitting. |
| `--text-col` | Column to use as `text` for `text` format. |
| `--text-template`| Template string with `{column_name}` variables for `text` format. |
| `--prompt-col` | Column to map to `prompt`. |
| `--completion-col` | Column to map to `completion`. |
| `--messages-col` | Column containing conversation turns list for `chat` format. |
| `--user-col` | Column for user turn in multi-column `chat` format. |
| `--assistant-col`| Column for assistant turn in multi-column `chat` format. |
| `--system-col` | Column for system prompt in multi-column `chat` format. |
| `--chosen-col` | Column for preferred response in `dpo` format. |
| `--rejected-col`| Column for dispreferred response in `dpo` format. |
| `--manifest-file`| Custom file path where machine-readable `mlx_manifest.json` will be saved. |
| `--prefill-state`| Pre-populate TUI state from a JSON string or path to JSON file. |
| `--spawn-terminal`| Launch interactive TUI in an external macOS Terminal window. |
| `--mcp` | Start Model Context Protocol (MCP) server over stdio. |
| `--tui` | Force launch full-screen curses TUI. |
| `--no-tui`, `--cli` | Run line-by-line CLI wizard instead of curses TUI. |

---

## 🤖 AI Agent Integration & MCP Support

MLX-Commander is designed from the ground up for the AI agent era (Antigravity, Claude Desktop, Cursor, Zed, Cline). 

Rather than having an agent interrogate users with 10 questions in chat, agents can **inspect schemas, pre-populate MLX-Commander with recommended mappings and splits, and launch the TUI for tactile confirmation with live preview**.

### 1. Model Context Protocol (MCP) Server

Connect MLX-Commander directly to **Claude Desktop**, **Cursor**, or any MCP-compatible agent:

```json
{
  "mcpServers": {
    "mlx-commander": {
      "command": "uvx",
      "args": ["mlx-commander", "--mcp"]
    }
  }
}
```

Exposed MCP Tools:
- `inspect_dataset`: Return column names, row counts, detected formats, and sample records.
- `launch_conversion_tui`: Pre-populate settings and pop up the TUI in macOS Terminal.app for user review. Returns the conversion manifest.
- `convert_dataset_headless`: Perform direct conversion in the background.

### 2. Pre-Populated TUI Handoff
Agents can pass pre-computed configuration directly:
```bash
mlx-commander --tui --spawn-terminal \
  --dataset dataset.parquet \
  --format chat \
  --messages-col conversations \
  --train 85 --valid 15 \
  --manifest-file ./mlx_dataset/mlx_manifest.json
```
The command spawns a native macOS Terminal window, displays the live JSONL preview immediately, and returns exit code `0` on conversion or `130` on cancellation.

### 3. Agent Skill Specification
A ready-to-use Antigravity / Agent Skill definition is included at [`.agents/skills/mlx-dataset-prep/SKILL.md`](.agents/skills/mlx-dataset-prep/SKILL.md) and [`skills/mlx-dataset-prep/SKILL.md`](skills/mlx-dataset-prep/SKILL.md).

---

## 🛠️ Step-by-Step Wizard Walkthrough

1. **Step 1: Dataset Source**: Select your dataset folder or file on your drive. The tool validates the file, inspects column names, row counts, and existing splits.
2. **Step 2: MLX Format**: Choose your target format (`Text`, `Chat / Messages`, `Prompt & Completion`, `DPO / Preference`).
3. **Step 3: Column Mapping**: Match dataset columns to MLX fields or enter a formatting template. The tool automatically detects candidate columns.
4. **Step 4: Splitting & Seed**: Configure Train / Valid / Test percentages. A random seed is automatically generated, and you can accept it or provide your own.
5. **Step 5: Output & Preview**: Specify the destination folder, review a live preview of the formatted JSONL lines, and confirm to write the files.
6. **Step 6: Ready to Fine-Tune**: Review written file sizes, row counts, and copy the generated `mlx_lm.lora` fine-tuning command.

---

## 🚀 Running Fine-Tuning with Apple MLX

Once your dataset is converted, fine-tune an LLM on Apple Silicon with `mlx-lm`:

```bash
mlx_lm.lora \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data ./mlx_dataset \
    --mask-prompt \
    --iters 600 \
    --batch-size 4
```

---

## 🧪 Running Unit Tests

Run the test suite with Python's built-in `unittest`:

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```
