"""
Dataset Inspection and Loading for Hugging Face datasets stored on disk.
Supports:
- Hugging Face datasets saved with `dataset.save_to_disk()`
- Hugging Face DatasetDict saved with `dataset_dict.save_to_disk()`
- Local Parquet, Arrow, JSON, JSONL, and CSV files/folders
- Hugging Face cache directories
- Fallback loaders when `datasets` library is not yet installed
"""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    import datasets
    from datasets import Dataset, DatasetDict, load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


@dataclass
class LoadedDataset:
    source_path: str
    is_split: bool
    split_names: List[str]
    split_counts: Dict[str, int]
    columns: List[str]
    total_rows: int
    sample_records: List[Dict[str, Any]]
    _raw_splits: Dict[str, Any] = field(default_factory=dict, repr=False)

    def iter_records(self, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Yield records as standard Python dicts for the specified split or all records."""
        target_splits = [split] if split else self.split_names
        for s_name in target_splits:
            data = self._raw_splits.get(s_name)
            if data is None:
                continue

            if HAS_DATASETS and isinstance(data, (Dataset, datasets.Dataset)):
                for row in data:
                    yield dict(row)
            elif HAS_PYARROW and isinstance(data, pa.Table):
                for batch in data.to_batches():
                    pydicts = batch.to_pylist()
                    for r in pydicts:
                        yield r
            elif isinstance(data, list):
                for item in data:
                    yield item
            elif callable(data):
                for item in data():
                    yield item

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Return all records across all splits as a list."""
        return list(self.iter_records())


def inspect_dataset_path(path_str: str) -> Tuple[bool, str]:
    """
    Validate path and return (is_valid, error_or_info_message).
    """
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return False, f"Path does not exist: {path}"
    return True, f"Found: {path}"


def load_from_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON lines or standard JSON array from file."""
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        # Check first non-whitespace char
        pos = f.tell()
        first_char = f.read(1)
        while first_char and first_char.isspace():
            first_char = f.read(1)
        f.seek(pos)

        if first_char == "[":
            # JSON array
            data = json.load(f)
            if isinstance(data, list):
                records = [d for d in data if isinstance(d, dict)]
        else:
            # JSONL
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        obj = json.loads(line_str)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except json.JSONDecodeError:
                        continue
    return records


def load_from_csv(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from CSV file."""
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def load_from_arrow_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from an Apache Arrow IPC stream/file."""
    if not HAS_PYARROW:
        raise RuntimeError("Reading .arrow files directly requires 'pyarrow' installed.")
    try:
        with pa.memory_map(str(file_path), "r") as source:
            try:
                reader = pa_ipc.open_file(source)
                table = reader.read_all()
            except Exception:
                reader = pa_ipc.open_stream(source)
                table = reader.read_all()
        return table.to_pylist()
    except Exception as e:
        raise RuntimeError(f"Failed to read Arrow file {file_path}: {e}")


def load_from_parquet_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from a Parquet file."""
    if not HAS_PYARROW:
        raise RuntimeError("Reading .parquet files requires 'pyarrow' installed.")
    table = pq.read_table(str(file_path))
    return table.to_pylist()


def is_hf_save_to_disk_dir(path: Path) -> bool:
    """Check if directory has the signature of datasets.save_to_disk()."""
    if not path.is_dir():
        return False
    # Check for dataset_dict.json or dataset_info.json or state.json
    has_dict = (path / "dataset_dict.json").exists()
    has_info = (path / "dataset_info.json").exists()
    has_state = (path / "state.json").exists()
    has_arrow = bool(list(path.glob("*.arrow"))) or bool(list(path.glob("*/*.arrow")))
    return (has_dict or (has_info and has_arrow) or (has_state and has_arrow))


def load_local_dataset(path_str: str) -> LoadedDataset:
    """
    Load a Hugging Face dataset or dataset files from the local drive.
    Returns a unified LoadedDataset instance.
    """
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    # Case 1: HF datasets library is available and directory is a saved dataset
    if HAS_DATASETS:
        if is_hf_save_to_disk_dir(path):
            try:
                loaded = load_from_disk(str(path))
                if isinstance(loaded, DatasetDict):
                    split_names = list(loaded.keys())
                    split_counts = {k: len(loaded[k]) for k in split_names}
                    first_split = loaded[split_names[0]]
                    cols = first_split.column_names
                    total = sum(split_counts.values())
                    samples = [dict(first_split[i]) for i in range(min(5, len(first_split)))]
                    return LoadedDataset(
                        source_path=str(path),
                        is_split=True,
                        split_names=split_names,
                        split_counts=split_counts,
                        columns=cols,
                        total_rows=total,
                        sample_records=samples,
                        _raw_splits=dict(loaded),
                    )
                elif isinstance(loaded, Dataset):
                    cols = loaded.column_names
                    total = len(loaded)
                    samples = [dict(loaded[i]) for i in range(min(5, len(loaded)))]
                    return LoadedDataset(
                        source_path=str(path),
                        is_split=False,
                        split_names=["default"],
                        split_counts={"default": total},
                        columns=cols,
                        total_rows=total,
                        sample_records=samples,
                        _raw_splits={"default": loaded},
                    )
            except Exception as e:
                # Log or fall through to file-based loader
                pass

    # Case 2: Direct file or directory of data files (Parquet, JSONL, Arrow, CSV)
    raw_splits: Dict[str, List[Dict[str, Any]]] = {}

    if path.is_file():
        ext = path.suffix.lower()
        records: List[Dict[str, Any]] = []
        if ext in (".json", ".jsonl"):
            records = load_from_json_or_jsonl(path)
        elif ext == ".csv":
            records = load_from_csv(path)
        elif ext == ".parquet":
            records = load_from_parquet_file(path)
        elif ext == ".arrow":
            records = load_from_arrow_file(path)
        else:
            # Try jsonl by default
            records = load_from_json_or_jsonl(path)
        raw_splits["default"] = records

    elif path.is_dir():
        # Check if subdirectories correspond to splits (e.g. train, test, validation)
        split_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        # If HF save_to_disk with dataset_dict.json
        dict_json_file = path / "dataset_dict.json"
        if dict_json_file.exists():
            try:
                with open(dict_json_file, "r") as f:
                    dict_info = json.load(f)
                    expected_splits = dict_info.get("splits", [])
                    for s in expected_splits:
                        s_dir = path / s
                        arrow_files = list(s_dir.glob("*.arrow"))
                        s_records: List[Dict[str, Any]] = []
                        for af in arrow_files:
                            s_records.extend(load_from_arrow_file(af))
                        raw_splits[s] = s_records
            except Exception:
                pass

        if not raw_splits:
            # Check for standard split files in the root dir: train.jsonl, test.jsonl, etc.
            found_split_files = False
            for s in ["train", "valid", "validation", "val", "test"]:
                for ext in [".jsonl", ".parquet", ".arrow", ".json", ".csv"]:
                    candidate = path / f"{s}{ext}"
                    if candidate.exists():
                        found_split_files = True
                        if ext in (".json", ".jsonl"):
                            raw_splits[s] = load_from_json_or_jsonl(candidate)
                        elif ext == ".parquet":
                            raw_splits[s] = load_from_parquet_file(candidate)
                        elif ext == ".arrow":
                            raw_splits[s] = load_from_arrow_file(candidate)
                        elif ext == ".csv":
                            raw_splits[s] = load_from_csv(candidate)
                        break

            if not found_split_files:
                # Aggregate all data files in the directory
                all_records: List[Dict[str, Any]] = []
                data_files = sorted(
                    [f for f in path.glob("*") if f.suffix.lower() in (".jsonl", ".json", ".parquet", ".arrow", ".csv")]
                )
                for df in data_files:
                    ext = df.suffix.lower()
                    if ext in (".json", ".jsonl"):
                        all_records.extend(load_from_json_or_jsonl(df))
                    elif ext == ".parquet":
                        all_records.extend(load_from_parquet_file(df))
                    elif ext == ".arrow":
                        all_records.extend(load_from_arrow_file(df))
                    elif ext == ".csv":
                        all_records.extend(load_from_csv(df))
                raw_splits["default"] = all_records

    if not raw_splits or all(len(v) == 0 for v in raw_splits.values()):
        raise ValueError(
            f"No dataset records found in '{path}'. Supported formats include Hugging Face save_to_disk folders, "
            f"Arrow (.arrow), Parquet (.parquet), JSON Lines (.jsonl), JSON (.json), or CSV (.csv)."
        )

    split_names = [k for k in raw_splits.keys() if len(raw_splits[k]) > 0]
    split_counts = {k: len(raw_splits[k]) for k in split_names}
    total_rows = sum(split_counts.values())

    # Extract columns from the first available record
    columns: List[str] = []
    sample_records: List[Dict[str, Any]] = []
    for s_name in split_names:
        for r in raw_splits[s_name]:
            if not columns:
                columns = list(r.keys())
            if len(sample_records) < 5:
                sample_records.append(r)
            if len(sample_records) >= 5:
                break
        if len(sample_records) >= 5:
            break

    is_split = len(split_names) > 1 or (len(split_names) == 1 and split_names[0] != "default")

    return LoadedDataset(
        source_path=str(path),
        is_split=is_split,
        split_names=split_names,
        split_counts=split_counts,
        columns=columns,
        total_rows=total_rows,
        sample_records=sample_records,
        _raw_splits=raw_splits,
    )
