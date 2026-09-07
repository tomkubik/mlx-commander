"""
pure_parquet.py - A lightweight, pure-Python Parquet reader.

Zero external dependencies: uses only Python's standard library (struct, io, gzip, datetime).
Provides seamless out-of-the-box loading of .parquet files (including Snappy, Gzip, and
Uncompressed codecs, dictionary encoding, and nullable columns) when pyarrow is not installed.
"""

from __future__ import annotations

import datetime
import gzip
import io
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def decompress_snappy_raw(data: bytes) -> bytes:
    """
    Pure-Python raw Snappy block decompressor.
    Parquet stores raw compressed Snappy frames (LZ77 byte-oriented format).
    """
    src = io.BytesIO(data)

    # Read varint uncompressed length
    length = 0
    shift = 0
    while True:
        b = src.read(1)
        if not b:
            break
        val = b[0]
        length |= (val & 0x7F) << shift
        if (val & 0x80) == 0:
            break
        shift += 7

    out = bytearray()
    while len(out) < length:
        tag_b = src.read(1)
        if not tag_b:
            break
        tag = tag_b[0]
        elem = tag & 0x03

        if elem == 0:  # Literal
            lit_len = tag >> 2
            if lit_len < 60:
                lit_len += 1
            elif lit_len == 60:
                lit_len = src.read(1)[0] + 1
            elif lit_len == 61:
                lit_len = struct.unpack("<H", src.read(2))[0] + 1
            elif lit_len == 62:
                b = src.read(3)
                lit_len = (b[0] | (b[1] << 8) | (b[2] << 16)) + 1
            elif lit_len == 63:
                lit_len = struct.unpack("<I", src.read(4))[0] + 1
            out.extend(src.read(lit_len))

        elif elem == 1:  # Copy with 1-byte offset
            copy_len = ((tag >> 2) & 0x07) + 4
            offset = ((tag >> 5) << 8) | src.read(1)[0]
            for _ in range(copy_len):
                out.append(out[-offset])

        elif elem == 2:  # Copy with 2-byte offset
            copy_len = (tag >> 2) + 1
            offset = struct.unpack("<H", src.read(2))[0]
            for _ in range(copy_len):
                out.append(out[-offset])

        elif elem == 3:  # Copy with 4-byte offset
            copy_len = (tag >> 2) + 1
            offset = struct.unpack("<I", src.read(4))[0]
            for _ in range(copy_len):
                out.append(out[-offset])

    return bytes(out)


class ThriftCompactReader:
    """Minimal self-describing Thrift Compact Protocol binary decoder."""

    def __init__(self, stream: io.RawIOBase | io.BufferedIOBase) -> None:
        self.stream = stream

    def read_varint(self) -> int:
        res = 0
        shift = 0
        while True:
            b = self.stream.read(1)
            if not b:
                raise EOFError("Unexpected EOF reading varint")
            val = b[0]
            res |= (val & 0x7F) << shift
            if (val & 0x80) == 0:
                break
            shift += 7
        return res

    def read_zigzag(self) -> int:
        n = self.read_varint()
        return (n >> 1) ^ -(n & 1)

    def read_binary(self) -> bytes:
        length = self.read_varint()
        return self.stream.read(length)

    def read_struct(self) -> Dict[int, Any]:
        fields: Dict[int, Any] = {}
        last_id = 0
        while True:
            header_b = self.stream.read(1)
            if not header_b:
                break
            header = header_b[0]
            if header == 0:  # STOP
                break

            type_id = header & 0x0F
            delta = (header >> 4) & 0x0F
            if delta == 0:
                field_id = self.read_zigzag()
            else:
                field_id = last_id + delta
            last_id = field_id

            val = self.read_value(type_id)
            fields[field_id] = val
        return fields

    def read_value(self, type_id: int) -> Any:
        # Thrift Compact Protocol Types:
        # 1: TRUE, 2: FALSE, 3: BYTE, 4: I16, 5: I32, 6: I64, 7: DOUBLE, 8: BINARY,
        # 9: LIST, 10: SET, 11: MAP, 12: STRUCT
        if type_id == 1:
            return True
        elif type_id == 2:
            return False
        elif type_id == 3:
            return self.stream.read(1)[0]
        elif type_id in (4, 5, 6):
            return self.read_zigzag()
        elif type_id == 7:
            return struct.unpack("<d", self.stream.read(8))[0]
        elif type_id == 8:
            return self.read_binary()
        elif type_id in (9, 10):  # LIST or SET
            size_and_type = self.stream.read(1)[0]
            size = (size_and_type >> 4) & 0x0F
            elem_type = size_and_type & 0x0F
            if size == 15:
                size = self.read_varint()
            return [self.read_value(elem_type) for _ in range(size)]
        elif type_id == 11:  # MAP
            size = self.read_varint()
            if size == 0:
                return {}
            types = self.stream.read(1)[0]
            ktype = (types >> 4) & 0x0F
            vtype = types & 0x0F
            res = {}
            for _ in range(size):
                k = self.read_value(ktype)
                v = self.read_value(vtype)
                res[k] = v
            return res
        elif type_id == 12:  # STRUCT
            return self.read_struct()
        else:
            raise ValueError(f"Unknown Thrift Compact type_id: {type_id}")


def read_rle_bitpacked_hybrid(
    stream: io.BytesIO,
    bit_width: int,
    length: Optional[int] = None,
) -> List[int]:
    """Read values using Parquet RLE/Bit-packed hybrid encoding."""
    end_pos = stream.tell() + length if length is not None else None
    res: List[int] = []

    def read_uvarint() -> int:
        r = 0
        s = 0
        while True:
            b = stream.read(1)
            if not b:
                return 0
            val = b[0]
            r |= (val & 0x7F) << s
            if (val & 0x80) == 0:
                break
            s += 7
        return r

    while True:
        if end_pos is not None and stream.tell() >= end_pos:
            break
        header = read_uvarint()
        if header == 0 and (end_pos is not None and stream.tell() >= end_pos):
            break
        if (header & 1) == 0:
            # RLE run: count values of width bytes
            count = header >> 1
            width_bytes = (bit_width + 7) // 8
            raw = stream.read(width_bytes)
            if not raw:
                break
            raw = raw + b"\x00" * (4 - len(raw))
            val = struct.unpack("<i", raw)[0]
            res.extend([val] * count)
        else:
            # Bit-packed run: num_groups * 8 values
            num_groups = header >> 1
            count = num_groups * 8
            byte_count = (bit_width * count) // 8
            raw_bytes = stream.read(byte_count)
            if not raw_bytes:
                break

            mask = (1 << bit_width) - 1
            bit_buf = 0
            bits_in_buf = 0
            byte_idx = 0
            for _ in range(count):
                while bits_in_buf < bit_width and byte_idx < len(raw_bytes):
                    bit_buf |= raw_bytes[byte_idx] << bits_in_buf
                    bits_in_buf += 8
                    byte_idx += 1
                val = bit_buf & mask
                bit_buf >>= bit_width
                bits_in_buf -= bit_width
                res.append(val)

    return res


def read_plain_values(
    stream: io.BytesIO,
    type_id: int,
    count: int,
    type_length: Optional[int] = None,
) -> List[Any]:
    """Read primitive values stored with PLAIN encoding."""
    if count <= 0:
        return []

    # Parquet Physical Types:
    # 0: BOOLEAN, 1: INT32, 2: INT64, 3: INT96, 4: FLOAT, 5: DOUBLE, 6: BYTE_ARRAY, 7: FIXED_LEN_BYTE_ARRAY
    if type_id == 0:  # BOOLEAN
        vals = []
        byte_count = (count + 7) // 8
        raw = stream.read(byte_count)
        for i in range(count):
            byte_val = raw[i // 8]
            bit = (byte_val >> (i % 8)) & 1
            vals.append(bool(bit))
        return vals
    elif type_id == 1:  # INT32
        data = stream.read(4 * count)
        return list(struct.unpack(f"<{count}i", data))
    elif type_id == 2:  # INT64
        data = stream.read(8 * count)
        return list(struct.unpack(f"<{count}q", data))
    elif type_id == 3:  # INT96 (nanosecond timestamp or big int)
        vals = []
        for _ in range(count):
            raw = stream.read(12)
            q, i = struct.unpack("<qi", raw)
            vals.append((q, i))
        return vals
    elif type_id == 4:  # FLOAT
        data = stream.read(4 * count)
        return list(struct.unpack(f"<{count}f", data))
    elif type_id == 5:  # DOUBLE
        data = stream.read(8 * count)
        return list(struct.unpack(f"<{count}d", data))
    elif type_id == 6:  # BYTE_ARRAY (4-byte length prefix + bytes)
        vals = []
        for _ in range(count):
            l_bytes = stream.read(4)
            if not l_bytes:
                break
            length = struct.unpack("<i", l_bytes)[0]
            vals.append(stream.read(length))
        return vals
    elif type_id == 7:  # FIXED_LEN_BYTE_ARRAY
        flen = type_length or 1
        return [stream.read(flen) for _ in range(count)]
    else:
        raise ValueError(f"Unsupported Parquet type ID: {type_id}")


def is_parquet_file(file_path: Union[str, Path]) -> bool:
    """Return True if the file has Parquet magic header and footer bytes (PAR1)."""
    p = Path(file_path)
    if not p.is_file() or p.stat().st_size < 12:
        return False
    try:
        with open(p, "rb") as f:
            f.seek(0)
            if f.read(4) != b"PAR1":
                return False
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    except Exception:
        return False


def read_parquet_records(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read all records from a Parquet file using pure Python.
    Returns a list of dicts: [{"col1": val1, "col2": val2, ...}, ...]
    """
    path = Path(file_path)
    with open(path, "rb") as f:
        # 1. Magic bytes check
        f.seek(0)
        if f.read(4) != b"PAR1":
            raise ValueError(f"File {path.name} is not a valid Parquet file (missing PAR1 header)")
        f.seek(-4, 2)
        if f.read(4) != b"PAR1":
            raise ValueError(f"File {path.name} is not a valid Parquet file (missing PAR1 footer)")

        # 2. FileMetaData Footer
        f.seek(-8, 2)
        footer_len = struct.unpack("<I", f.read(4))[0]
        f.seek(-(8 + footer_len), 2)
        meta_bytes = f.read(footer_len)

        fmeta = ThriftCompactReader(io.BytesIO(meta_bytes)).read_struct()
        schema_elements = fmeta[2]

        # SchemaElement:
        # 1: type, 2: type_length, 3: repetition_type, 4: name, 5: num_children, 6: converted_type
        schema_by_name: Dict[str, Dict[int, Any]] = {}
        col_names: List[str] = []
        for se in schema_elements[1:]:  # Skip root schema element
            name_raw = se[4]
            name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
            schema_by_name[name] = se
            col_names.append(name)

        row_groups = fmeta.get(4, [])
        all_records: List[Dict[str, Any]] = []

        for rg in row_groups:
            col_chunks = rg[1]
            rg_num_rows = rg[3]
            rg_data: Dict[str, List[Any]] = {name: [] for name in col_names}

            for c in col_chunks:
                cmd = c[3]  # ColumnMetaData
                path_parts = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in cmd[3]]
                col_name = path_parts[-1]
                if col_name not in schema_by_name:
                    continue

                se = schema_by_name[col_name]
                col_type = cmd[1]  # Physical type
                codec = cmd[4]     # 0=UNCOMPRESSED, 1=SNAPPY, 2=GZIP, 6=ZSTD
                data_offset = cmd[9]
                dict_offset = cmd.get(11)

                dict_items: Optional[List[Any]] = None

                def decompress_page(cdata: bytes) -> bytes:
                    if codec == 0:
                        return cdata
                    elif codec == 1:
                        return decompress_snappy_raw(cdata)
                    elif codec == 2:
                        return gzip.decompress(cdata)
                    elif codec == 6:
                        # Try external zstandard if available
                        try:
                            import zstandard as zstd
                            return zstd.ZstdDecompressor().decompress(cdata)
                        except ImportError:
                            raise ValueError(
                                f"Parquet file uses ZSTD compression. Install 'zstandard' or 'pyarrow' to read it."
                            )
                    else:
                        raise ValueError(f"Unsupported Parquet compression codec: {codec}")

                # Read Dictionary Page if present
                if dict_offset is not None:
                    f.seek(dict_offset, 0)
                    ph = ThriftCompactReader(f).read_struct()
                    compressed_sz = ph[3]
                    comp_data = f.read(compressed_sz)
                    raw_dict = decompress_page(comp_data)
                    dict_hdr = ph[7]
                    num_dict_vals = dict_hdr[1]
                    dict_items = read_plain_values(io.BytesIO(raw_dict), col_type, num_dict_vals, se.get(2))

                # Read Data Pages
                f.seek(data_offset, 0)
                col_values: List[Any] = []

                while len(col_values) < rg_num_rows:
                    ph = ThriftCompactReader(f).read_struct()
                    ptype = ph[1]  # 0=DATA_PAGE, 2=DICTIONARY_PAGE
                    comp_sz = ph[3]
                    comp_data = f.read(comp_sz)

                    if ptype == 2:  # Dictionary page at data offset
                        raw_dict = decompress_page(comp_data)
                        dict_hdr = ph[7]
                        num_dict_vals = dict_hdr[1]
                        dict_items = read_plain_values(io.BytesIO(raw_dict), col_type, num_dict_vals, se.get(2))
                        continue

                    if ptype not in (0, 3):  # Ignore unsupported page types (index pages, etc.)
                        continue

                    raw_data = decompress_page(comp_data)
                    s_stream = io.BytesIO(raw_data)
                    daph = ph[5]
                    num_page_vals = daph[1]
                    page_enc = daph[2]  # 0=PLAIN, 2=PLAIN_DICTIONARY, 8=RLE_DICTIONARY

                    # Definition levels (null handling for OPTIONAL columns)
                    rep_type = se.get(3)  # 0=REQUIRED, 1=OPTIONAL, 2=REPEATED
                    def_levels: Optional[List[int]] = None
                    if rep_type == 1:  # OPTIONAL
                        def_len_bytes = s_stream.read(4)
                        if def_len_bytes:
                            def_len = struct.unpack("<i", def_len_bytes)[0]
                            def_levels = read_rle_bitpacked_hybrid(s_stream, 1, def_len)

                    if page_enc == 0:  # PLAIN
                        non_null_count = (
                            sum(1 for d in def_levels if d == 1)
                            if def_levels is not None
                            else num_page_vals
                        )
                        plain_vals = read_plain_values(s_stream, col_type, non_null_count, se.get(2))
                        if def_levels is not None:
                            val_iter = iter(plain_vals)
                            for dl in def_levels[:num_page_vals]:
                                col_values.append(next(val_iter) if dl == 1 else None)
                        else:
                            col_values.extend(plain_vals)

                    elif page_enc in (2, 8):  # PLAIN_DICTIONARY / RLE_DICTIONARY
                        if dict_items is None:
                            raise ValueError(f"Dictionary page missing for dictionary-encoded column {col_name}")
                        bit_width = s_stream.read(1)[0]
                        indices = read_rle_bitpacked_hybrid(s_stream, bit_width)
                        if def_levels is not None:
                            idx_iter = iter(indices)
                            for dl in def_levels[:num_page_vals]:
                                if dl == 1:
                                    col_values.append(dict_items[next(idx_iter)])
                                else:
                                    col_values.append(None)
                        else:
                            for idx in indices[:num_page_vals]:
                                col_values.append(dict_items[idx])

                # Post-process types (decode UTF-8, JSON, dates, etc.)
                converted_type = se.get(6)  # 0=UTF8, 5=DATE, 8=TIMESTAMP_MILLIS, 14=JSON
                clean_vals: List[Any] = []
                for v in col_values[:rg_num_rows]:
                    if v is None:
                        clean_vals.append(None)
                    elif isinstance(v, bytes):
                        if converted_type == 14:  # JSON
                            try:
                                clean_vals.append(json.loads(v.decode("utf-8")))
                            except Exception:
                                clean_vals.append(v.decode("utf-8", "replace"))
                        else:
                            try:
                                clean_vals.append(v.decode("utf-8"))
                            except UnicodeDecodeError:
                                clean_vals.append(v)
                    elif converted_type == 5 and isinstance(v, int):  # DATE
                        clean_vals.append(str(datetime.date.fromordinal(v)))
                    elif converted_type == 8 and isinstance(v, int):  # TIMESTAMP_MILLIS
                        clean_vals.append(datetime.datetime.utcfromtimestamp(v / 1000.0).isoformat())
                    else:
                        clean_vals.append(v)

                rg_data[col_name] = clean_vals

            # Construct row dicts for this row group
            for i in range(rg_num_rows):
                row = {name: rg_data[name][i] for name in col_names if i < len(rg_data[name])}
                all_records.append(row)

        return all_records
