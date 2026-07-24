#!/usr/bin/env python3

import json
import struct
import sys
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

ANIMA_KOHYA_ATTN_MAP = {
    "self_attn_q_proj": "attn1.to_q",
    "self_attn_k_proj": "attn1.to_k",
    "self_attn_v_proj": "attn1.to_v",
    "self_attn_output_proj": "attn1.to_out.0",
    "cross_attn_q_proj": "attn2.to_q",
    "cross_attn_k_proj": "attn2.to_k",
    "cross_attn_v_proj": "attn2.to_v",
    "cross_attn_output_proj": "attn2.to_out.0",
    "mlp_layer1": "ff.net.0.proj",
    "mlp_layer2": "ff.net.2",
    "adaln_modulation_self_attn_1": "norm1.linear_1",
    "adaln_modulation_self_attn_2": "norm1.linear_2",
    "adaln_modulation_cross_attn_1": "norm2.linear_1",
    "adaln_modulation_cross_attn_2": "norm2.linear_2",
    "adaln_modulation_mlp_1": "norm3.linear_1",
    "adaln_modulation_mlp_2": "norm3.linear_2",
}

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

LORA_SUFFIXES = (
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_A.default.weight",
    ".lora_B.default.weight",
    ".alpha",
    ".dora_scale",
)


def human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def read_safetensors_header(path):
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            raise ValueError("File is too small.")
        header_len = int.from_bytes(length_bytes, "little", signed=False)
        if header_len <= 0 or header_len > file_size - 8:
            raise ValueError("File header is invalid.")
        header_bytes = f.read(header_len)
        if len(header_bytes) < header_len:
            raise ValueError("File header is truncated.")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"File header could not be parsed: {e}.")
    if not isinstance(header, dict):
        raise ValueError("File header is not a JSON object.")
    return header


def maybe_json(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "[") and stripped[-1:] in ("}", "]"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def strip_lora_suffix(key):
    for suffix in LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    if key.endswith(".weight"):
        return key[: -len(".weight")]
    return key


def print_summary(path, metadata, tensors):
    print(f"File: {path}")
    print(f"  On disk:        {human_size(path.stat().st_size)}")
    print(f"  Tensors:        {len(tensors)}")
    print(f"  Metadata keys:  {len(metadata)}")

    dtype_counts = {}
    data_bytes = 0
    for info in tensors.values():
        dtype = info.get("dtype", "?")
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        offsets = info.get("data_offsets")
        if isinstance(offsets, list) and len(offsets) == 2:
            data_bytes += max(0, offsets[1] - offsets[0])
    if dtype_counts:
        dtypes = ", ".join(
            f"{name}: {count}" for name, count in sorted(dtype_counts.items())
        )
        print(f"  Tensor dtypes:  {dtypes}")
    if data_bytes:
        print(f"  Tensor data:    {human_size(data_bytes)}")


def print_trigger_info(metadata):
    print("\n== Trigger words / activation ==")
    if not metadata:
        print("  (No metadata in this file?)")
        return

    printed_something = False

    direct_keys = [
        "ss_output_name",
        "modelspec.title",
        "modelspec.trigger_phrase",
        "ss_training_comment",
        "activation text",
        "instance_prompt",
    ]
    for key in direct_keys:
        if key in metadata and str(metadata[key]).strip():
            print(f"  {key}: {metadata[key]}")
            printed_something = True

    seen = set(direct_keys)
    for key in sorted(metadata):
        if key in seen:
            continue
        low = key.lower()
        if any(w in low for w in ("trigger", "activation", "instance_prompt")):
            print(f"  {key}: {metadata[key]}")
            printed_something = True

    freq = maybe_json(metadata.get("ss_tag_frequency", ""))
    if isinstance(freq, dict):
        totals = {}
        for dataset in freq.values():
            if isinstance(dataset, dict):
                for tag, count in dataset.items():
                    try:
                        totals[tag] = totals.get(tag, 0) + int(count)
                    except (TypeError, ValueError):
                        continue
        if totals:
            top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:25]
            print("  Top training tags (ss_tag_frequency):")
            for tag, count in top:
                print(f"    {count:>7}  {tag}")
            printed_something = True

    if not printed_something:
        print("  (No trigger metadata found?)")


def print_all_metadata(metadata):
    if not metadata:
        return
    print("\n== All metadata ==")
    for key in sorted(metadata):
        value = maybe_json(metadata[key])
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        if len(rendered) > 5000:
            rendered = rendered[:5000] + "... [truncated]"
        print(f"  {key}: {rendered}")


def analyze_modules(tensors):
    print("\n== Modules ==")
    keys = list(tensors)
    if not keys:
        print("  (no tensors)")
        return

    is_kohya = any(k.startswith(("lora_unet_", "lora_te")) for k in keys)
    is_diffusers = any(
        (".lora_a." in k.lower())
        or (".lora_b." in k.lower())
        or (".lora_down." in k and not k.startswith("lora_unet_"))
        for k in keys
    )

    if is_kohya:
        print("  Detected format: kohya (lora_unet_* / lora_te_*)")
        _analyze_kohya(keys)
    elif is_diffusers:
        print("  Detected format: diffusers / PEFT (dotted module paths)")
        _analyze_diffusers(keys)
    else:
        print("  Unknown format. Listing distinct module paths.")
        modules = sorted({strip_lora_suffix(k) for k in keys})
        for module in modules:
            print(f"    {module}")


def _analyze_kohya(keys):
    import re

    block_re = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")

    blocks = set()
    block_suffixes = set()
    other_keys = set()

    for key in keys:
        name_part = key.split(".", 1)[0]
        match = block_re.match(name_part)
        if match:
            blocks.add(int(match.group(1)))
            block_suffixes.add(match.group(2))
        else:
            other_keys.add(name_part)

    if blocks:
        lo, hi = min(blocks), max(blocks)
        print(f"  Transformer blocks touched: {len(blocks)} (index {lo}-{hi})")

    mapped = sorted(s for s in block_suffixes if s in ANIMA_KOHYA_ATTN_MAP)
    unmapped = sorted(s for s in block_suffixes if s not in ANIMA_KOHYA_ATTN_MAP)

    print(f"\n  Per-block modules mapped by Animus ({len(mapped)}):")
    for suffix in mapped:
        print(f"    {suffix:<32} -> {ANIMA_KOHYA_ATTN_MAP[suffix]}")

    if unmapped:
        print(f"\n  Per-block modules NOT mapped by Animus ({len(unmapped)}):")
        for suffix in unmapped:
            print(f"    {suffix}")
    else:
        print("\n  All per-block modules are mapped by Animus. ✓")

    if other_keys:
        print(f"\n  Non-block kohya modules ({len(other_keys)}):")
        print("  (Animus only converts lora_unet_blocks_*, so these are dropped)")
        for name in sorted(other_keys):
            print(f"    {name}")


def _analyze_diffusers(keys):
    modules = sorted({strip_lora_suffix(k) for k in keys})
    print(f"  Distinct target modules ({len(modules)}):")
    for module in modules:
        print(f"    {module}")
    print("\n  No conversion needed.")


def is_png(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == PNG_SIGNATURE
    except OSError:
        return False


def read_png_text(path):
    chunks = []
    with open(path, "rb") as f:
        if f.read(8) != PNG_SIGNATURE:
            raise ValueError("Not a PNG file.")
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            length, ctype = struct.unpack(">I4s", head)
            data = f.read(length)
            f.read(4)  # CRC, unchecked.
            ctype = ctype.decode("latin-1")
            if ctype == "IEND":
                break
            if ctype == "tEXt":
                key, _, val = data.partition(b"\x00")
                chunks.append((key.decode("latin-1"), val.decode("latin-1")))
            elif ctype == "zTXt":
                key, _, rest = data.partition(b"\x00")
                try:
                    val = zlib.decompress(rest[1:]).decode("latin-1")
                except zlib.error:
                    val = "<undecodable zTXt>"
                chunks.append((key.decode("latin-1"), val))
            elif ctype == "iTXt":
                key, _, rest = data.partition(b"\x00")
                compressed = rest[:1] == b"\x01"
                rest = rest[2:]
                _lang, _, rest = rest.partition(b"\x00")
                _tkey, _, text = rest.partition(b"\x00")
                if compressed:
                    try:
                        text = zlib.decompress(text)
                    except zlib.error:
                        text = b"<undecodable iTXt>"
                chunks.append((key.decode("latin-1"), text.decode("utf-8", "replace")))
    return chunks


def print_png_metadata(path, chunks):
    print(f"File: {path}")
    print(f"  On disk:        {human_size(path.stat().st_size)}")
    print(f"  Text chunks:    {len(chunks)}")
    if not chunks:
        print("\n  (No text metadata found.)")
        return
    print("\n== PNG metadata ==")
    for key, value in chunks:
        if len(value) > 5000:
            value = value[:5000] + "... [truncated]"
        print(f"  {key}: {value}")


def main(argv):
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if (len(argv) == 2 and argv[1] in ("-h", "--help")) else 2

    path = Path(argv[1]).expanduser()
    if not path.is_file():
        print(f"Error: not a file: {path}.", file=sys.stderr)
        return 2

    if is_png(path):
        try:
            chunks = read_png_text(path)
        except (OSError, ValueError) as e:
            print(f"Error reading {path}: {e}.", file=sys.stderr)
            return 1
        print_png_metadata(path, chunks)
        return 0

    try:
        header = read_safetensors_header(path)
    except (OSError, ValueError) as e:
        print(f"Error reading {path}: {e}.", file=sys.stderr)
        return 1

    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        metadata = {}
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    print_summary(path, metadata, tensors)
    print_trigger_info(metadata)
    analyze_modules(tensors)
    print_all_metadata(metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
