#!/usr/bin/env python3
"""Scan GGUFs for MTP (nextn_predict_layers / nextn tensors)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def read_string(fh) -> str:
    (n,) = struct.unpack("<Q", fh.read(8))
    return fh.read(n).decode("utf-8", errors="replace")


def read_value(fh, vtype: int):
    if vtype == 0:
        return struct.unpack("<B", fh.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", fh.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", fh.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", fh.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", fh.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", fh.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", fh.read(4))[0]
    if vtype == 7:
        return bool(struct.unpack("<B", fh.read(1))[0])
    if vtype == 8:
        return read_string(fh)
    if vtype == 9:
        return struct.unpack("<Q", fh.read(8))[0]
    if vtype == 10:
        return struct.unpack("<q", fh.read(8))[0]
    if vtype == 11:
        return struct.unpack("<d", fh.read(8))[0]
    if vtype == 12:
        (etype,) = struct.unpack("<I", fh.read(4))
        (n,) = struct.unpack("<Q", fh.read(8))
        return [read_value(fh, etype) for _ in range(n)]
    raise ValueError(vtype)


def check_gguf(path: Path) -> dict:
    with path.open("rb") as fh:
        if fh.read(4) != b"GGUF":
            return {"error": "not gguf"}
        struct.unpack("<I", fh.read(4))  # version
        n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))
        meta: dict = {}
        for _ in range(n_kv):
            key = read_string(fh)
            (vtype,) = struct.unpack("<I", fh.read(4))
            meta[key] = read_value(fh, vtype)

        arch = meta.get("general.architecture", "?")
        nextn = None
        nextn_keys = {}
        for k, v in meta.items():
            lk = k.lower()
            if "nextn" in lk or "mtp" in lk:
                nextn_keys[k] = v
            if k.endswith("nextn_predict_layers") or "mtp_num_hidden_layers" in k:
                nextn = v

        nextn_tensors = 0
        samples: list[str] = []
        for _ in range(n_tensors):
            tname = read_string(fh)
            (ndims,) = struct.unpack("<I", fh.read(4))
            fh.read(8 * ndims + 4 + 8)
            low = tname.lower()
            if "nextn" in low or "mtp" in low:
                nextn_tensors += 1
                if len(samples) < 3:
                    samples.append(tname)

        mtp = (isinstance(nextn, int) and nextn > 0) or nextn_tensors > 0
        return {
            "arch": arch,
            "nextn": nextn,
            "nextn_keys": nextn_keys,
            "nextn_tensors": nextn_tensors,
            "samples": samples,
            "mtp": mtp,
            "n_tensors": n_tensors,
        }


def main() -> int:
    root = Path(r"D:\AI stuff")
    files = sorted(p for p in root.glob("*.gguf") if "mmproj" not in p.name.lower())
    print(f"Found {len(files)} GGUFs", flush=True)
    for f in files:
        print(f"... {f.name}", flush=True)
        try:
            r = check_gguf(f)
            flag = "MTP" if r["mtp"] else "---"
            print(
                f"[{flag}] {f.name} | arch={r['arch']} | "
                f"nextn_predict_layers={r['nextn']} | nextn_tensors={r['nextn_tensors']}",
                flush=True,
            )
            if r["nextn_keys"]:
                print(f"      meta: {r['nextn_keys']}", flush=True)
            if r["samples"]:
                print(f"      tensors: {r['samples']}", flush=True)
        except Exception as exc:
            print(f"[ERR] {f.name}: {type(exc).__name__}: {exc}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
