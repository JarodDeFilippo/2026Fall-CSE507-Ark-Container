"""Convert a .safetensors file into a flat torch state dict (.pth), without the safetensors package (not in the containers).

The format (github.com/huggingface/safetensors): an 8-byte little-endian header length N, N bytes of JSON
{name: {"dtype", "shape", "data_offsets": [begin, end]}, optional "__metadata__"}, then the raw little-endian tensor
bytes, offsets counted from the end of the header. The .pth is written next to the target as <out>.partial, reloaded and
compared byte for byte with every tensor's slice of the .safetensors buffer, and only then linked to <out>.

The input must be RAD-DINO's official DINOv2-compatible checkpoint, backbone_compatible.safetensors from the
microsoft/rad-dino Hugging Face repo (its README, section "Weights for fine-tuning", loads it into DINOv2's official
dinov2_vitb14 with strict=True). Names are kept as they are; nothing is remapped. The Hugging Face-format
model.safetensors (embeddings.*, encoder.layer.*, split q/k/v) is deliberately not supported: a tensor name starting
with "embeddings." or "encoder." raises ValueError before anything is written.

Run from base/: python convert_safetensors_to_pth.py <in.safetensors> <out.pth>    (refuses to overwrite <out.pth>)
"""
import json
import math
import os
import struct
import sys

import torch

DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
          "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def read_safetensors(path):
    """(header without __metadata__, data buffer, {name: tensor}) in the header's key order; raises ValueError on a malformed file."""
    if sys.byteorder != "little":
        raise RuntimeError("safetensors data is little-endian; torch.frombuffer reads native order")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise ValueError("{}: shorter than the 8-byte header length".format(path))
    (header_len,) = struct.unpack("<Q", data[:8])
    if 8 + header_len > len(data):
        raise ValueError("{}: header length {} runs past the end of the file".format(path, header_len))
    header = json.loads(data[8:8 + header_len])
    header.pop("__metadata__", None)
    buffer = memoryview(data)[8 + header_len:]
    tensors = {}
    for name, info in header.items():
        if info["dtype"] not in DTYPES:
            raise ValueError("{}: unsupported dtype {}".format(name, info["dtype"]))
        dtype, shape = DTYPES[info["dtype"]], info["shape"]
        begin, end = info["data_offsets"]
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if not 0 <= begin <= end <= len(buffer) or end - begin != size:
            raise ValueError("{}: data_offsets {} do not hold {} {} values of shape {}".format(name, [begin, end], math.prod(shape), info["dtype"], shape))
        # bytearray: a writable copy, so the tensor owns its memory and torch.frombuffer does not warn
        tensors[name] = torch.frombuffer(bytearray(buffer[begin:end]), dtype=dtype).reshape(shape) if size else torch.empty(shape, dtype=dtype)
    return header, buffer, tensors


def check_round_trip(pth_path, header, buffer):
    """Raise unless the saved file holds exactly the header's tensors, each byte-equal to its slice of the buffer."""
    saved = torch.load(pth_path, map_location="cpu", weights_only=True)
    if list(saved) != list(header):
        raise RuntimeError("{}: keys differ from the .safetensors header".format(pth_path))
    for name, info in header.items():
        tensor, (begin, end) = saved[name], info["data_offsets"]
        if (tensor.dtype != DTYPES[info["dtype"]] or list(tensor.shape) != info["shape"]
                or tensor.reshape(-1).view(torch.uint8).numpy().tobytes() != bytes(buffer[begin:end])):
            raise RuntimeError("{}: {} does not round-trip".format(pth_path, name))
    return saved


def convert(src, dst):
    if os.path.exists(dst):
        raise FileExistsError("refusing to overwrite {}".format(dst))
    partial = dst + ".partial"
    if os.path.exists(partial):
        raise FileExistsError("{} exists (an interrupted conversion?); remove it first".format(partial))
    header, buffer, tensors = read_safetensors(src)
    prefix = next((p for p in ("embeddings.", "encoder.") if any(name.startswith(p) for name in header)), None)
    if prefix:  # Hugging Face Dinov2Model names: refused before anything is written, never remapped
        raise ValueError("{}: tensor names start with '{}', the Hugging Face Dinov2Model layout of model.safetensors, which is deliberately "
                         "not supported; download backbone_compatible.safetensors from the microsoft/rad-dino repo instead".format(src, prefix))
    try:
        torch.save(tensors, partial)
        saved = check_round_trip(partial, header, buffer)
        os.link(partial, dst)  # fails instead of replacing if dst appeared meanwhile
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    dtypes = sorted({info["dtype"] for info in header.values()})
    return "wrote {}: {} tensors, {:,} values ({}); every tensor byte-equal to its .safetensors slice".format(
        dst, len(saved), sum(t.numel() for t in saved.values()), ", ".join(dtypes))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python convert_safetensors_to_pth.py <in.safetensors> <out.pth>")
    print(convert(sys.argv[1], sys.argv[2]))
