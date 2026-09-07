#!/usr/bin/env python3

# autopep8: off

import os
import sys

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

if not os.environ.get("PYTHONIOENCODING"):
    os.environ["PYTHONIOENCODING"] = "utf-8"

import errno
import gc
import json
import math
import re
import shlex
import shutil
import struct
import signal
import subprocess
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path


def _init_fontconfig():
    import ctypes
    import ctypes.util

    for name in (
        ctypes.util.find_library("fontconfig"),
        "libfontconfig.so.1",
        "libfontconfig.so",
    ):
        if not name:
            continue
        try:
            ctypes.CDLL(name).FcInit()
            return
        except OSError:
            continue


_init_fontconfig()

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk
import warnings

import numpy
import torch
from torch import nn
from torch.nn import functional as F

# autopep8: on

APP_NAME = "animus"
WINDOW_TITLE = "Animus Upscale"


def _xdg_home(variable, fallback):
    value = os.environ.get(variable)
    if value and os.path.isabs(value):
        return Path(value)
    return Path.home() / fallback


CONFIG_DIR = _xdg_home("XDG_CONFIG_HOME", ".config") / APP_NAME
DATA_DIR = _xdg_home("XDG_DATA_HOME", ".local/share") / APP_NAME

CONFIG_FILE = CONFIG_DIR / "upscale.json"
OUTPUT_DIR = DATA_DIR / "upscales"
MODEL_DIR = DATA_DIR / "upscalers"

MIN_FREE_DISK_MARGIN = 256 * 1024 * 1024
WORKER_THREAD_TIMEOUT = 10.0
FFMPEG_TERM_TIMEOUT = 5.0

REAL_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"

BUILTIN_MODELS = (
    (
        "Anime video (fast)",
        "realesr-animevideov3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
    ),
    (
        "General video (slower)",
        "realesr-general-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
    ),
    (
        "Anime (heavy, RRDB)",
        "RealESRGAN_x4plus_anime_6B.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    (
        "Photo (heavy, RRDB)",
        "RealESRGAN_x4plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
    ),
)

DEFAULT_MODEL = BUILTIN_MODELS[0][1]
CUSTOM_MODEL_ID = "custom"

OUTPUT_PRESETS = (
    ("x2", "2x source size", "scale", 2.0),
    ("x3", "3x source size", "scale", 3.0),
    ("x4", "4x source size", "scale", 4.0),
    ("h480", "480p (fit height)", "fit", 480),
    ("h576", "576p (fit height)", "fit", 576),
    ("h720", "720p (fit height)", "fit", 720),
    ("h960", "960p (fit height)", "fit", 960),
    ("h1080", "1080p (fit height)", "fit", 1080),
    ("h1440", "1440p (fit height)", "fit", 1440),
    ("h2160", "2160p (fit height)", "fit", 2160),
    ("custom", "Custom size...", "custom", 0),
)

DEFAULT_PRESET = "x2"
DEFAULT_TILE = 256
DEFAULT_TILE_PAD = 24
DEFAULT_PRE_PAD = 8
UI_FONT = b"* { font-family: monospace; font-size: 12pt; }"
PREVIEW_INTERVAL = 0.75
PREVIEW_MAX_SIZE = 640

ENCODERS = (
    ("libx264", "H.264 (libx264)"),
    ("libx265", "H.265 (libx265)"),
    ("ffv1", "FFV1 (lossless)"),
)
DEFAULT_ENCODER = "libx264"

X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)
DEFAULT_ENCODER_PRESET = "slow"
DEFAULT_CRF = 17

PRECISIONS = (
    ("float32", "32-bit float (default)"),
    ("bfloat16", "bfloat16"),
    ("float16", "float16"),
)
PRECISION_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
DEFAULT_PRECISION = "float32"

CONTAINERS = (
    ("mkv", "Matroska (.mkv)"),
    ("mp4", "MP4 (.mp4)"),
)
DEFAULT_CONTAINER = "mkv"

AUDIO_MODES = (
    ("copy", "Copy from source"),
    ("aac", "Re-encode to AAC"),
    ("none", "Drop audio"),
)
DEFAULT_AUDIO = "copy"

VIDEO_PATTERNS = (
    "*.mp4",
    "*.mkv",
    "*.avi",
    "*.mov",
    "*.webm",
    "*.m4v",
    "*.mpg",
    "*.mpeg",
    "*.wmv",
    "*.flv",
    "*.ts",
    "*.vob",
    "*.ogv",
    "*.3gp",
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_FFMPEG_PROGRESS = re.compile(
    r"^(frame|fps|bitrate|total_size|out_time\w*|dup_frames|drop_frames|speed"
    r"|progress|stream_\d+_\d+_q)=(.*)$"
)


def _strip_ansi(text):
    return _ANSI_ESCAPE.sub("", text)


def _format_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _format_duration(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def spin_value(spin):
    return round(spin.get_value(), spin.get_digits())


def even(value):
    value = int(round(value))
    return max(2, value - (value % 2))


def _activation(act_type, num_feat):
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.ReLU(inplace=True)


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    ):
        super().__init__()
        self.upscale = upscale
        self.num_out_ch = num_out_ch

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(_activation(act_type, num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(_activation(act_type, num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super().__init__()
        self.scale = scale
        self.num_out_ch = num_out_ch
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def read_state_dict(path):
    path = Path(path)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
    else:
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

    for key in ("params_ema", "params", "state_dict", "model"):
        if isinstance(state_dict, dict) and isinstance(state_dict.get(key), dict):
            state_dict = state_dict[key]
            break

    if not isinstance(state_dict, dict):
        raise ValueError(f"{path.name} does not hold a state dict.")

    return {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}


def _body_indices(state_dict):
    indices = {}
    for key, tensor in state_dict.items():
        match = re.fullmatch(r"body\.(\d+)\.weight", key)
        if match and hasattr(tensor, "dim"):
            indices[int(match.group(1))] = tensor.dim()
    return indices


def build_upscaler(state_dict):
    if "conv_first.weight" in state_dict:
        weight = state_dict["conv_first.weight"]
        num_feat = int(weight.shape[0])
        packed_in = int(weight.shape[1])
        scale = {3: 4, 12: 2, 48: 1}.get(packed_in)
        if scale is None:
            raise ValueError(
                f"conv_first takes {packed_in} channels, which is not an "
                "x1, x2 or x4 RRDBNet."
            )

        blocks = set()
        for key in state_dict:
            match = re.match(r"body\.(\d+)\.", key)
            if match:
                blocks.add(int(match.group(1)))
        if not blocks:
            raise ValueError("The checkpoint has no RRDB blocks.")

        num_block = max(blocks) + 1
        num_grow_ch = int(state_dict["body.0.rdb1.conv1.weight"].shape[0])
        num_out_ch = int(state_dict["conv_last.weight"].shape[0])

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )
        size_multiple = {1: 4, 2: 2}.get(scale, 1)
        min_overlap = (2 + num_block * 15 + 2) * size_multiple
        description = f"RRDBNet x{scale} ({num_block} blocks, {num_feat} features)"
        return model, scale, size_multiple, min_overlap, description

    indices = _body_indices(state_dict)
    conv_indices = sorted(i for i, dim in indices.items() if dim == 4)
    if not conv_indices or conv_indices[0] != 0:
        raise ValueError(
            "Unrecognized checkpoint. This reads the two Real-ESRGAN "
            "architectures - SRVGGNetCompact (realesr-*v3) and RRDBNet "
            "(RealESRGAN_x*plus, plain ESRGAN) and nothing else. Newer "
            "designs such as SPAN, OmniSR or DAT would each need their own "
            "reader."
        )

    first = state_dict["body.0.weight"]
    num_feat = int(first.shape[0])
    num_in_ch = int(first.shape[1])
    last = conv_indices[-1]
    out_planes = int(state_dict[f"body.{last}.weight"].shape[0])

    num_conv = (last - 2) // 2
    if num_conv < 0 or last != 2 + 2 * num_conv:
        raise ValueError(f"Unexpected SRVGG body layout (last conv at {last}).")

    upscale = int(round(math.sqrt(out_planes / max(num_in_ch, 1))))
    if upscale < 1 or num_in_ch * upscale * upscale != out_planes:
        raise ValueError(
            f"Cannot derive the scale from a {out_planes}-channel final conv."
        )

    act_type = "prelu" if indices.get(1) == 1 else "relu"
    model = SRVGGNetCompact(
        num_in_ch=num_in_ch,
        num_out_ch=num_in_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=upscale,
        act_type=act_type,
    )
    min_overlap = num_conv + 2
    description = (
        f"SRVGGNetCompact x{upscale} ({num_conv} convs, {num_feat} features, "
        f"{act_type})"
    )
    return model, upscale, 1, min_overlap, description


def load_upscaler(path, device, channels_last=True, dtype=torch.float32):
    state_dict = read_state_dict(path)
    model, scale, size_multiple, min_overlap, description = build_upscaler(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    model = model.to(device=device, dtype=dtype)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    return model, scale, size_multiple, min_overlap, description


NCNN_MAGIC = 7767517


def write_ncnn_model(state_dict, param_path, bin_path):
    model, scale, size_multiple, min_overlap, description = build_upscaler(state_dict)
    if not isinstance(model, SRVGGNetCompact):
        raise ValueError(
            f"{description} is not a compact generator. Only those are "
            "converted, because the heavy ones are not worth running on video."
        )
    model.load_state_dict(state_dict, strict=True)

    convolutions, activations = [], []
    for index, layer in enumerate(model.body):
        if isinstance(layer, nn.Conv2d):
            convolutions.append((index, layer))
        else:
            activations.append((index, layer))

    lines = [
        "Input            data      0 1 data",
        "Split            fork      1 2 data body skip",
    ]
    blobs = ["data", "body", "skip"]
    weights = []
    previous = "body"

    for order, (index, conv) in enumerate(convolutions):
        blob = f"c{order}"
        blobs.append(blob)
        weight = state_dict[f"body.{index}.weight"]
        lines.append(
            f"Convolution      conv{order} 1 1 {previous} {blob} "
            f"0={conv.out_channels} 1={conv.kernel_size[1]} "
            f"11={conv.kernel_size[0]} 3={conv.stride[1]} 13={conv.stride[0]} "
            f"4={conv.padding[1]} 14={conv.padding[0]} 5=1 6={weight.numel()}"
        )
        weights.append((True, weight))
        weights.append((False, state_dict[f"body.{index}.bias"]))
        previous = blob

        if order >= len(activations):
            continue
        act_index, activation = activations[order]
        blob = f"a{order}"
        blobs.append(blob)
        if isinstance(activation, nn.PReLU):
            slope = state_dict[f"body.{act_index}.weight"]
            lines.append(
                f"PReLU            act{order} 1 1 {previous} {blob} "
                f"0={slope.numel()}"
            )
            weights.append((False, slope))
        else:
            negative = float(getattr(activation, "negative_slope", 0.0))
            lines.append(
                f"ReLU             act{order} 1 1 {previous} {blob} " f"0={negative}"
            )
        previous = blob

    blobs += ["shuffled", "nearest", "out"]
    lines.append(f"PixelShuffle     shuffle   1 1 {previous} shuffled 0={scale} 1=0")
    lines.append(
        f"Interp           nearest   1 1 skip nearest 0=1 "
        f"1={float(scale)} 2={float(scale)}"
    )
    lines.append("BinaryOp         add       2 1 shuffled nearest out 0=0")

    Path(param_path).parent.mkdir(parents=True, exist_ok=True)
    with open(param_path, "w") as handle:
        handle.write(f"{NCNN_MAGIC}\n{len(lines)} {len(blobs)}\n")
        handle.write("\n".join(lines) + "\n")

    with open(bin_path, "wb") as handle:
        for flagged, tensor in weights:
            if flagged:
                handle.write(struct.pack("<I", 0))  # 0 => float32
            handle.write(
                tensor.detach().to(torch.float32).contiguous().numpy().tobytes()
            )

    return scale, size_multiple, min_overlap, description


class NcnnUpscaler:
    def __init__(
        self, param_path, bin_path, num_out_ch=3, threads=0, gpu=None, fp16=True
    ):
        import ncnn

        self._ncnn = ncnn
        self.num_out_ch = num_out_ch
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = gpu is not None
        self.net.opt.use_fp16_packed = bool(fp16)
        self.net.opt.use_fp16_storage = bool(fp16)
        self.net.opt.use_fp16_arithmetic = bool(fp16)
        if gpu is not None:
            self.net.set_vulkan_device(int(gpu))
        if threads:
            self.net.opt.num_threads = int(threads)
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

        self.on_gpu = bool(self.net.opt.use_vulkan_compute)

        self._allocators = []
        if self.on_gpu:
            try:
                device = self.net.vulkan_device()
                blob = ncnn.VkBlobAllocator(device)
                staging = ncnn.VkStagingAllocator(device)
                self.net.opt.blob_vkallocator = blob
                self.net.opt.workspace_vkallocator = blob
                self.net.opt.staging_vkallocator = staging
                self._allocators = [blob, staging]
            except Exception as e:
                print(
                    f"Could not hold on to the GPU allocators ({e}). NCNN will "
                    "take and return them every frame instead."
                )

    def __call__(self, frame):
        import time

        t0 = time.perf_counter()
        chw = frame[0].detach().to(torch.float32).cpu().contiguous().numpy()
        t1 = time.perf_counter()
        extractor = self.net.create_extractor()
        extractor.input("data", self._ncnn.Mat(chw))
        status, result = extractor.extract("out")
        t2 = time.perf_counter()
        if status != 0:
            raise RuntimeError(f"ncnn returned {status} from the network.")
        out = torch.from_numpy(numpy.array(result)).unsqueeze(0)
        t3 = time.perf_counter()
        #        print(
        #            f"upload {(t1 - t0) * 1e3:7.1f}ms | "
        #            f"gpu {(t2 - t1) * 1e3:7.1f}ms | "
        #            f"download {(t3 - t2) * 1e3:7.1f}ms",
        #            flush=True,
        #        )
        return out

    def close(self):
        try:
            self.net.clear()
        except Exception:
            pass
        for allocator in getattr(self, "_allocators", []):
            try:
                allocator.clear()
            except Exception:
                pass
        self._allocators = []


def ncnn_model_paths(weights):
    weights = Path(weights)
    return (weights.with_suffix(".ncnn.param"), weights.with_suffix(".ncnn.bin"))


def load_ncnn_upscaler(weights, gpu=None, threads=0, fp16=True):
    weights = Path(weights)
    param_path, bin_path = ncnn_model_paths(weights)

    stale = (
        not param_path.exists()
        or not bin_path.exists()
        or param_path.stat().st_mtime < weights.stat().st_mtime
    )
    state_dict = read_state_dict(weights)
    if stale:
        print(f"Converting {weights.name} to ncnn...")
        scale, size_multiple, min_overlap, description = write_ncnn_model(
            state_dict, param_path, bin_path
        )
    else:
        _model, scale, size_multiple, min_overlap, description = build_upscaler(
            state_dict
        )

    model = NcnnUpscaler(param_path, bin_path, threads=threads, gpu=gpu, fp16=fp16)

    if gpu is not None and not model.on_gpu:
        print(
            f"ncnn could not use Vulkan device {gpu} and fell back to its own "
            "CPU backend, which is why this will be slow. The device list is "
            "built from what Vulkan enumerates, which is not the same as what "
            "ncnn can open."
        )

    where = f"Vulkan device {gpu}" if model.on_gpu else "CPU"
    precision = "fp16" if (fp16 and model.on_gpu) else "fp32"
    return (
        model,
        scale,
        size_multiple,
        min_overlap,
        f"{description} via NCNN ({where}, {precision})",
    )


def ncnn_devices():
    try:
        import ncnn
    except Exception:
        return []

    try:
        count = ncnn.get_gpu_count()
    except Exception:
        return []

    kinds = {0: "discrete", 1: "integrated", 2: "virtual", 3: "software"}
    devices = []
    for index in range(count):
        name, kind = f"device {index}", ""
        try:
            info = ncnn.get_gpu_info(index)
            name = info.device_name()
            kind = kinds.get(info.type(), "")
        except Exception:
            pass
        label = f"Vulkan via ncnn - {name}"
        if kind:
            label += f" [{kind}]"
        devices.append((f"ncnn:{index}", label, kind))
    return devices


def preferred_device():
    found = ncnn_devices()
    for wanted in ("discrete", "integrated"):
        for ident, _label, kind in found:
            if kind == wanted:
                return ident
    return "cpu"


def _pad_frame(frame, pre_pad, size_multiple):
    height, width = frame.shape[-2:]
    pad = max(0, min(pre_pad, height - 1, width - 1))

    right = pad + (-(width + 2 * pad)) % size_multiple
    bottom = pad + (-(height + 2 * pad)) % size_multiple

    if pad == 0 and right == 0 and bottom == 0:
        return frame, 0

    padding = (pad, right, pad, bottom)
    if right < width and bottom < height:
        return F.pad(frame, padding, mode="reflect"), pad
    return F.pad(frame, padding, mode="replicate"), pad


def upscale_frame(
    model,
    frame,
    scale,
    tile,
    tile_pad,
    size_multiple=1,
    pre_pad=DEFAULT_PRE_PAD,
    stop_check=None,
):
    height, width = frame.shape[-2:]
    padded, offset = _pad_frame(frame, pre_pad, size_multiple)
    padded_h, padded_w = padded.shape[-2:]

    if tile <= 0 or (tile >= padded_w and tile >= padded_h):
        output = model(padded)
    else:
        step = max(size_multiple, tile - (tile % size_multiple))
        overlap = tile_pad + (-tile_pad) % size_multiple

        output = padded.new_empty(
            (padded.shape[0], model.num_out_ch, padded_h * scale, padded_w * scale)
        )

        for y in range(0, padded_h, step):
            for x in range(0, padded_w, step):
                if stop_check is not None and stop_check():
                    return None

                in_x1 = min(x + step, padded_w)
                in_y1 = min(y + step, padded_h)
                pad_x0 = max(x - overlap, 0)
                pad_x1 = min(in_x1 + overlap, padded_w)
                pad_y0 = max(y - overlap, 0)
                pad_y1 = min(in_y1 + overlap, padded_h)

                patch = padded[:, :, pad_y0:pad_y1, pad_x0:pad_x1]
                out_patch = model(patch)

                cut_x0 = (x - pad_x0) * scale
                cut_y0 = (y - pad_y0) * scale
                cut_x1 = cut_x0 + (in_x1 - x) * scale
                cut_y1 = cut_y0 + (in_y1 - y) * scale

                output[:, :, y * scale : in_y1 * scale, x * scale : in_x1 * scale] = (
                    out_patch[:, :, cut_y0:cut_y1, cut_x0:cut_x1]
                )

    if offset:
        output = output[
            :,
            :,
            offset * scale : (offset + height) * scale,
            offset * scale : (offset + width) * scale,
        ]
    else:
        output = output[:, :, : height * scale, : width * scale]

    return output


def download_model(url, dest, progress=None, stop_check=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "Animus"})

    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length") or 0)

            if total:
                free = shutil.disk_usage(dest.parent).free
                if free < total + MIN_FREE_DISK_MARGIN:
                    raise OSError(
                        errno.ENOSPC,
                        f"Need {_format_size(total + MIN_FREE_DISK_MARGIN)} for "
                        f"{dest.name}, only {_format_size(free)} free.",
                    )

            done = 0
            last_report = 0.0
            with open(temp, "wb") as handle:
                while True:
                    if stop_check is not None and stop_check():
                        raise KeyboardInterrupt()
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if progress is not None and now - last_report > 0.5:
                        last_report = now
                        progress(done, total)

            if progress is not None:
                progress(done, total)

        if total and done != total:
            raise OSError(
                f"Short download for {dest.name}: got {_format_size(done)} of "
                f"{_format_size(total)}."
            )

        os.replace(temp, dest)
        return dest
    except BaseException:
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass
        raise


def _parse_fraction(text, fallback=0.0):
    if not text:
        return fallback
    try:
        if "/" in str(text):
            num, _, den = str(text).partition("/")
            den = float(den)
            return float(num) / den if den else fallback
        return float(text)
    except (TypeError, ValueError):
        return fallback


def probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]

    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    except FileNotFoundError:
        raise RuntimeError("ffprobe was not found. Install ffmpeg and try again.")

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffprobe could not read {Path(path).name}: {detail}")

    payload = json.loads(completed.stdout.decode("utf-8", "replace") or "{}")
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"{Path(path).name} has no video stream.")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"{Path(path).name} reports a zero-sized frame.")

    rotation = 0.0
    for side_data in video.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = _parse_fraction(side_data.get("rotation"), 0.0)
    if not rotation:
        rotation = _parse_fraction((video.get("tags") or {}).get("rotate"), 0.0)
    if int(abs(rotation)) % 180 == 90:
        width, height = height, width

    fps = _parse_fraction(video.get("avg_frame_rate"), 0.0)
    if fps <= 0:
        fps = _parse_fraction(video.get("r_frame_rate"), 0.0)
    if fps <= 0:
        fps = 25.0

    duration = _parse_fraction(video.get("duration"), 0.0)
    if duration <= 0:
        duration = _parse_fraction((payload.get("format") or {}).get("duration"), 0.0)

    frames = 0
    try:
        frames = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frames = 0
    frames_exact = frames > 0
    if frames <= 0 and duration > 0:
        frames = int(round(duration * fps))

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return {
        "path": str(path),
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "frames": max(frames, 0),
        "frames_exact": frames_exact,
        "codec": video.get("codec_name") or "?",
        "pix_fmt": video.get("pix_fmt") or "?",
        "has_audio": has_audio,
        "interlaced": (video.get("field_order") or "progressive")
        not in ("progressive", "unknown", ""),
    }


def target_size(info, preset_id, custom_width, custom_height):
    width, height = info["width"], info["height"]

    for ident, _label, mode, value in OUTPUT_PRESETS:
        if ident != preset_id:
            continue
        if mode == "scale":
            return even(width * value), even(height * value)
        if mode == "fit":
            return even(width * (value / height)), even(value)
        break

    return even(custom_width), even(custom_height)


def source_filters(deinterlace, extra_filters):
    filters = []
    if deinterlace:
        filters.append("yadif=0:-1:0")
    extra_filters = (extra_filters or "").strip()
    if extra_filters:
        filters.append(extra_filters)
    return ",".join(filters)


def output_filters(model_width, model_height, out_width, out_height, extra_filters):
    filters = []
    if (out_width, out_height) != (model_width, model_height):
        filters.append(f"scale={out_width}:{out_height}:flags=lanczos")
    extra_filters = (extra_filters or "").strip()
    if extra_filters:
        filters.append(extra_filters)
    return ",".join(filters)


def _probe_size(source, filters):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        *source,
        "-vf",
        f"{filters or 'null'},showinfo",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return None

    found = re.findall(rb"Parsed_showinfo[^\n]*?\bs:(\d+)x(\d+)", completed.stderr)
    if not found:
        return None
    width, height = found[-1]
    return int(width), int(height)


def probe_filtered_size(path, filters):
    return _probe_size(["-i", str(path)], filters)


def probe_output_size(width, height, filters):
    return _probe_size(["-f", "lavfi", "-i", f"color=s={width}x{height}"], filters)


def build_decoder_command(info, start, limit, deinterlace, extra_filters):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]

    if start > 0:
        command += ["-ss", f"{start:.3f}"]
    if limit > 0:
        command += ["-t", f"{limit:.3f}"]
    command += ["-i", info["path"]]

    filters = source_filters(deinterlace, extra_filters)
    if filters:
        command += ["-vf", filters]

    command += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return command


def build_encoder_command(
    info,
    model_width,
    model_height,
    out_width,
    out_height,
    fps,
    dest,
    encoder,
    crf,
    preset,
    audio,
    start,
    limit,
    extra_filters,
):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-progress",
        "pipe:2",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{model_width}x{model_height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
    ]

    want_audio = audio != "none" and info["has_audio"]
    if want_audio:
        if start > 0:
            command += ["-ss", f"{start:.3f}"]
        if limit > 0:
            command += ["-t", f"{limit:.3f}"]
        command += ["-i", info["path"]]
        command += ["-map", "0:v:0", "-map", "1:a:0?"]
        command += ["-c:a", "copy" if audio == "copy" else "aac"]
    else:
        command += ["-map", "0:v:0", "-an"]

    filters = output_filters(
        model_width, model_height, out_width, out_height, extra_filters
    )
    if filters:
        command += ["-vf", filters]

    if encoder == "ffv1":
        command += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"]
    else:
        command += [
            "-c:v",
            encoder,
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
        ]

    command += [str(dest)]
    return command


def container_for(encoder, chosen=DEFAULT_CONTAINER):
    if encoder == "ffv1":
        return "mkv"
    return chosen if chosen in dict(CONTAINERS) else DEFAULT_CONTAINER


def read_exact(stream, buffer):
    view = memoryview(buffer)
    filled = 0
    while filled < len(buffer):
        got = stream.readinto(view[filled:])
        if not got:
            break
        filled += got
    return filled


def write_exact(stream, data):
    view = memoryview(data).cast("B")
    while view:
        written = stream.write(view)
        if not written:
            raise OSError("The encoder stopped accepting frame data.")
        view = view[written:]


def process_stream(
    model,
    source,
    sink,
    width,
    height,
    scale,
    tile,
    tile_pad,
    size_multiple=1,
    device="cpu",
    dtype=torch.float32,
    channels_last=True,
    stop_check=None,
    on_frame=None,
):
    device = torch.device(device)
    frame_bytes = width * height * 3
    buffer = bytearray(frame_bytes)
    written = 0

    while stop_check is None or not stop_check():
        got = read_exact(source, buffer)
        if got == 0:
            break
        if got < frame_bytes:
            print(
                f"Warning: the decoder stopped {frame_bytes - got} bytes into a "
                "frame. Treating that as the end of the stream."
            )
            break

        frame = (
            torch.frombuffer(buffer, dtype=torch.uint8)
            .reshape(height, width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
            .div_(255.0)
        )
        if channels_last:
            frame = frame.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            result = upscale_frame(
                model,
                frame,
                scale,
                tile,
                tile_pad,
                size_multiple=size_multiple,
                stop_check=stop_check,
            )
            if result is None:
                break
            rgb = (
                result[0]
                .mul_(255.0)
                .round_()
                .clamp_(0.0, 255.0)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
            )
            if on_frame is not None:
                on_frame(written + 1, rgb)

        del frame, result
        write_exact(sink, rgb.numpy())
        del rgb
        written += 1

    return written


def terminate_process(process, timeout=FFMPEG_TERM_TIMEOUT):
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=timeout)
        except Exception:
            pass
    except Exception:
        pass


def cpu_flags():
    try:
        with open("/proc/cpuinfo", "r") as handle:
            for line in handle:
                if line.startswith("flags"):
                    return frozenset(line.partition(":")[2].split())
    except OSError:
        pass
    return frozenset()


CPU_FLAGS = cpu_flags()
# Without one of these, bfloat16 is emulated and costs more than float32.
BF16_IN_HARDWARE = bool(CPU_FLAGS & {"amx_bf16", "avx512_bf16"})


def available_devices():
    devices = ["cpu"]

    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                devices.append(f"cuda:{index}")
    except Exception:
        pass

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for index in range(xpu.device_count()):
                devices.append(f"xpu:{index}")
    except Exception:
        pass

    try:
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass

    devices.extend(ident for ident, _label, _kind in ncnn_devices())

    return devices


VULKAN_MISSING_OPS = ("aten::prelu", "aten::pixel_shuffle")


def describe_device(name):
    for ident, label, _kind in ncnn_devices():
        if ident == name:
            return label
    try:
        if name.startswith("cuda"):
            index = int(name.partition(":")[2] or 0)
            return f"{name} - {torch.cuda.get_device_name(index)}"
        if name.startswith("xpu"):
            index = int(name.partition(":")[2] or 0)
            return f"{name} - {torch.xpu.get_device_name(index)}"
    except Exception:
        pass
    return name


def describe_torch_build():
    parts = [f"torch {torch.__version__}"]
    try:
        parts.append(
            "oneDNN " + ("on" if torch.backends.mkldnn.is_available() else "off")
        )
    except Exception:
        pass
    try:
        capability = torch.backends.cpu.get_cpu_capability()
        if BF16_IN_HARDWARE:
            capability += "+bf16"
        parts.append(capability)
    except Exception:
        pass
    try:
        parts.append(
            "OpenMP " + ("on" if torch.backends.openmp.is_available() else "off")
        )
    except Exception:
        pass
    parts.append(f"{torch.get_num_threads()} threads")
    return ", ".join(parts)


class ConsoleRedirector:
    def __init__(self, text_view, original_stream):
        self.text_view = text_view
        self.text_buffer = text_view.get_buffer()
        self.original_stream = original_stream
        self.needs_newline = False
        self._lock = threading.Lock()

    def _update_newline_tracking(self, text):
        if text:
            with self._lock:
                self.needs_newline = not text.endswith("\n")

    def write(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()
        GLib.idle_add(self._append_text, text, False)
        self._update_newline_tracking(text)

    def write_with_newline(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()

        with self._lock:
            needs_newline = self.needs_newline

        GLib.idle_add(self._append_text, text, needs_newline)
        self._update_newline_tracking(text)

    def _append_text(self, text, prepend_newline):
        text = _strip_ansi(text)
        end_iter = self.text_buffer.get_end_iter()
        if prepend_newline:
            self.text_buffer.insert(end_iter, "\n" + text)
        else:
            self.text_buffer.insert(end_iter, text)

        mark = self.text_buffer.create_mark(
            None, self.text_buffer.get_end_iter(), False
        )
        self.text_view.scroll_mark_onscreen(mark)
        self.text_buffer.delete_mark(mark)
        return False

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.original_stream.fileno()

    def __getattr__(self, name):
        original = self.__dict__.get("original_stream")
        if original is not None:
            return getattr(original, name)
        raise AttributeError(name)


class UpscaleGUI(Gtk.Window):
    def __init__(self):
        super().__init__(title=WINDOW_TITLE)
        warnings.filterwarnings("ignore")
        self.set_wmclass("Animus", "Animus")
        self.set_default_size(880, 900)
        self.set_border_width(10)

        self.source = None
        self.working = False
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.decoder = None
        self.encoder = None
        self.current_output = None
        self._loading_settings = False
        self._nudging = False
        self._user_set = set()
        self._last_preview = 0.0
        self._muxed_bytes = 0
        self._total_label = "?"

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        self._apply_font()

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(main_box)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_size_request(-1, 330)

        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        controls_box.set_border_width(10)
        scrolled.add(controls_box)
        main_box.pack_start(scrolled, False, False, 0)

        controls_box.pack_start(self._build_source_area(), False, False, 0)
        controls_box.pack_start(self._build_output_row(), False, False, 0)
        controls_box.pack_start(self._build_model_rows(), False, False, 0)
        controls_box.pack_start(self._build_advanced(), False, False, 0)

        self.notebook = Gtk.Notebook()
        main_box.pack_start(self.notebook, True, True, 0)
        self._build_console_tab()
        self._build_preview_tab()
        self.notebook.set_current_page(0)

        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(True)
        self.progress.set_text("Idle")
        main_box.pack_start(self.progress, False, False, 0)

        main_box.pack_start(self._build_buttons(), False, False, 0)

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text, self.original_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.original_stderr)

        self._report_environment()
        self.load_settings()

    def _apply_font(self):
        # XXX
        try:
            provider = Gtk.CssProvider()
            provider.load_from_data(UI_FONT)
            Gtk.StyleContext.add_provider_for_screen(
                Gdk.Screen.get_default(),
                provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )
        except Exception as e:
            print(f"Could not set the interface font: {e}.")

    def _build_source_area(self):
        frame = Gtk.Frame(label="Source")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_border_width(10)
        frame.add(box)

        self.hint_label = Gtk.Label(label="Choose a video to upscale.")
        box.pack_start(self.hint_label, False, False, 0)

        self.source_label = Gtk.Label(label="No file selected.")
        self.source_label.set_line_wrap(True)
        self.source_label.set_justify(Gtk.Justification.CENTER)
        box.pack_start(self.source_label, False, False, 0)

        path_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.source_entry = Gtk.Entry()
        self.source_entry.set_placeholder_text("Path to a video file")
        self.source_entry.connect("activate", self.on_source_entry_activate)
        path_box.pack_start(self.source_entry, True, True, 0)

        browse_button = Gtk.Button(label="Browse...")
        browse_button.connect("clicked", self.on_browse_source)
        path_box.pack_start(browse_button, False, False, 0)
        box.pack_start(path_box, False, False, 0)

        return frame

    def _build_output_row(self):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

        label = Gtk.Label(label="Output size:")
        label.set_size_request(100, -1)
        label.set_xalign(0)
        box.pack_start(label, False, False, 0)

        self.preset_combo = Gtk.ComboBoxText()
        for ident, text, _mode, _value in OUTPUT_PRESETS:
            self.preset_combo.append(ident, text)
        self.preset_combo.set_active_id(DEFAULT_PRESET)
        self.preset_combo.connect("changed", self.on_output_changed)
        box.pack_start(self.preset_combo, False, False, 0)

        self.out_width_spin = Gtk.SpinButton()
        self.out_width_spin.set_adjustment(
            Gtk.Adjustment(
                value=1280, lower=16, upper=16384, step_increment=2, page_increment=16
            )
        )
        self.out_width_spin.set_size_request(90, -1)
        self.out_width_spin.connect("value-changed", self.on_output_changed)
        box.pack_start(self.out_width_spin, False, False, 0)

        times = Gtk.Label(label="x")
        box.pack_start(times, False, False, 0)

        self.out_height_spin = Gtk.SpinButton()
        self.out_height_spin.set_adjustment(
            Gtk.Adjustment(
                value=960, lower=16, upper=16384, step_increment=2, page_increment=16
            )
        )
        self.out_height_spin.set_size_request(90, -1)
        self.out_height_spin.connect("value-changed", self.on_output_changed)
        box.pack_start(self.out_height_spin, False, False, 0)

        self.size_label = Gtk.Label(label="")
        self.size_label.set_xalign(0)
        box.pack_start(self.size_label, True, True, 6)

        return box

    def _build_model_rows(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)

        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        label = Gtk.Label(label="Model:")
        label.set_size_request(100, -1)
        label.set_xalign(0)
        model_box.pack_start(label, False, False, 0)

        self.model_combo = Gtk.ComboBoxText()
        for text, filename, _url in BUILTIN_MODELS:
            self.model_combo.append(filename, text)
        self.model_combo.append(CUSTOM_MODEL_ID, "Custom weights...")
        self.model_combo.set_active_id(DEFAULT_MODEL)
        self.model_combo.connect("changed", self.on_model_changed)
        model_box.pack_start(self.model_combo, False, False, 0)

        self.model_note = Gtk.Label(label="")
        self.model_note.set_xalign(0)
        model_box.pack_start(self.model_note, True, True, 6)
        box.pack_start(model_box, False, False, 0)

        custom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        custom_label = Gtk.Label(label="Weights file:")
        custom_label.set_size_request(100, -1)
        custom_label.set_xalign(0)
        custom_box.pack_start(custom_label, False, False, 0)

        self.model_entry = Gtk.Entry()
        self.model_entry.set_placeholder_text(
            "A .pth or .safetensors Real-ESRGAN / ESRGAN checkpoint"
        )
        custom_box.pack_start(self.model_entry, True, True, 0)

        model_browse = Gtk.Button(label="Browse...")
        model_browse.connect("clicked", self.on_browse_model)
        custom_box.pack_start(model_browse, False, False, 0)

        self.custom_model_box = custom_box
        box.pack_start(custom_box, False, False, 0)

        return box

    def _spin(self, value, lower, upper, step, page, digits=0, width=90):
        spin = Gtk.SpinButton()
        spin.set_adjustment(
            Gtk.Adjustment(
                value=value,
                lower=lower,
                upper=upper,
                step_increment=step,
                page_increment=page,
            )
        )
        spin.set_digits(digits)
        spin.set_size_request(width, -1)
        return spin

    def _build_advanced(self):
        expander = Gtk.Expander(label="Advanced")
        grid = Gtk.Grid(row_spacing=6, column_spacing=8)
        grid.set_border_width(10)
        expander.add(grid)

        row = 0

        grid.attach(self._label("Device:"), 0, row, 1, 1)
        self.device_combo = Gtk.ComboBoxText()
        for device in available_devices():
            self.device_combo.append(device, describe_device(device))
        self.device_combo.set_active_id(preferred_device())
        self.device_combo.connect("changed", self.on_device_changed)
        grid.attach(self.device_combo, 1, row, 2, 1)

        grid.attach(self._label("Threads:"), 3, row, 1, 1)
        self.threads_spin = self._spin(
            min(os.cpu_count() or 4, 64), 1, 256, 1, 4, width=80
        )
        grid.attach(self.threads_spin, 4, row, 1, 1)
        row += 1

        grid.attach(self._label("Tile:"), 0, row, 1, 1)
        self.tile_spin = self._spin(DEFAULT_TILE, 0, 4096, 32, 128, width=90)
        self.tile_spin.connect("value-changed", self._note_user_choice, "tile")
        grid.attach(self.tile_spin, 1, row, 1, 1)

        grid.attach(self._label("Overlap:"), 2, row, 1, 1)
        self.tile_pad_spin = self._spin(DEFAULT_TILE_PAD, 0, 256, 4, 16, width=80)
        grid.attach(self.tile_pad_spin, 3, row, 1, 1)

        self.channels_last_check = Gtk.CheckButton(label="channels_last")
        self.channels_last_check.set_active(True)
        grid.attach(self.channels_last_check, 4, row, 1, 1)
        row += 1

        grid.attach(self._label("Source filters:"), 0, row, 1, 1)
        self.deinterlace_check = Gtk.CheckButton(label="Deinterlace (yadif)")
        grid.attach(self.deinterlace_check, 1, row, 2, 1)

        self.compile_check = Gtk.CheckButton(label="torch.compile")
        grid.attach(self.compile_check, 3, row, 2, 1)
        row += 1

        grid.attach(self._label("Precision:"), 0, row, 1, 1)
        self.precision_combo = Gtk.ComboBoxText()
        for ident, text in PRECISIONS:
            self.precision_combo.append(ident, text)
        self.precision_combo.set_active_id(DEFAULT_PRECISION)
        self.precision_combo.connect("changed", self._note_user_choice, "precision")
        grid.attach(self.precision_combo, 1, row, 2, 1)
        row += 1

        grid.attach(self._label("Extra -vf before:"), 0, row, 1, 1)
        self.filters_pre_entry = Gtk.Entry()
        grid.attach(self.filters_pre_entry, 1, row, 5, 1)
        row += 1

        grid.attach(self._label("Extra -vf after:"), 0, row, 1, 1)
        self.filters_post_entry = Gtk.Entry()
        grid.attach(self.filters_post_entry, 1, row, 5, 1)
        row += 1

        grid.attach(self._label("Encoder:"), 0, row, 1, 1)
        self.encoder_combo = Gtk.ComboBoxText()
        for ident, text in ENCODERS:
            self.encoder_combo.append(ident, text)
        self.encoder_combo.set_active_id(DEFAULT_ENCODER)
        self.encoder_combo.connect("changed", self.on_encoder_changed)
        grid.attach(self.encoder_combo, 1, row, 1, 1)

        grid.attach(self._label("CRF:"), 2, row, 1, 1)
        self.crf_spin = self._spin(DEFAULT_CRF, 0, 51, 1, 5, width=80)
        grid.attach(self.crf_spin, 3, row, 1, 1)

        self.preset_encoder_combo = Gtk.ComboBoxText()
        for name in X264_PRESETS:
            self.preset_encoder_combo.append(name, name)
        self.preset_encoder_combo.set_active_id(DEFAULT_ENCODER_PRESET)
        grid.attach(self.preset_encoder_combo, 4, row, 1, 1)
        row += 1

        grid.attach(self._label("Container:"), 0, row, 1, 1)
        self.container_combo = Gtk.ComboBoxText()
        for ident, text in CONTAINERS:
            self.container_combo.append(ident, text)
        self.container_combo.set_active_id(DEFAULT_CONTAINER)
        grid.attach(self.container_combo, 1, row, 2, 1)
        row += 1

        grid.attach(self._label("Audio:"), 0, row, 1, 1)
        self.audio_combo = Gtk.ComboBoxText()
        for ident, text in AUDIO_MODES:
            self.audio_combo.append(ident, text)
        self.audio_combo.set_active_id(DEFAULT_AUDIO)
        grid.attach(self.audio_combo, 1, row, 1, 1)

        grid.attach(self._label("Start (s):"), 2, row, 1, 1)
        self.start_spin = self._spin(0, 0, 999999, 1, 30, digits=1, width=90)
        grid.attach(self.start_spin, 3, row, 1, 1)

        grid.attach(self._label("Limit (s):"), 4, row, 1, 1)
        self.limit_spin = self._spin(0, 0, 999999, 1, 30, digits=1, width=90)
        grid.attach(self.limit_spin, 5, row, 1, 1)
        row += 1

        grid.attach(self._label("Save to:"), 0, row, 1, 1)
        self.output_entry = Gtk.Entry()
        self.output_entry.set_text(str(OUTPUT_DIR))
        grid.attach(self.output_entry, 1, row, 4, 1)

        output_browse = Gtk.Button(label="Browse...")
        output_browse.connect("clicked", self.on_browse_output_dir)
        grid.attach(output_browse, 5, row, 1, 1)

        return expander

    def _label(self, text):
        label = Gtk.Label(label=text)
        label.set_xalign(0)
        return label

    def _build_console_tab(self):
        console_scrolled = Gtk.ScrolledWindow()
        console_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.console_text = Gtk.TextView()
        self.console_text.set_editable(False)
        self.console_text.set_wrap_mode(Gtk.WrapMode.CHAR)
        self.console_text.set_cursor_visible(False)

        console_scrolled.add(self.console_text)
        self.notebook.append_page(console_scrolled, Gtk.Label(label="Console Output"))

    def _build_preview_tab(self):
        preview_scrolled = Gtk.ScrolledWindow()
        preview_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.preview_image = Gtk.Image()
        preview_box.pack_start(self.preview_image, True, True, 0)

        self.preview_note = Gtk.Label(label="The newest upscaled frame appears here.")
        preview_box.pack_start(self.preview_note, False, False, 5)

        preview_scrolled.add(preview_box)
        self.notebook.append_page(preview_scrolled, Gtk.Label(label="Preview"))

    def _build_buttons(self):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

        self.start_button = Gtk.Button(label="Upscale")
        self.start_button.connect("clicked", self.on_start_clicked)
        self.start_button.set_sensitive(False)
        box.pack_start(self.start_button, True, True, 0)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.connect("clicked", self.on_stop_clicked)
        self.stop_button.set_sensitive(False)
        box.pack_start(self.stop_button, True, True, 0)

        self.delete_button = Gtk.Button(label="Delete Output")
        self.delete_button.connect("clicked", self.on_delete_clicked)
        self.delete_button.set_sensitive(False)
        box.pack_start(self.delete_button, True, True, 0)

        self.defaults_button = Gtk.Button(label="Restore Defaults")
        self.defaults_button.connect("clicked", self.on_restore_defaults_clicked)
        box.pack_start(self.defaults_button, True, True, 0)

        return box

    def update_status(self, message):
        if hasattr(sys.stdout, "write_with_newline"):
            sys.stdout.write_with_newline(f"{message}\n")
        else:
            print(f"{message}\n", end="")

    def _report_environment(self):
        print(f"{WINDOW_TITLE}: {describe_torch_build()}.")

        devices = available_devices()
        if devices == ["cpu"]:
            print(
                "No accelerator backend is compiled into this torch build, so "
                "everything runs on the CPU."
            )
        else:
            print(f"Devices this build can reach: {', '.join(devices)}.")
            for ident, label, _kind in ncnn_devices():
                print(f"  {ident}  {label.replace('Vulkan via ncnn - ', '')}")

        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            print(
                "Warning: ffmpeg and ffprobe are not on PATH. Install ffmpeg "
                "before upscaling anything."
            )

    def load_settings(self):
        self._loading_settings = True
        settings = {}
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r") as f:
                    settings = json.load(f)

                if settings.get("preset") in [p[0] for p in OUTPUT_PRESETS]:
                    self.preset_combo.set_active_id(settings["preset"])
                if "out_width" in settings:
                    self.out_width_spin.set_value(settings["out_width"])
                if "out_height" in settings:
                    self.out_height_spin.set_value(settings["out_height"])

                model = settings.get("model")
                if model:
                    known = [m[1] for m in BUILTIN_MODELS] + [CUSTOM_MODEL_ID]
                    if model in known:
                        self.model_combo.set_active_id(model)
                if "model_path" in settings:
                    self.model_entry.set_text(settings["model_path"])

                device = settings.get("device")
                if device and self.device_combo.set_active_id(device) is False:
                    self.device_combo.set_active_id("cpu")

                for key, spin in (
                    ("threads", self.threads_spin),
                    ("tile", self.tile_spin),
                    ("tile_pad", self.tile_pad_spin),
                    ("crf", self.crf_spin),
                    ("start", self.start_spin),
                    ("limit", self.limit_spin),
                ):
                    if key in settings:
                        spin.set_value(settings[key])

                for key, check in (
                    ("channels_last", self.channels_last_check),
                    ("deinterlace", self.deinterlace_check),
                    ("compile", self.compile_check),
                ):
                    if key in settings:
                        check.set_active(bool(settings[key]))

                if settings.get("precision") in PRECISION_DTYPES:
                    self.precision_combo.set_active_id(settings["precision"])

                for key, entry in (
                    ("filters", self.filters_pre_entry),
                    ("filters_pre", self.filters_pre_entry),
                    ("filters_post", self.filters_post_entry),
                ):
                    if key in settings:
                        entry.set_text(settings[key])

                if settings.get("encoder") in [e[0] for e in ENCODERS]:
                    self.encoder_combo.set_active_id(settings["encoder"])
                if settings.get("encoder_preset") in X264_PRESETS:
                    self.preset_encoder_combo.set_active_id(settings["encoder_preset"])
                if settings.get("container") in dict(CONTAINERS):
                    self.container_combo.set_active_id(settings["container"])
                if settings.get("audio") in [a[0] for a in AUDIO_MODES]:
                    self.audio_combo.set_active_id(settings["audio"])
                if settings.get("output_dir"):
                    self.output_entry.set_text(settings["output_dir"])
        except Exception as e:
            print(f"Error loading settings: {e}.")
        finally:
            self._loading_settings = False

        if isinstance(settings, dict):
            for key, value in self._auto_knobs().items():
                if key in settings and self._read_knob(key) != value:
                    self._user_set.add(key)

        self.on_model_changed()
        self.on_encoder_changed()
        self.on_output_changed()
        self.on_device_changed()

    def save_settings(self):
        try:
            settings = {
                "preset": self.preset_combo.get_active_id() or DEFAULT_PRESET,
                "out_width": int(self.out_width_spin.get_value()),
                "out_height": int(self.out_height_spin.get_value()),
                "model": self.model_combo.get_active_id() or DEFAULT_MODEL,
                "model_path": self.model_entry.get_text(),
                "device": self.device_combo.get_active_id() or "cpu",
                "threads": int(self.threads_spin.get_value()),
                "tile": int(self.tile_spin.get_value()),
                "tile_pad": int(self.tile_pad_spin.get_value()),
                "channels_last": self.channels_last_check.get_active(),
                "deinterlace": self.deinterlace_check.get_active(),
                "compile": self.compile_check.get_active(),
                "precision": (
                    self.precision_combo.get_active_id() or DEFAULT_PRECISION
                ),
                "filters_pre": self.filters_pre_entry.get_text(),
                "filters_post": self.filters_post_entry.get_text(),
                "encoder": self.encoder_combo.get_active_id() or DEFAULT_ENCODER,
                "encoder_preset": (
                    self.preset_encoder_combo.get_active_id() or DEFAULT_ENCODER_PRESET
                ),
                "crf": int(self.crf_spin.get_value()),
                "container": (
                    self.container_combo.get_active_id() or DEFAULT_CONTAINER
                ),
                "audio": self.audio_combo.get_active_id() or DEFAULT_AUDIO,
                "start": spin_value(self.start_spin),
                "limit": spin_value(self.limit_spin),
                "output_dir": self.output_entry.get_text(),
            }

            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}!")

    def on_source_entry_activate(self, entry):
        text = entry.get_text().strip()
        if text:
            self.set_source(text)

    def on_browse_source(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select a video",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        video_filter = Gtk.FileFilter()
        video_filter.set_name("Video files")
        for pattern in VIDEO_PATTERNS:
            video_filter.add_pattern(pattern)
            video_filter.add_pattern(pattern.upper())
        dialog.add_filter(video_filter)

        all_filter = Gtk.FileFilter()
        all_filter.set_name("All files")
        all_filter.add_pattern("*")
        dialog.add_filter(all_filter)

        if dialog.run() == Gtk.ResponseType.OK:
            selected = dialog.get_filename()
            if selected:
                dialog.destroy()
                self.set_source(selected)
                return

        dialog.destroy()

    def set_source(self, path):
        path = Path(path).expanduser()
        self.source_entry.set_text(str(path))

        if not path.is_file():
            self.source = None
            self.source_label.set_text(f"Not a file: {path}")
            self.start_button.set_sensitive(False)
            self.on_output_changed()
            return

        try:
            info = probe_video(path)
        except Exception as e:
            self.source = None
            self.source_label.set_text(str(e))
            self.start_button.set_sensitive(False)
            self.on_output_changed()
            print(f"Could not read {path.name}: {e}")
            return

        self.source = info
        aspect = ""
        if info["height"]:
            ratio = info["width"] / info["height"]
            aspect = f" ({ratio:.2f}:1)"

        details = [
            f"{info['width']}x{info['height']}{aspect}",
            f"{info['fps']:.3f} fps",
            _format_duration(info["duration"]),
            (
                f"{info['frames']} frames"
                if info["frames_exact"]
                else f"~{info['frames']} frames"
            ),
            info["codec"],
            info["pix_fmt"],
        ]
        if info["interlaced"]:
            details.append("interlaced")
        if info["has_audio"]:
            details.append("has audio")

        self.source_label.set_text(f"{path.name}\n" + "  |  ".join(details))
        self.hint_label.set_text("Ready.")

        if info["interlaced"] and not self.deinterlace_check.get_active():
            self.deinterlace_check.set_active(True)
            print("ffprobe reports interlaced fields, so Deinterlace is now on.")

        self.start_button.set_sensitive(not self.working)
        self.on_output_changed()

    def on_output_changed(self, widget=None):
        preset = self.preset_combo.get_active_id() or DEFAULT_PRESET
        custom = preset == "custom"
        self.out_width_spin.set_sensitive(custom)
        self.out_height_spin.set_sensitive(custom)

        if self.source is None:
            self.size_label.set_text("Waiting for a video.")
            return

        width, height = target_size(
            self.source,
            preset,
            int(self.out_width_spin.get_value()),
            int(self.out_height_spin.get_value()),
        )
        factor = width / max(self.source["width"], 1)
        self.size_label.set_text(
            f"{self.source['width']}x{self.source['height']}  ->  "
            f"{width}x{height}  (x{factor:.2f})"
        )

    def on_model_changed(self, widget=None):
        model = self.model_combo.get_active_id() or DEFAULT_MODEL
        custom = model == CUSTOM_MODEL_ID
        self.custom_model_box.set_sensitive(custom)

        if custom:
            self.model_note.set_text("Bring your own checkpoint.")
            return

        local = MODEL_DIR / model
        if local.exists():
            self.model_note.set_text(
                f"{_format_size(local.stat().st_size)} in {MODEL_DIR}"
            )
        else:
            self.model_note.set_text("Will be downloaded on first use.")

    def _auto_knobs(self):
        gpu = (self.device_combo.get_active_id() or "cpu").startswith("ncnn:")
        return {
            "tile": 0 if gpu else DEFAULT_TILE,
            "precision": "float16" if gpu else DEFAULT_PRECISION,
        }

    def _read_knob(self, key):
        if key == "tile":
            return int(self.tile_spin.get_value())
        return self.precision_combo.get_active_id() or ""

    def _write_knob(self, key, value):
        self._nudging = True
        try:
            if key == "tile":
                self.tile_spin.set_value(value)
            else:
                self.precision_combo.set_active_id(value)
        finally:
            self._nudging = False

    def _note_user_choice(self, widget, key):
        if not self._loading_settings and not self._nudging:
            self._user_set.add(key)

    def on_device_changed(self, widget=None):
        if self._loading_settings:
            return
        gpu = (self.device_combo.get_active_id() or "cpu").startswith("ncnn:")
        for key, value in self._auto_knobs().items():
            if key in self._user_set or self._read_knob(key) == value:
                continue
            self._write_knob(key, value)

    def on_encoder_changed(self, widget=None):
        lossless = (self.encoder_combo.get_active_id() or "") == "ffv1"
        self.crf_spin.set_sensitive(not lossless)
        self.preset_encoder_combo.set_sensitive(not lossless)

    def on_browse_model(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select upscaler weights",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        weights_filter = Gtk.FileFilter()
        weights_filter.set_name("Checkpoints")
        for pattern in ("*.pth", "*.pt", "*.safetensors", "*.bin"):
            weights_filter.add_pattern(pattern)
        dialog.add_filter(weights_filter)

        if dialog.run() == Gtk.ResponseType.OK:
            selected = dialog.get_filename()
            if selected:
                self.model_entry.set_text(selected)

        dialog.destroy()

    def on_browse_output_dir(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select an output folder",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        if dialog.run() == Gtk.ResponseType.OK:
            selected = dialog.get_filename()
            if selected:
                self.output_entry.set_text(selected)

        dialog.destroy()

    def on_restore_defaults_clicked(self, button):
        self._loading_settings = True
        self._user_set.clear()
        try:
            self._restore_defaults()
        finally:
            self._loading_settings = False
        self.on_device_changed()
        self.update_status("Settings restored to defaults.")

    def _restore_defaults(self):
        self.preset_combo.set_active_id(DEFAULT_PRESET)
        self.model_combo.set_active_id(DEFAULT_MODEL)
        self.model_entry.set_text("")
        self.device_combo.set_active_id(preferred_device())
        self.threads_spin.set_value(min(os.cpu_count() or 4, 64))
        self.tile_spin.set_value(DEFAULT_TILE)
        self.tile_pad_spin.set_value(DEFAULT_TILE_PAD)
        self.channels_last_check.set_active(True)
        self.deinterlace_check.set_active(False)
        self.compile_check.set_active(False)
        self.precision_combo.set_active_id(DEFAULT_PRECISION)
        self.filters_pre_entry.set_text("")
        self.filters_post_entry.set_text("")
        self.encoder_combo.set_active_id(DEFAULT_ENCODER)
        self.preset_encoder_combo.set_active_id(DEFAULT_ENCODER_PRESET)
        self.crf_spin.set_value(DEFAULT_CRF)
        self.container_combo.set_active_id(DEFAULT_CONTAINER)
        self.audio_combo.set_active_id(DEFAULT_AUDIO)
        self.start_spin.set_value(0)
        self.limit_spin.set_value(0)
        self.output_entry.set_text(str(OUTPUT_DIR))

    def _collect_job(self):
        info = self.source
        if info is None:
            return None

        preset = self.preset_combo.get_active_id() or DEFAULT_PRESET
        out_width, out_height = target_size(
            info,
            preset,
            int(self.out_width_spin.get_value()),
            int(self.out_height_spin.get_value()),
        )

        model_id = self.model_combo.get_active_id() or DEFAULT_MODEL
        if model_id == CUSTOM_MODEL_ID:
            weights = self.model_entry.get_text().strip()
            if not weights:
                self.update_status("Choose a weights file first.")
                return None
            model_path = Path(weights).expanduser()
            model_url = None
        else:
            model_path = MODEL_DIR / model_id
            model_url = next(u for _label, f, u in BUILTIN_MODELS if f == model_id)

        encoder = self.encoder_combo.get_active_id() or DEFAULT_ENCODER
        container = self.container_combo.get_active_id() or DEFAULT_CONTAINER
        output_dir = Path(
            self.output_entry.get_text().strip() or str(OUTPUT_DIR)
        ).expanduser()
        stem = Path(info["path"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = output_dir / (
            f"{stem}_{out_width}x{out_height}_{timestamp}."
            f"{container_for(encoder, container)}"
        )

        device = self.device_combo.get_active_id() or "cpu"
        precision = self.precision_combo.get_active_id() or DEFAULT_PRECISION
        if device.startswith("ncnn:") and precision == "bfloat16":
            print(
                "NCNN's Vulkan path has no bfloat16 here. Using float16, "
                "which is what a GPU wants anyway."
            )
        if device == "cpu" and precision == "bfloat16":
            if BF16_IN_HARDWARE:
                print(
                    "This CPU has bfloat16 instructions (AMX or AVX512-BF16), "
                    "so this should be a real speedup."
                )
            else:
                print(
                    "This CPU has no bfloat16 instructions, so torch will "
                    "emulate it and it will most likely be slower than "
                    "float32."
                )
        elif device == "cpu" and precision == "float16":
            print(
                "float16 exists for GPUs. On a CPU it is emulated and slower "
                "than float32."
            )

        return {
            "info": info,
            "out_width": out_width,
            "out_height": out_height,
            "model_path": model_path,
            "model_url": model_url,
            "device": device,
            "dtype": PRECISION_DTYPES[precision],
            "compile": self.compile_check.get_active(),
            "threads": int(self.threads_spin.get_value()),
            "tile": int(self.tile_spin.get_value()),
            "tile_pad": int(self.tile_pad_spin.get_value()),
            "channels_last": self.channels_last_check.get_active(),
            "deinterlace": self.deinterlace_check.get_active(),
            "filters_pre": self.filters_pre_entry.get_text(),
            "filters_post": self.filters_post_entry.get_text(),
            "encoder": encoder,
            "encoder_preset": (
                self.preset_encoder_combo.get_active_id() or DEFAULT_ENCODER_PRESET
            ),
            "crf": int(self.crf_spin.get_value()),
            "audio": self.audio_combo.get_active_id() or DEFAULT_AUDIO,
            "start": spin_value(self.start_spin),
            "limit": spin_value(self.limit_spin),
            "dest": dest,
        }

    def on_start_clicked(self, button):
        if self.working or self.source is None:
            return

        job = self._collect_job()
        if job is None:
            return

        self.save_settings()

        self.working = True
        self.stop_event.clear()
        self.start_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.delete_button.set_sensitive(False)
        self.progress.set_fraction(0.0)
        self.progress.set_text("Starting...")

        self.worker_thread = threading.Thread(
            target=self.upscale_thread, args=(job,), daemon=True
        )
        self.worker_thread.start()

    def on_delete_clicked(self, button):
        target = self.current_output
        if target is None:
            return
        try:
            if Path(target).exists():
                os.remove(target)
                self.update_status(f"Deleted {target}.")
            else:
                self.update_status(f"{target} is already gone.")
        except OSError as e:
            self.update_status(f"Could not delete {target}: {e}")
            return
        self.current_output = None
        self.delete_button.set_sensitive(False)

    def on_stop_clicked(self, button):
        if not self.working:
            return
        self.stop_event.set()
        self.update_status(
            "Stopping after the current frame. The encoder is left to close "
            "the file properly, so what has been written stays playable."
        )
        terminate_process(self.decoder)

    def _drain(self, stream, progress=False):
        collected = []

        def reader():
            try:
                for line in iter(stream.readline, b""):
                    text = line.decode("utf-8", "replace")
                    if progress:
                        match = _FFMPEG_PROGRESS.match(text.strip())
                        if match:
                            if match.group(1) == "total_size":
                                try:
                                    self._muxed_bytes = int(match.group(2))
                                except ValueError:
                                    pass
                            continue
                    collected.append(text)
                    print(f"ffmpeg: {text.rstrip()}", file=sys.stderr)
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        threading.Thread(target=reader, daemon=True).start()
        return collected

    def _download_progress(self, done, total):
        if total:
            fraction = done / total
            text = f"Downloading {_format_size(done)} / {_format_size(total)}"
        else:
            fraction = 0.0
            text = f"Downloading {_format_size(done)}"
        GLib.idle_add(self._set_progress, fraction, text)

    def _expected_frames(self, info, start, limit):
        span = 0.0
        if info["duration"] > 0:
            span = max(info["duration"] - start, 0.0)
        if limit > 0:
            span = min(span, limit) if span > 0 else limit
        if span > 0:
            return max(int(round(span * info["fps"])), 0)
        return info["frames"]

    def upscale_thread(self, job):
        started = time.monotonic()
        info = job["info"]
        encoder_errors = []

        try:
            torch.set_num_threads(max(1, job["threads"]))

            model_path = job["model_path"]
            if not model_path.exists():
                if not job["model_url"]:
                    raise FileNotFoundError(f"No such weights file: {model_path}")
                self.update_status(f"Downloading {model_path.name}...")
                download_model(
                    job["model_url"],
                    model_path,
                    progress=self._download_progress,
                    stop_check=self.stop_event.is_set,
                )
                self.update_status(f"Saved {model_path}.")

            self.update_status(f"Loading {model_path.name}...")
            on_ncnn = job["device"].startswith("ncnn:")
            if on_ncnn:
                loaded = load_ncnn_upscaler(
                    model_path,
                    gpu=int(job["device"].partition(":")[2] or 0),
                    threads=job["threads"],
                    fp16=job["dtype"] is not torch.float32,
                )
            else:
                loaded = load_upscaler(
                    model_path,
                    device=job["device"],
                    channels_last=job["channels_last"],
                    dtype=job["dtype"],
                )
            model, native_scale, size_multiple, min_overlap, description = loaded

            torch_device = "cpu" if on_ncnn else job["device"]
            torch_dtype = torch.float32 if on_ncnn else job["dtype"]
            channels_last = False if on_ncnn else job["channels_last"]
            if on_ncnn:
                self.update_status(f"{description}.")
            else:
                self.update_status(
                    f"{description}, running on {job['device']} in "
                    f"{str(torch_dtype).replace('torch.', '')}."
                )

            if job["compile"] and not on_ncnn:
                self.update_status(
                    "Compiling with torch.compile. The first few frames pay "
                    "for it so the rest should be faster."
                )
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(
                        f"torch.compile is not usable here ({e}). Carrying on "
                        "without it."
                    )

            tile = job["tile"]
            tile_pad = job["tile_pad"]
            if tile > 0 and tile_pad < min_overlap:
                if min_overlap * 2 <= tile:
                    tile_pad = min_overlap
                    self.update_status(
                        f"Raised the tile overlap to {tile_pad} px, the radius "
                        "this network actually reads. Tiled output now matches "
                        "whole-frame output exactly."
                    )
                else:
                    self.update_status(
                        f"Note: this network reads {min_overlap} px around every "
                        f"output pixel but the tiles only carry {tile_pad} px of "
                        "context, so tile seams may show and shimmer between "
                        "frames. Set Tile to 0 to process whole frames, or raise "
                        "Tile and Overlap if there is memory for it."
                    )

            if self.stop_event.is_set():
                raise KeyboardInterrupt()

            source_width, source_height = info["width"], info["height"]
            filters = source_filters(job["deinterlace"], job["filters_pre"])
            if filters:
                measured = probe_filtered_size(info["path"], filters)
                if measured is None:
                    print(
                        "Warning: could not measure the frame size after "
                        f"'{filters}'. Assuming the filters leave it at "
                        f"{source_width}x{source_height}."
                    )
                elif measured != (source_width, source_height):
                    source_width, source_height = measured
                    self.update_status(
                        f"'{filters}' resizes the source to "
                        f"{source_width}x{source_height} so upscaling from that."
                    )

            model_width = source_width * native_scale
            model_height = source_height * native_scale
            out_width, out_height = job["out_width"], job["out_height"]
            if (out_width, out_height) != (model_width, model_height):
                self.update_status(
                    f"The network outputs {model_width}x{model_height}. FFmpeg "
                    f"will resample that to {out_width}x{out_height} (lanczos)."
                )

            post_filters = (job["filters_post"] or "").strip()
            if post_filters:
                measured = probe_output_size(out_width, out_height, post_filters)
                if measured is None:
                    print(
                        "Warning: could not measure the frame size after "
                        f"'{post_filters}'. FFmpeg will apply it anyway."
                    )
                elif measured != (out_width, out_height):
                    self.update_status(
                        f"'{post_filters}' runs after the upscale, so the file "
                        f"ends up {measured[0]}x{measured[1]}, not "
                        f"{out_width}x{out_height}."
                    )

            total = self._expected_frames(info, job["start"], job["limit"])
            self._total_label = str(total) if info["frames_exact"] else f"~{total}"

            decoder_command = build_decoder_command(
                info,
                job["start"],
                job["limit"],
                job["deinterlace"],
                job["filters_pre"],
            )
            encoder_command = build_encoder_command(
                info,
                model_width,
                model_height,
                out_width,
                out_height,
                info["fps"],
                job["dest"],
                job["encoder"],
                job["crf"],
                job["encoder_preset"],
                job["audio"],
                job["start"],
                job["limit"],
                job["filters_post"],
            )
            print(f"decode: {shlex.join(decoder_command)}")
            print(f"encode: {shlex.join(encoder_command)}")

            job["dest"].parent.mkdir(parents=True, exist_ok=True)

            self.decoder = subprocess.Popen(
                decoder_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.encoder = subprocess.Popen(
                encoder_command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._muxed_bytes = 0
            self._drain(self.decoder.stderr)
            encoder_errors = self._drain(self.encoder.stderr, progress=True)

            stop_check = self.stop_event.is_set
            started = time.monotonic()

            def on_frame(index, result):
                preview = self._maybe_preview(result)
                if preview is not None:
                    GLib.idle_add(self._show_preview, *preview)
                self._report_progress(index, total, started)

            try:
                frames_done = process_stream(
                    model,
                    self.decoder.stdout,
                    self.encoder.stdin,
                    source_width,
                    source_height,
                    native_scale,
                    tile,
                    tile_pad,
                    size_multiple=size_multiple,
                    device=torch_device,
                    dtype=torch_dtype,
                    channels_last=channels_last,
                    stop_check=stop_check,
                    on_frame=on_frame,
                )
            except BrokenPipeError:
                raise RuntimeError(
                    "ffmpeg closed the pipe:\n" + "".join(encoder_errors).strip()
                )

            stopped = self.stop_event.is_set()

            try:
                self.encoder.stdin.close()
            except OSError:
                pass

            encoder_status = self.encoder.wait()
            try:
                self.decoder.stdout.close()
            except OSError:
                pass
            decoder_status = self.decoder.wait()

            if encoder_status != 0:
                raise RuntimeError(
                    f"ffmpeg exited with {encoder_status} while encoding:\n"
                    + "".join(encoder_errors).strip()
                )
            if not stopped and decoder_status not in (0, -13, 255):
                print(f"Note: the decoder exited with {decoder_status}.")

            elapsed = time.monotonic() - started
            size = job["dest"].stat().st_size if job["dest"].exists() else 0
            rate = frames_done / elapsed if elapsed > 0 else 0.0
            self.current_output = job["dest"]

            if stopped:
                self.update_status(
                    f"Stopped after {frames_done} frames. {job['dest']} holds "
                    f"what was finished ({_format_size(size)})."
                )
                GLib.idle_add(self._set_progress, 0.0, "Stopped")
            else:
                self.update_status(
                    f"Done. {frames_done} frames in {_format_duration(elapsed)} "
                    f"({rate:.2f} fps), {_format_size(size)} written to "
                    f"{job['dest']}."
                )
                GLib.idle_add(self._set_progress, 1.0, "Finished")

        except KeyboardInterrupt:
            self.update_status("Stopped.")
            GLib.idle_add(self._set_progress, 0.0, "Stopped")
        except Exception as e:
            if self.stop_event.is_set():
                self.update_status("Stopped.")
            else:
                traceback.print_exc()
                self.update_status(f"Error: {e}")
            GLib.idle_add(self._set_progress, 0.0, "Failed")
        finally:
            terminate_process(self.decoder)
            terminate_process(self.encoder)
            self.decoder = None
            self.encoder = None
            gc.collect()
            GLib.idle_add(self._finish)

    def _report_progress(self, done, total, started):
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0

        if total > 0:
            fraction = min(done / total, 1.0)
            remaining = (total - done) / rate if rate > 0 else None
            text = (
                f"Frame {done}/{self._total_label}  |  {rate:.2f} fps  |  "
                f"elapsed {_format_duration(elapsed)}  |  "
                f"left {_format_duration(remaining)}"
            )
        else:
            fraction = 0.0
            text = (
                f"Frame {done}  |  {rate:.2f} fps  |  "
                f"elapsed {_format_duration(elapsed)}"
            )

        written = self._muxed_bytes
        if written > 1 << 20 and done >= 100:
            text += f"  |  {_format_size(written)}"
            if total > 0:
                text += f", ~{_format_size(written / done * total)} projected"

        GLib.idle_add(self._set_progress, fraction, text)

    def _maybe_preview(self, rgb):
        now = time.monotonic()
        if now - self._last_preview < PREVIEW_INTERVAL:
            return None
        self._last_preview = now

        try:
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            step = max(1, -(-max(height, width) // PREVIEW_MAX_SIZE))
            small = rgb[::step, ::step].contiguous()
            return (small.numpy().tobytes(), int(small.shape[1]), int(small.shape[0]))
        except Exception as e:
            print(f"Preview failed: {e}.", file=sys.stderr)
            return None

    def _set_progress(self, fraction, text):
        self.progress.set_fraction(max(0.0, min(1.0, fraction)))
        self.progress.set_text(text)
        return False

    def _show_preview(self, rgb_bytes, width, height):
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                GLib.Bytes.new(rgb_bytes),
                GdkPixbuf.Colorspace.RGB,
                False,
                8,
                width,
                height,
                width * 3,
            )
            self.preview_image.set_from_pixbuf(pixbuf)
            self.preview_note.set_text(f"Newest frame, shown at {width}x{height}.")
        except Exception as e:
            print(f"Could not display the preview: {e}.", file=sys.stderr)
        return False

    def _finish(self):
        self.working = False
        self.start_button.set_sensitive(self.source is not None)
        self.stop_button.set_sensitive(False)
        self.delete_button.set_sensitive(
            self.current_output is not None and Path(self.current_output).exists()
        )
        self.on_model_changed()
        return False

    def on_window_close(self, widget, event):
        self.save_settings()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        if self.working:
            self.stop_event.set()
            terminate_process(self.decoder)
            terminate_process(self.encoder)

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=WORKER_THREAD_TIMEOUT)

        return False


def _has_encoder(name):
    try:
        listed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return f" {name} " in listed.stdout.decode("utf-8", "replace")


def _self_test_ncnn(check):
    try:
        import ncnn  # noqa: F401
    except Exception:
        return

    import tempfile

    torch.manual_seed(0)
    reference = SRVGGNetCompact(num_feat=16, num_conv=3, upscale=2)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    frame = torch.rand(1, 3, 24, 32)
    with torch.inference_mode():
        want = reference(frame)

    with tempfile.TemporaryDirectory(prefix="animus-ncnn-") as workdir:
        weights = Path(workdir) / "probe.pth"
        torch.save({"params": reference.state_dict()}, weights)

        targets = [(None, "CPU")]
        targets += [
            (int(ident.partition(":")[2]), label)
            for ident, label, _kind in ncnn_devices()
        ]

        for gpu, label in targets:
            for fp16 in ((False,) if gpu is None else (False, True)):
                try:
                    model, scale, multiple, overlap, _d = load_ncnn_upscaler(
                        weights, gpu=gpu, threads=2, fp16=fp16
                    )
                    got = model(frame)
                except Exception as e:
                    check(f"ncnn {label} runs", False, f"{type(e).__name__}: {e}")
                    continue

                precision = "fp16" if fp16 else "fp32"
                name = f"ncnn {label} {precision}"
                if gpu is not None:
                    check(
                        f"{name} really ran on the GPU",
                        model.on_gpu,
                        (
                            "Vulkan"
                            if model.on_gpu
                            else "ncnn fell back to its CPU backend"
                        ),
                    )
                check(
                    f"{name} shape",
                    tuple(got.shape) == tuple(want.shape),
                    f"{tuple(got.shape)} against torch {tuple(want.shape)}",
                )
                if tuple(got.shape) != tuple(want.shape):
                    continue

                delta = (got - want).abs().max().item()
                limit = 6e-2 if fp16 else 1e-4
                check(
                    f"{name} matches torch",
                    delta < limit,
                    f"max |diff| = {delta:.2e} ({delta * 255:.2f}/255)",
                )
                model.close()


def _self_test_pipeline(check):
    import tempfile

    encoder = "libx264" if _has_encoder("libx264") else "ffv1"
    with tempfile.TemporaryDirectory(prefix="animus-selftest-") as workdir:
        work = Path(workdir)
        source = work / "source.mkv"

        made = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=10:duration=0.5",
                "-c:v",
                "ffv1",
                str(source),
            ],
            stderr=subprocess.PIPE,
            check=False,
        )
        if made.returncode != 0:
            check(
                "ffmpeg makes a test clip",
                False,
                made.stderr.decode("utf-8", "replace").strip(),
            )
            return

        for filters, want in (
            ("", (64, 48)),
            ("yadif=0:-1:0", (64, 48)),
            ("scale=32:24", (32, 24)),
            ("crop=40:20:0:0", (40, 20)),
        ):
            got = probe_filtered_size(source, filters)
            check(
                f"frame size after '{filters or 'no filters'}'", got == want, str(got)
            )
        check(
            "an unusable filter chain reports nothing",
            probe_filtered_size(source, "not_a_real_filter=1") is None,
        )

        for filters, want in (
            ("", (128, 96)),
            ("crop=100:80:0:0", (100, 80)),
            ("scale=64:48:flags=lanczos,crop=32:24:0:0", (32, 24)),
        ):
            got = probe_output_size(128, 96, filters)
            check(f"128x96 after '{filters or 'no filters'}'", got == want, str(got))

        info = probe_video(source)
        check(
            "ffprobe reads the clip",
            (info["width"], info["height"]) == (64, 48) and info["frames"] > 0,
            f"{info['width']}x{info['height']}, {info['frames']} frames, "
            f"{info['fps']:.2f} fps",
        )

        model = SRVGGNetCompact(num_feat=8, num_conv=2, upscale=2)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        out_width, out_height = target_size(info, "x2", 0, 0)
        check(
            "target size from the clip",
            (out_width, out_height) == (128, 96),
            f"{out_width}x{out_height}",
        )

        dest = work / f"out.{container_for(encoder)}"
        decoder = subprocess.Popen(
            build_decoder_command(info, 0, 0, False, ""),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        writer = subprocess.Popen(
            build_encoder_command(
                info,
                128,
                96,
                out_width,
                out_height,
                info["fps"],
                dest,
                encoder,
                20,
                "veryfast",
                "copy",
                0,
                0,
                "",
            ),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frames = process_stream(
            model, decoder.stdout, writer.stdin, info["width"], info["height"], 2, 32, 4
        )
        writer.stdin.close()
        noise = writer.stderr.read().decode("utf-8", "replace")
        errors = "\n".join(
            line
            for line in noise.splitlines()
            if line.strip() and not _FFMPEG_PROGRESS.match(line.strip())
        ).strip()
        encoded = writer.wait()
        decoder.wait()

        check(
            f"{encoder} round trip",
            encoded == 0 and dest.exists(),
            errors or f"{frames} frames",
        )
        if not dest.exists():
            return

        result = probe_video(dest)
        check(
            "encoded size",
            (result["width"], result["height"]) == (out_width, out_height),
            f"{result['width']}x{result['height']}",
        )
        check(
            "every frame written",
            result["frames"] == frames,
            f"{result['frames']} of {frames}",
        )


def measure_reach(model, scale, size_multiple, expected):
    margin = expected + 4
    size = 2 * margin
    size += (-size) % max(size_multiple, 1)
    centre = size // 2
    centre -= centre % max(size_multiple, 1)

    base = torch.rand(1, 3, size, size)
    probed = base.clone()
    probed[0, :, centre, centre] += 10.0

    with torch.inference_mode():
        difference = (model(probed) - model(base)).abs().sum(dim=1)[0] != 0

    rows = torch.nonzero(difference.any(dim=1)).flatten()
    cols = torch.nonzero(difference.any(dim=0)).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None, False

    top, bottom = int(rows[0]) // scale, int(rows[-1]) // scale
    left, right = int(cols[0]) // scale, int(cols[-1]) // scale
    reach = max(centre - top, bottom - centre, centre - left, right - centre)
    clipped = top <= 0 or left <= 0 or bottom >= size - 1 or right >= size - 1
    return reach, clipped


def benchmark(target=None):
    print(describe_torch_build() + ".")
    print(f"Devices: {', '.join(available_devices())}.\n")

    width, height, frames = 640, 480, 0
    source = "a 640x480 frame"
    if target:
        try:
            info = probe_video(target)
            width, height = info["width"], info["height"]
            frames = info["frames"]
            source = (
                f"{width}x{height} from {Path(target).name}, "
                f"{frames} frames at {info['fps']:.3f} fps"
            )
        except Exception as e:
            print(f"Could not read {target}: {e}\nFalling back to 640x480.\n")

    installed = [
        (label, MODEL_DIR / filename)
        for label, filename, _url in BUILTIN_MODELS
        if (MODEL_DIR / filename).exists()
    ]
    if not installed:
        print(
            f"No weights in {MODEL_DIR} yet. Run one model from the window "
            "once to fetch it, then come back."
        )
        return 1

    print(f"Timing {source}. This takes a minute.\n")
    frame = torch.rand(1, 3, height, width).contiguous(
        memory_format=torch.channels_last
    )

    def time_model(path, threads, device="cpu", tile=DEFAULT_TILE):
        torch.set_num_threads(threads)
        if device == "ncnn-cpu":
            model, scale, multiple, overlap, description = load_ncnn_upscaler(
                path,
                gpu=None,
                threads=threads,
                fp16=False,
            )
        elif device.startswith("ncnn:"):
            model, scale, multiple, overlap, description = load_ncnn_upscaler(
                path,
                gpu=int(device.partition(":")[2] or 0),
                threads=threads,
                fp16=True,
            )
        else:
            model, scale, multiple, overlap, description = load_upscaler(
                path, "cpu", channels_last=True
            )

        def once():
            return upscale_frame(model, frame, scale, tile, overlap, multiple)

        with torch.inference_mode():
            started = time.monotonic()
            once()
            first = time.monotonic() - started
            once()

            repeats = max(1, min(5, int(6.0 / max(first, 0.05))))
            started = time.monotonic()
            for _ in range(repeats):
                once()
            return (time.monotonic() - started) / repeats, description

    default_threads = min(os.cpu_count() or 4, 64)
    print(f"  {'model':26s} {'s/frame':>9s} {'fps':>7s}  whole job")
    print(f"  {'-' * 26} {'-' * 9} {'-' * 7}  {'-' * 12}")

    best = None
    for label, path in installed:
        seconds, _description = time_model(path, default_threads)
        projection = _format_duration(seconds * frames) if frames else "-"
        print(f"  {label:26s} {seconds:9.2f} {1 / seconds:7.2f}  {projection}")
        if best is None or seconds < best[0]:
            best = (seconds, label, path)

    total = os.cpu_count() or 4
    counts = sorted({1, max(1, total // 4), max(1, total // 2), total, default_threads})
    if len(counts) > 1:
        print(
            f"\n  Thread counts, on {best[1]} / cpu (the torch CPU path, "
            f"not a GPU) - {total} logical CPUs:"
        )
        for threads in counts:
            seconds, _description = time_model(best[2], threads)
            projection = _format_duration(seconds * frames) if frames else "-"
            marker = "  <- default" if threads == default_threads else ""
            print(
                f"  {threads:3d} threads{'':14s} {seconds:9.2f} "
                f"{1 / seconds:7.2f}  {projection}{marker}"
            )

    listed = ncnn_devices()
    accelerators = [i for i, _l, kind in listed if kind != "software"]
    skipped = [i for i, _l, kind in listed if kind == "software"]
    if not accelerators:
        accelerators, skipped = [i for i, _l, _k in listed], []
    timings = {}
    if accelerators:
        print(f"\n  Devices, on {best[1]}:")
        baseline = None
        for device in ["cpu", "ncnn-cpu"] + accelerators:
            tile = 0 if device.startswith("ncnn:") else DEFAULT_TILE
            try:
                seconds, _description = time_model(
                    best[2], default_threads, device, tile=tile
                )
            except Exception as e:
                print(f"  {device:26s} unavailable: {e}")
                continue
            timings[device] = seconds
            if baseline is None:
                baseline = seconds
            projection = _format_duration(seconds * frames) if frames else "-"
            speedup = f"{baseline / seconds:.1f}x" if seconds else "-"
            shape = "whole frame" if tile == 0 else f"tile {tile}"
            print(
                f"  {device:14s} {shape:12s} {seconds:9.2f} "
                f"{1 / seconds:7.2f}  {projection}  {speedup}"
            )

    if skipped:
        print(f"  (skipped {', '.join(skipped)}: software rasterisers)")

    fastest = min(timings, key=timings.get) if timings else "cpu"
    print(f"\n  Tile sizes, on {best[1]} / {fastest}:")
    for tile in (0, 128, 256, 384, 512):
        try:
            seconds, _description = time_model(
                best[2], default_threads, fastest, tile=tile
            )
        except Exception as e:
            print(f"  {'tile ' + str(tile):26s} unavailable: {e}")
            continue
        projection = _format_duration(seconds * frames) if frames else "-"
        label = "whole frame" if tile == 0 else f"tile {tile}"
        print(f"  {label:26s} {seconds:9.2f} {1 / seconds:7.2f}  {projection}")

    torch.set_num_threads(default_threads)
    print(
        "\nMore threads is not always faster: on a CPU with both performance "
        "and efficiency cores, every parallel region waits for the slowest "
        "core in it."
    )
    return 0


def self_test():
    print(describe_torch_build() + ".")
    print(f"Devices: {', '.join(available_devices())}.")

    cases = (
        (
            "SRVGG x4 prelu",
            SRVGGNetCompact(num_feat=16, num_conv=4, upscale=4, act_type="prelu"),
            4,
            1,
        ),
        (
            "SRVGG x2 relu",
            SRVGGNetCompact(num_feat=16, num_conv=2, upscale=2, act_type="relu"),
            2,
            1,
        ),
        ("RRDB x4", RRDBNet(scale=4, num_feat=8, num_block=2, num_grow_ch=4), 4, 1),
        ("RRDB x2", RRDBNet(scale=2, num_feat=8, num_block=2, num_grow_ch=4), 2, 2),
        ("RRDB x1", RRDBNet(scale=1, num_feat=8, num_block=1, num_grow_ch=4), 1, 4),
    )

    failures = []

    def check(label, ok, detail=""):
        print(
            f"{'ok  ' if ok else 'FAIL'}  {label}" + (f" - {detail}" if detail else "")
        )
        if not ok:
            failures.append(label)

    torch.manual_seed(0)
    frame = torch.rand(1, 3, 37, 53)
    wide = F.interpolate(
        torch.rand(1, 3, 24, 24), size=(192, 192), mode="bicubic"
    ).clamp(0, 1)

    for label, reference, want_scale, want_multiple in cases:
        reference.eval()
        state_dict = reference.state_dict()

        try:
            model, scale, multiple, overlap, description = build_upscaler(state_dict)
        except Exception as e:
            check(f"{label}: detected", False, str(e))
            continue

        check(f"{label}: architecture", type(model) is type(reference), description)
        check(f"{label}: scale", scale == want_scale, f"got {scale}")
        check(f"{label}: size multiple", multiple == want_multiple, f"got {multiple}")

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            check(f"{label}: strict load", False, str(e))
            continue
        check(f"{label}: strict load", True, f"overlap {overlap} px")

        model.eval()
        with torch.inference_mode():
            want = upscale_frame(reference, frame, scale, 0, 0, multiple)
            whole = upscale_frame(model, frame, scale, 0, 0, multiple)
            tiled = upscale_frame(model, frame, scale, 16, overlap, multiple)

        height, width = frame.shape[-2:]
        shape = (1, 3, height * scale, width * scale)
        check(
            f"{label}: output shape",
            tuple(whole.shape) == shape,
            str(tuple(whole.shape)),
        )
        check(f"{label}: finite", bool(torch.isfinite(whole).all()))
        check(f"{label}: matches the reference module", torch.equal(whole, want))

        delta = (whole - tiled).abs().max().item()
        check(
            f"{label}: tiling with full overlap matches whole frames",
            delta < 1e-4,
            f"max |diff| = {delta:.2e}",
        )

        reach, clipped = measure_reach(model, scale, multiple, overlap)
        if reach is None:
            check(f"{label}: reach measurable", False, "the probe did not" " propagate")
        else:
            check(
                f"{label}: min_overlap covers the {reach} px it reads",
                reach <= overlap and not clipped,
                f"reads {reach} px, carries {overlap}"
                + (", and the probe hit the edge" if clipped else ""),
            )

        stopped = upscale_frame(
            model, wide, scale, 32, overlap, multiple, stop_check=lambda: True
        )
        check(f"{label}: a stop unwinds without raising", stopped is None)

    sizes = (
        (("x2", 640, 480, 0, 0), (1280, 960)),
        (("x4", 640, 480, 0, 0), (2560, 1920)),
        (("h720", 640, 480, 0, 0), (960, 720)),
        (("h960", 640, 480, 0, 0), (1280, 960)),
        (("h1080", 720, 576, 0, 0), (1350, 1080)),
        (("custom", 640, 480, 1281, 961), (1280, 960)),
    )
    for (preset, width, height, custom_w, custom_h), want in sizes:
        got = target_size(
            {"width": width, "height": height}, preset, custom_w, custom_h
        )
        check(f"target_size {preset} from {width}x{height}", got == want, str(got))

    chains = (
        ("nothing to do", (128, 96, 128, 96, ""), ""),
        (
            "the extra chain alone",
            (128, 96, 128, 96, "crop=100:80:0:0"),
            "crop=100:80:0:0",
        ),
        ("the resample alone", (128, 96, 64, 48, ""), "scale=64:48:flags=lanczos"),
        (
            "the resample and then the extra chain",
            (128, 96, 64, 48, "crop=32:24:0:0"),
            "scale=64:48:flags=lanczos,crop=32:24:0:0",
        ),
    )
    for label, arguments, want in chains:
        got = output_filters(*arguments)
        check(f"output filters: {label}", got == want, got or "(none)")

    _self_test_ncnn(check)

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        _self_test_pipeline(check)
    else:
        print(
            "note  ffmpeg and ffprobe are not in the PATH, so the video pipeline "
            "was not exercised"
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


def main():
    arguments = sys.argv[1:]

    if "--self-test" in arguments:
        sys.exit(self_test())

    if "--benchmark" in arguments:
        rest = [a for a in arguments if a != "--benchmark"]
        sys.exit(benchmark(rest[0] if rest else None))

    if any(a in ("-h", "--help") for a in arguments):
        print("Usage: animus-upscale [--self-test] [--benchmark] [VIDEO]")
        print()
        print("  --self-test   check the networks and the ffmpeg pipeline")
        print("  --benchmark   time the installed models")
        sys.exit(0)

    def sigint_handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    window = UpscaleGUI()
    window.connect("delete-event", window.on_window_close)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.set_focus(None)

    if len(sys.argv) > 1:
        window.set_source(sys.argv[1])

    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received - shutting down...")
        window.stop_event.set()
        terminate_process(window.decoder)
        terminate_process(window.encoder)
        sys.exit(0)


if __name__ == "__main__":
    main()
