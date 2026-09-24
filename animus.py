#!/usr/bin/env python3

# autopep8: off
# isort: off

import os
import sys

TQDM_UPDATE_INTERVAL = 0.05
TQDM_MIN_ITERATIONS = 1
TQDM_MAX_INTERVAL = 1.0

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_TQDM_MININTERVAL"] = str(TQDM_UPDATE_INTERVAL)
os.environ["HF_HUB_TQDM_ASCII"] = "1"

if not os.environ.get("PYTHONIOENCODING"):
    os.environ["PYTHONIOENCODING"] = "utf-8"

import ctypes
import errno
import contextlib
import gc
import json
import logging as stdlib_logging
import math
import queue
import random
import re
import shlex
import shutil
import signal
import struct
import subprocess
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import ClassVar


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

warnings.filterwarnings(
    "ignore", message=".*device_type of 'cuda', but CUDA is not available.*"
)

import numpy
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import logging

try:
    from diffusers import (
        ModularPipeline,
        CosmosTransformer3DModel,
        GGUFQuantizationConfig,
        FlowMatchEulerDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    ANIMA_AVAILABLE = True
except Exception:  # pragma: Requires diffusers >= 0.39.0.  # noqa: BLE001
    ModularPipeline = None
    CosmosTransformer3DModel = None
    GGUFQuantizationConfig = None
    FlowMatchEulerDiscreteScheduler = None
    UniPCMultistepScheduler = None
    ANIMA_AVAILABLE = False

try:
    from diffusers import AutoencoderKL

    VAE_AVAILABLE = ANIMA_AVAILABLE
except Exception:  # pragma: Requires diffusers.  # noqa: BLE001
    AutoencoderKL = None
    VAE_AVAILABLE = False

try:
    from diffusers import ClassifierFreeGuidance
except Exception:  # noqa: BLE001
    try:
        from diffusers.guiders import ClassifierFreeGuidance
    except Exception:  # noqa: BLE001
        ClassifierFreeGuidance = None

from PIL import Image as _PILImage
from PIL.PngImagePlugin import PngInfo

_PILImage.preinit()

import animus_vulkan

# isort: on
# autopep8: on

torch.backends.nnpack.enabled = False

logging.set_verbosity_error()

warnings.filterwarnings("ignore", message=".*peft_config.*multiple adapters.*")

try:
    import tqdm as tqdm_module

    _original_tqdm_init = tqdm_module.tqdm.__init__

    def _patched_tqdm_init(self, *args, **kwargs):
        kwargs = kwargs.copy() if kwargs else {}

        if "ascii" not in kwargs:
            kwargs["ascii"] = True

        if "mininterval" not in kwargs:
            kwargs["mininterval"] = TQDM_UPDATE_INTERVAL

        if "maxinterval" not in kwargs:
            kwargs["maxinterval"] = TQDM_MAX_INTERVAL

        if "miniters" not in kwargs:
            kwargs["miniters"] = TQDM_MIN_ITERATIONS

        return _original_tqdm_init(self, *args, **kwargs)

    tqdm_module.tqdm.__init__ = _patched_tqdm_init
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not patch tqdm: {e}.", file=sys.stderr)
except Exception as e:  # noqa: BLE001
    print(f"Warning: unexpected error while patching tqdm: {e}.", file=sys.stderr)

APP_NAME = "animus"
WINDOW_TITLE = "Animus"

WINDOW_WIDTH = 920
WINDOW_HEIGHT = 900

CONTROLS_HEIGHT = 400

CONSOLE_CSS_CLASS = "animus-console"
CONSOLE_CSS = f"""
textview.{CONSOLE_CSS_CLASS},
textview.{CONSOLE_CSS_CLASS} text {{
    font-family: monospace;
    font-size: 12pt;
}}
""".encode()
PROGRESS_CSS = b"""
progressbar {
    font-size: inherit;
    color: inherit;
}
"""


def _xdg_home(variable, fallback):
    value = os.environ.get(variable)
    if value and os.path.isabs(value):
        return Path(value)
    return Path.home() / fallback


CONFIG_DIR = _xdg_home("XDG_CONFIG_HOME", ".config") / APP_NAME
DATA_DIR = _xdg_home("XDG_DATA_HOME", ".local/share") / APP_NAME

LEGACY_DIR = Path.home() / ".config" / APP_NAME

_RELOCATED = []


def _relocate(name, parent):
    dest = parent / name
    src = LEGACY_DIR / name

    if src == dest or dest.exists() or not src.exists() or src.is_symlink():
        return dest

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dest)
    except OSError as e:
        print(f"Warning: could not move {src} to {dest}: {e}. Still using {src}.")
        return src

    _RELOCATED.append(name)
    print(f"Moved {src} to {dest}.")
    return dest


CONFIG_FILE = _relocate("settings.json", CONFIG_DIR)
IMAGE_DIR = _relocate("outputs", DATA_DIR)
LORA_DIR = _relocate("loras", DATA_DIR)
DIT_DIR = _relocate("models", DATA_DIR)


def _prune_embedding_dir():
    for stale in (DATA_DIR / "embeddings", LEGACY_DIR / "embeddings"):
        try:
            stale.rmdir()
        except OSError:
            continue
        print(f"Removed the unused {stale}.")


_prune_embedding_dir()


def _repoint_settings_paths():
    moves = [
        (LEGACY_DIR / "models", DIT_DIR),
        (LEGACY_DIR / "loras", LORA_DIR),
    ]
    moves = [(old, new) for old, new in moves if old != new]

    if not CONFIG_FILE.exists():
        return

    def repoint(value):
        for old, new in moves:
            try:
                return str(new / Path(value).relative_to(old))
            except ValueError:
                continue
        return value

    try:
        with open(CONFIG_FILE, "r") as f:
            settings = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Warning: could not read {CONFIG_FILE} to update its paths: {e}.")
        return

    if not isinstance(settings, dict):
        return

    section = settings.get("generate")
    if not isinstance(section, dict):
        section = settings

    changed = False

    model = section.get("model")
    if isinstance(model, str):
        moved = repoint(model)
        if moved != model:
            section["model"] = moved
            changed = True

    for entry in section.get("loras") or []:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        moved = repoint(path)
        if moved != path:
            entry["path"] = moved
            changed = True

    for key in ("embeddings", "embedding"):
        if key in section:
            del section[key]
            changed = True

    if not changed:
        return

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except OSError as e:
        print(f"Warning: could not rewrite the paths in {CONFIG_FILE}: {e}.")
        return

    print(f"Rewrote the relocated paths in {CONFIG_FILE}.")


_repoint_settings_paths()


def _prune_legacy_dir():
    if not _RELOCATED:
        return

    if LEGACY_DIR != CONFIG_DIR:
        try:
            LEGACY_DIR.rmdir()
            print(f"Removed the now-empty {LEGACY_DIR}.")
            return
        except OSError:
            pass

    try:
        leftover = sorted(p.name for p in LEGACY_DIR.iterdir() if p != CONFIG_FILE)
    except OSError:
        return

    if leftover:
        print(f"Left as-is in {LEGACY_DIR}: {', '.join(leftover)}.")


_prune_legacy_dir()

VIDEO_DIR = DATA_DIR / "upscales"
UPSCALER_DIR = DATA_DIR / "upscalers"

LEGACY_UPSCALE_FILE = CONFIG_DIR / "upscale.json"


MIN_FREE_DISK_MARGIN = 256 * 1024 * 1024

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


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
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def spin_value(spin):
    return round(spin.get_value(), spin.get_digits())


def spin_text(spin):
    return f"{spin_value(spin):.{spin.get_digits()}f}"


def even(value):
    value = round(value)
    return max(2, value - (value % 2))


def update_status(message):
    if hasattr(sys.stdout, "write_with_newline"):
        sys.stdout.write_with_newline(f"{message}\n")
    else:
        print(f"{message}\n", end="")


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


GENERATION_THREAD_TIMEOUT = 5.0
LOAD_THREAD_TIMEOUT = 3.0

NUM_LORA_SLOTS = 4

ANIMA_COMPONENTS_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
ANIMA_DEFAULT_DIT = (
    "https://huggingface.co/dancingmirrors/Anima/anima-turbo-v1.0-Q4_K_M.gguf"
)
ANIMA_DEFAULT_STEPS = 8
ANIMA_DEFAULT_GUIDANCE = 1.5
ANIMA_DEFAULT_SIZE = 512

ANIMA_SAMPLERS = ("Euler", "Euler Ancestral", "UniPC")
ANIMA_DEFAULT_SAMPLER = "Euler"
ANIMA_DEFAULT_SHIFT = 3.0
ANIMA_DEFAULT_SEED = -1
ANIMA_SEED_MAX = 2**32 - 1

ANIMA_TOKEN_LIMIT = 512

TOKEN_WARNING_FRACTION = 0.875

UNQUANTIZED_PREFIX = "unquantized:"

PREVIEW_DISPLAY_SIZE = 512
ANIMA_LATENT_RGB_FACTORS = [
    [-0.1299, -0.1692, 0.2932],
    [0.0671, 0.0406, 0.0442],
    [0.3568, 0.2548, 0.1747],
    [0.0372, 0.2344, 0.1420],
    [0.0313, 0.0189, -0.0328],
    [0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925],
    [0.0166, 0.1902, 0.1975],
    [-0.0412, 0.0267, -0.1364],
    [-0.1293, 0.0740, 0.1636],
    [0.0680, 0.3019, 0.1128],
    [0.0032, 0.0581, 0.0639],
    [-0.1251, 0.0927, 0.1699],
    [0.0060, -0.0633, 0.0005],
    [0.3477, 0.2275, 0.2950],
    [0.1984, 0.0913, 0.1861],
]
ANIMA_LATENT_RGB_BIAS = [-0.1835, -0.0868, -0.3360]

GGUF_QUANT_PREFERENCE = (
    "q4_k_m",
    "q4_k_s",
    "q4_k",
    "q4_0",
    "q5_k_m",
    "q5_k_s",
    "q5_k",
    "q5_0",
    "q6_k",
    "q8_0",
)
DEFAULT_DECODE_EVERY = 5

FLOW_SAMPLERS = ("Euler", "Euler Ancestral", "UniPC")
SIGMA_SCHEDULES = ("Default", "Beta", "Karras", "Exponential")
DEFAULT_SIGMAS = "Default"
SIGMA_CONFIG_KEYS = {
    "Beta": "use_beta_sigmas",
    "Karras": "use_karras_sigmas",
    "Exponential": "use_exponential_sigmas",
}
FLUX_LATENT_RGB_FACTORS = [
    [-0.0346, 0.0244, 0.0681],
    [0.0034, 0.0210, 0.0687],
    [0.0275, -0.0668, -0.0433],
    [-0.0174, 0.0160, 0.0617],
    [0.0859, 0.0721, 0.0329],
    [0.0004, 0.0383, 0.0115],
    [0.0405, 0.0861, 0.0915],
    [-0.0236, -0.0185, -0.0259],
    [-0.0245, 0.0250, 0.1180],
    [0.1008, 0.0755, -0.0421],
    [-0.0515, 0.0201, 0.0011],
    [0.0428, -0.0012, -0.0036],
    [0.0817, 0.0765, 0.0749],
    [-0.1264, -0.0522, -0.1103],
    [-0.0280, -0.0881, -0.0499],
    [-0.1262, -0.0982, -0.0778],
]
FLUX_LATENT_RGB_BIAS = [-0.0329, -0.0718, -0.0851]


GGUF_NOT_A_DIT = ("text_encoder", "gemma", "qwen", "clip", "t5", "llava", "vae")


QWEN_TEMPLATE_MIN_TOKENS = 5

ZIMAGE_COMPONENTS_REPO = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_GGUF_REPO = "leejet/Z-Image-Turbo-GGUF"
ZIMAGE_DEFAULT_DIT = ZIMAGE_GGUF_REPO
ZIMAGE_DEFAULT_STEPS = 8
ZIMAGE_DEFAULT_GUIDANCE = 0.0
ZIMAGE_DEFAULT_SIZE = 512
ZIMAGE_DEFAULT_SAMPLER = "Euler"
ZIMAGE_DEFAULT_SIGMAS = "Default"
ZIMAGE_DEFAULT_SHIFT = 3.0
ZIMAGE_DEFAULT_CFG_TRUNC = 1.0
ZIMAGE_MAX_SEQUENCE_LENGTH = 512
ZIMAGE_TOKEN_LIMIT = ZIMAGE_MAX_SEQUENCE_LENGTH
ZIMAGE_SIZE_MULTIPLE = 16
ZIMAGE_SAMPLERS = FLOW_SAMPLERS
ZIMAGE_TEXT_ENCODER_DTYPE = torch.bfloat16
ZIMAGE_CPU_DEVICE = "cpu"


def cast_floats(value, dtype):
    if torch.is_tensor(value):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, (list, tuple)):
        return type(value)(cast_floats(item, dtype) for item in value)
    return value


class _LoaderComplaints(stdlib_logging.Handler):
    def __init__(self):
        super().__init__(level=stdlib_logging.WARNING)
        self.messages = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception:  # noqa: BLE001, S110
            pass


@contextlib.contextmanager
def report_loader_complaints():
    handler = _LoaderComplaints()
    logger = stdlib_logging.getLogger("diffusers")
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)

    for message in handler.messages:
        lowered = message.lower()
        if any(
            word in lowered
            for word in ("unexpected", "missing", "not used", "mismatch")
        ):
            update_status(f"The loader reported: {message.strip()}")


def rank_gguf_variant(name, prefer, avoid):
    lowered = name.lower()
    if any(token in lowered for token in avoid):
        return 2
    if prefer and any(token in lowered for token in prefer):
        return 0
    return 1


def resolve_gguf_in_repo(repo, preference, prefer=(), avoid=()):
    try:
        from huggingface_hub import list_repo_files

        names = list_repo_files(repo)
    except Exception as e:
        raise ModelError(
            f"Could not read {repo} to pick a quantization ({e}). Give the "
            "DiT field a specific .gguf file or URL instead."
        ) from e

    every_gguf = [name for name in names if name.lower().endswith(".gguf")]
    files = sorted(
        name
        for name in every_gguf
        if not any(token in name.lower() for token in GGUF_NOT_A_DIT)
    )
    if files and (prefer or avoid):
        best = min(rank_gguf_variant(name, prefer, avoid) for name in files)
        files = [
            name for name in files if rank_gguf_variant(name, prefer, avoid) == best
        ]
    if not files:
        if every_gguf:
            raise ModelError(
                f"{repo} has GGUF files but none of them look like a "
                f"transformer: {', '.join(sorted(every_gguf)[:4])}. Give the "
                "DiT field the exact file you want."
            )
        return None

    for wanted in preference:
        for name in files:
            if wanted in name.lower():
                return f"https://huggingface.co/{repo}/{name}"

    try:
        from huggingface_hub import HfApi

        sizes = {
            sibling.rfilename: sibling.size
            for sibling in HfApi().model_info(repo, files_metadata=True).siblings
            if sibling.size
        }
        files.sort(key=lambda name: sizes.get(name, float("inf")))
    except Exception as e:  # noqa: BLE001
        print(f"Could not size the files in {repo}: {e}. Going by name.")

    return f"https://huggingface.co/{repo}/{files[0]}"


def _accepts_custom_sigmas(scheduler_cls):
    try:
        import inspect

        return "sigmas" in inspect.signature(scheduler_cls.set_timesteps).parameters
    except Exception:  # noqa: BLE001
        return True


def apply_sigma_schedule(config, sigmas):
    key = SIGMA_CONFIG_KEYS.get(sigmas)
    if key is None:
        return config
    for other in SIGMA_CONFIG_KEYS.values():
        config[other] = False
    if sigmas == "Beta":
        try:
            import scipy.stats  # noqa: F401
        except Exception:  # noqa: BLE001
            print(
                "Beta sigmas need scipy, which is not installed. Falling back "
                "to the schedule the checkpoint ships with."
            )
            return config
    config[key] = True
    return config


def flow_sigmas_usable(scheduler):
    try:
        scheduler.set_timesteps(8)
        sigmas = scheduler.sigmas
    except Exception:  # noqa: BLE001
        return True
    try:
        return bool(torch.isfinite(sigmas).all()) and float(sigmas.max()) <= 1.001
    except Exception:  # noqa: BLE001
        return True


def build_flow_scheduler(
    sampler, base_scheduler, shift=None, dynamic_shifting=None, sigmas=None
):
    config = dict(base_scheduler.config)
    config.pop("_use_default_values", None)
    if sigmas:
        config = apply_sigma_schedule(config, sigmas)

    if shift is None:
        shift = config.get("shift", 1.0)
    try:
        shift = float(shift)
    except (TypeError, ValueError):
        shift = 1.0
    if shift <= 0.0:
        shift = 1.0

    if dynamic_shifting is not None:
        config["use_dynamic_shifting"] = bool(dynamic_shifting)

    scheduler = _build_flow_scheduler(sampler, config, shift)
    if not sigmas or flow_sigmas_usable(scheduler):
        return scheduler

    update_status(
        f"{sigmas} sigmas come out on the wrong scale for {sampler} and "
        "would give a blank image, so this is running the sampler's own "
        "spacing instead."
    )
    plain = dict(config)
    for key in SIGMA_CONFIG_KEYS.values():
        plain[key] = False
    return _build_flow_scheduler(sampler, plain, shift)


def _build_flow_scheduler(sampler, config, shift):
    if sampler == "Euler Ancestral":
        return FlowMatchEulerDiscreteScheduler.from_config(
            config, shift=shift, stochastic_sampling=True
        )

    if sampler == "UniPC":
        spacing = {key: bool(config.get(key)) for key in SIGMA_CONFIG_KEYS.values()}
        return UniPCMultistepScheduler(
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            solver_order=2,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
            **spacing,
        )

    return FlowMatchEulerDiscreteScheduler.from_config(config, shift=shift)


def bind_scheduler_generator(scheduler, generator):
    if generator is None:
        return
    if not getattr(scheduler.config, "stochastic_sampling", False):
        return

    original_step = scheduler.step

    def seeded_step(*args, **kwargs):
        kwargs.setdefault("generator", generator)
        return original_step(*args, **kwargs)

    scheduler.step = seeded_step


_window = None
_anima_step_hook = None
_anima_stop_check = None

# isort: on


class ModelError(Exception):
    pass


def raise_exception_in_thread(thread_obj):
    if thread_obj is None or not thread_obj.is_alive():
        return False

    thread_id = None
    try:
        for tid, tobj in threading._active.items():
            if tobj is thread_obj:
                thread_id = tid
                break
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not access thread ID: {e}.")
        return False

    if thread_id is None:
        return False

    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(KeyboardInterrupt)
    )

    if res == 0:
        return False
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        return False

    return True


def is_direct_url(path_str):
    if not path_str or not isinstance(path_str, str):
        return False
    return path_str.startswith(("http://", "https://"))


def normalize_huggingface_url(url):
    if not url or not isinstance(url, str):
        return url

    if "huggingface.co" in url:
        url = url.replace("/blob/main/", "/")
        url = url.replace("/raw/main/", "/")
        url = url.replace("/resolve/main/", "/")

    return url


def _install_cosmos_torchvision_shim():
    if not ANIMA_AVAILABLE:
        return
    try:
        import torchvision  # noqa: F401

        return
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        import types

        from diffusers.models.transformers import transformer_cosmos

        if getattr(transformer_cosmos, "transforms", None) is not None:
            return

        class _InterpolationMode:
            NEAREST = "nearest"
            NEAREST_EXACT = "nearest-exact"
            BILINEAR = "bilinear"
            BICUBIC = "bicubic"

        def _resize(img, size, interpolation="nearest", *args, **kwargs):
            mode = interpolation if isinstance(interpolation, str) else "nearest"
            if isinstance(size, int):
                size = [size, size]
            squeeze = img.dim() == 3
            if squeeze:
                img = img.unsqueeze(0)
            out = torch.nn.functional.interpolate(img, size=list(size), mode=mode)
            return out.squeeze(0) if squeeze else out

        transformer_cosmos.transforms = types.SimpleNamespace(
            functional=types.SimpleNamespace(resize=_resize),
            InterpolationMode=_InterpolationMode,
        )
        print(
            "Using a resize shim for the Cosmos padding mask "
            "since torchvision isn't installed."
        )
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not install the Cosmos torchvision shim: {e}.")


def _set_anima_step_hook(fn):
    global _anima_step_hook
    _anima_step_hook = fn


def _set_anima_stop_check(fn):
    global _anima_stop_check
    _anima_stop_check = fn


def _install_anima_denoise_hook():
    if not ANIMA_AVAILABLE:
        return
    try:
        from diffusers.modular_pipelines.anima import denoise as anima_denoise

        wrapper = anima_denoise.AnimaDenoiseLoopWrapper
        if getattr(wrapper, "_animus_hooked", False):
            return
        orig_loop_step = wrapper.loop_step

        def _patched_loop_step(self, components, block_state, **kwargs):
            stop_check = _anima_stop_check
            if stop_check is not None and stop_check():
                raise KeyboardInterrupt()
            result = orig_loop_step(self, components, block_state, **kwargs)
            try:
                hook = _anima_step_hook
                if hook is not None:
                    bs = block_state
                    try:
                        _components, bs = result
                    except Exception:  # noqa: BLE001, S110
                        pass
                    hook(getattr(bs, "latents", None), kwargs.get("i"))
            except Exception:  # noqa: BLE001, S110
                pass
            return result

        wrapper.loop_step = _patched_loop_step
        wrapper._animus_hooked = True
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not install the preview hook: {e}.")


def is_huggingface_repo(path_str):
    if not path_str or not isinstance(path_str, str):
        return False

    path_obj = Path(path_str)

    if path_obj.exists():
        return False

    if len(path_str) >= 2 and path_str[1] == ":" and path_str[0].isalpha():
        return False

    parts = path_str.split("/")
    return (
        len(parts) == 2
        and not path_str.startswith((".", "/"))
        and not any(c in path_str for c in ["\\", "~"])
    )


WORKER_THREAD_TIMEOUT = 10.0
FFMPEG_TERM_TIMEOUT = 5.0

QUEUE_FRAMES = 3
QUEUE_BYTES = 256 * 1024 * 1024
PIPE_TARGET_BYTES = 1024 * 1024

REAL_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
LIVE_ACTION_SPAN = (
    "https://raw.githubusercontent.com/jcj83429/upscaling/"
    "5d8cdd2e17750b64be39ccab3a8763d91128fe15/2xLiveActionV1_SPAN"
)

MODEL_LICENSES = {
    "2xLiveActionV1_SPAN.pth": "Apache-2.0",
    "realesr-general-x4v3.pth": "BSD-3-Clause",
    "realesr-animevideov3.pth": "BSD-3-Clause",
    "RealESRGAN_x4plus_anime_6B.pth": "BSD-3-Clause",
    "RealESRGAN_x4plus.pth": "BSD-3-Clause",
}

PERMISSIVE_LICENSES = frozenset(("Apache-2.0", "BSD-3-Clause", "MIT", "CC0-1.0"))

BUILTIN_MODELS = (
    (
        "Live action video (SPAN)",
        "2xLiveActionV1_SPAN.pth",
        f"{LIVE_ACTION_SPAN}/2xLiveActionV1_SPAN_490000.pth",
    ),
    (
        "General video (slower)",
        "realesr-general-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
    ),
    (
        "Anime video (fast)",
        "realesr-animevideov3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
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

DEFAULT_MODEL = "2xLiveActionV1_SPAN.pth"
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

_FFMPEG_PROGRESS = re.compile(
    r"^(frame|fps|bitrate|total_size|out_time\w*|dup_frames|drop_frames|speed"
    r"|progress|stream_\d+_\d+_q)=(.*)$"
)


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


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain=2):
        super().__init__()
        self.sk = nn.Conv2d(c_in, c_out, 1, padding=0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, 1, padding=0, bias=True),
            nn.Conv2d(c_in * gain, c_out * gain, 3, padding=0, bias=True),
            nn.Conv2d(c_out * gain, c_out, 1, padding=0, bias=True),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=True)

    @torch.no_grad()
    def fuse(self):
        w1, b1 = self.conv[0].weight, self.conv[0].bias
        w2, b2 = self.conv[1].weight, self.conv[1].bias
        w3, b3 = self.conv[2].weight, self.conv[2].bias

        weight = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        bias = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        weight = (
            F.conv2d(weight.flip(2, 3).permute(1, 0, 2, 3), w3)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        bias = (w3 * bias.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        self.eval_conv.weight.copy_(weight + F.pad(self.sk.weight, [1, 1, 1, 1]))
        self.eval_conv.bias.copy_(bias + self.sk.bias)

        del self.sk, self.conv
        return self

    def forward(self, x):
        return self.eval_conv(x)


class SPAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1_r = Conv3XC(channels, channels)
        self.c2_r = Conv3XC(channels, channels)
        self.c3_r = Conv3XC(channels, channels)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = self.act1(self.c1_r(x))
        out2 = self.act1(self.c2_r(out1))
        out3 = self.c3_r(out2)
        attention = torch.sigmoid(out3) - 0.5
        return (out3 + x) * attention, out1


class SPAN(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=48,
        num_block=6,
        upscale=4,
        norm=False,
        img_range=255.0,
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super().__init__()
        self.num_out_ch = num_out_ch
        self.upscale = upscale
        self.num_block = num_block
        self.img_range = img_range
        self.register_buffer(
            "mean", torch.tensor(rgb_mean).view(1, 3, 1, 1), persistent=False
        )
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))

        self.conv_1 = Conv3XC(num_in_ch, feature_channels)
        for index in range(num_block):
            setattr(self, f"block_{index + 1}", SPAB(feature_channels))
        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, 1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels)
        self.upsampler = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                num_out_ch * upscale * upscale,
                3,
                padding=1,
                bias=True,
            ),
            nn.PixelShuffle(upscale),
        )

    @property
    def blocks(self):
        return [getattr(self, f"block_{i + 1}") for i in range(self.num_block)]

    @property
    def is_norm(self):
        return not hasattr(self, "no_norm")

    def fuse(self):
        for module in self.modules():
            if isinstance(module, Conv3XC):
                module.fuse()
        return self

    def forward(self, x):
        if self.is_norm:
            x = (x - self.mean.to(x.dtype)) * self.img_range

        feature = self.conv_1(x)
        flowing, first, inner = feature, None, None
        for index, block in enumerate(self.blocks):
            flowing, inner = block(flowing)
            if index == 0:
                first = flowing

        tail = self.conv_2(flowing)
        joined = self.conv_cat(torch.cat((feature, tail, first, inner), 1))
        return self.upsampler(joined)


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
        raise TypeError(f"{path.name} does not hold a state dict.")

    return {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}


def _body_indices(state_dict):
    indices = {}
    for key, tensor in state_dict.items():
        match = re.fullmatch(r"body\.(\d+)\.weight", key)
        if match and hasattr(tensor, "dim"):
            indices[int(match.group(1))] = tensor.dim()
    return indices


def build_span(state_dict):
    weight = state_dict["conv_1.sk.weight"]
    feature_channels = int(weight.shape[0])
    num_in_ch = int(weight.shape[1])

    blocks = set()
    for key in state_dict:
        match = re.match(r"block_(\d+)\.", key)
        if match:
            blocks.add(int(match.group(1)))
    if not blocks or sorted(blocks) != list(range(1, len(blocks) + 1)):
        raise ValueError("The SPAN checkpoint has no usable block numbering.")
    num_block = len(blocks)

    out_planes = int(state_dict["upsampler.0.weight"].shape[0])
    upscale = round(math.sqrt(out_planes / max(num_in_ch, 1)))
    if upscale < 1 or num_in_ch * upscale * upscale != out_planes:
        raise ValueError(
            f"Cannot derive the scale from a {out_planes}-channel upsampler."
        )

    model = SPAN(
        num_in_ch=num_in_ch,
        num_out_ch=num_in_ch,
        feature_channels=feature_channels,
        num_block=num_block,
        upscale=upscale,
        norm="no_norm" not in state_dict,
    )
    min_overlap = 3 * num_block + 3
    description = f"SPAN x{upscale} ({num_block} blocks, {feature_channels} features)"
    return model, upscale, 1, min_overlap, description


def build_upscaler(state_dict):
    if "conv_1.sk.weight" in state_dict:
        return build_span(state_dict)

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
            "Unrecognized checkpoint. This reads SPAN and the two "
            "Real-ESRGAN architectures - SRVGGNetCompact (realesr-*v3) and "
            "RRDBNet (RealESRGAN_x*plus, plain ESRGAN) and nothing else. "
            "Other designs such as OmniSR or DAT would each need their own "
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

    upscale = round(math.sqrt(out_planes / max(num_in_ch, 1)))
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
    if hasattr(model, "fuse"):
        model.fuse()
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    model = model.to(device=device, dtype=dtype)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    return model, scale, size_multiple, min_overlap, description


NCNN_MAGIC = 7767517
NCNN_OPTION_ENV = "ANIMUS_NCNN_OPTIONS"

_TRUTHY = ("", "1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def ncnn_option_overrides():
    raw = os.environ.get(NCNN_OPTION_ENV, "").strip()
    if not raw:
        return {}

    overrides = {}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name, value = name.strip(), value.strip().lower()
        if not (name.startswith("use_") or name in ("lightmode", "num_threads")):
            print(f"Ignoring {NCNN_OPTION_ENV} entry '{item}': not an option.")
            continue
        if value in _TRUTHY:
            overrides[name] = True
        elif value in _FALSY:
            overrides[name] = False
        else:
            try:
                overrides[name] = int(value)
            except ValueError:
                print(f"Ignoring {NCNN_OPTION_ENV} entry '{item}': not a value.")
    return overrides


BINARY_ADD, BINARY_SUB, BINARY_MUL = 0, 1, 2


class NcnnGraph:
    def __init__(self):
        self.layers = []

    def add(self, kind, name, inputs, outputs, params=(), weights=()):
        self.layers.append(
            {
                "kind": kind,
                "name": name,
                "inputs": list(inputs),
                "outputs": list(outputs),
                "params": list(params),
                "weights": list(weights),
            }
        )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def input(self, name="data"):
        return self.add("Input", name, [], [name])

    def conv(self, name, source, target, conv, weight, bias):
        return self.add(
            "Convolution",
            name,
            [source],
            [target],
            [
                f"0={conv.out_channels}",
                f"1={conv.kernel_size[1]}",
                f"11={conv.kernel_size[0]}",
                f"3={conv.stride[1]}",
                f"13={conv.stride[0]}",
                f"4={conv.padding[1]}",
                f"14={conv.padding[0]}",
                "5=1",
                f"6={weight.numel()}",
            ],
            [(True, weight), (False, bias)],
        )

    def unary(self, kind, name, source, target, params=(), weights=()):
        return self.add(kind, name, [source], [target], params, weights)

    def binary(self, name, op, left, right, target):
        return self.add("BinaryOp", name, [left, right], [target], [f"0={op}"])

    def scalar(self, name, op, source, target, value):
        return self.add(
            "BinaryOp", name, [source], [target], [f"0={op}", "1=1", f"2={value}"]
        )

    def _fork(self):
        demand = {}
        for layer in self.layers:
            for blob in layer["inputs"]:
                demand[blob] = demand.get(blob, 0) + 1

        supply, resolved = {}, []
        for layer in self.layers:
            layer["inputs"] = [
                supply[blob].pop(0) if blob in supply else blob
                for blob in layer["inputs"]
            ]
            resolved.append(layer)
            for blob in layer["outputs"]:
                count = demand.get(blob, 0)
                if count < 2:
                    continue
                copies = [f"{blob}_s{index}" for index in range(count)]
                supply[blob] = list(copies)
                resolved.append(
                    {
                        "kind": "Split",
                        "name": f"fork_{blob}",
                        "inputs": [blob],
                        "outputs": copies,
                        "params": [],
                        "weights": [],
                    }
                )
        self.layers = resolved

    def write(self, param_path, bin_path):
        self._fork()

        blobs = []
        for layer in self.layers:
            for blob in layer["outputs"]:
                if blob not in blobs:
                    blobs.append(blob)

        lines = []
        for layer in self.layers:
            fields = [
                f"{layer['kind']:<16}",
                f"{layer['name']:<9}",
                str(len(layer["inputs"])),
                str(len(layer["outputs"])),
                *layer["inputs"],
                *layer["outputs"],
                *layer["params"],
            ]
            lines.append(" ".join(fields).rstrip())

        Path(param_path).parent.mkdir(parents=True, exist_ok=True)
        with open(param_path, "w") as handle:
            handle.write(f"{NCNN_MAGIC}\n{len(lines)} {len(blobs)}\n")
            handle.write("\n".join(lines) + "\n")

        with open(bin_path, "wb") as handle:
            for layer in self.layers:
                for flagged, tensor in layer["weights"]:
                    if flagged:
                        handle.write(struct.pack("<I", 0))
                    handle.write(
                        tensor.detach().to(torch.float32).contiguous().numpy().tobytes()
                    )


def _write_compact_ncnn(graph, model, state_dict, scale):
    convolutions, activations = [], []
    for index, layer in enumerate(model.body):
        if isinstance(layer, nn.Conv2d):
            convolutions.append((index, layer))
        else:
            activations.append((index, layer))

    source = graph.input()
    previous = source

    for order, (index, conv) in enumerate(convolutions):
        previous = graph.conv(
            f"conv{order}",
            previous,
            f"c{order}",
            conv,
            state_dict[f"body.{index}.weight"],
            state_dict[f"body.{index}.bias"],
        )

        if order >= len(activations):
            continue
        act_index, activation = activations[order]
        if isinstance(activation, nn.PReLU):
            slope = state_dict[f"body.{act_index}.weight"]
            previous = graph.unary(
                "PReLU",
                f"act{order}",
                previous,
                f"a{order}",
                [f"0={slope.numel()}"],
                [(False, slope)],
            )
        else:
            negative = float(getattr(activation, "negative_slope", 0.0))
            previous = graph.unary(
                "ReLU", f"act{order}", previous, f"a{order}", [f"0={negative}"]
            )

    shuffled = graph.unary(
        "PixelShuffle", "shuffle", previous, "shuffled", [f"0={scale}", "1=0"]
    )
    nearest = graph.unary(
        "Interp",
        "nearest",
        source,
        "nearest",
        ["0=1", f"1={float(scale)}", f"2={float(scale)}"],
    )
    graph.binary("add", BINARY_ADD, shuffled, nearest, "out")


def _write_span_ncnn(graph, model, scale):
    source = graph.input()

    if model.is_norm:
        mean = model.mean.reshape(-1)
        channels = int(mean.numel())
        rate = float(model.img_range)
        weight = torch.zeros(channels, channels, 1, 1)
        for index in range(channels):
            weight[index, index, 0, 0] = rate
        source = graph.add(
            "Convolution",
            "norm",
            [source],
            ["normed"],
            [
                f"0={channels}",
                "1=1",
                "11=1",
                "3=1",
                "13=1",
                "4=0",
                "14=0",
                "5=1",
                f"6={weight.numel()}",
            ],
            [(True, weight), (False, -mean * rate)],
        )

    def emit(name, module, source, target):
        return graph.conv(name, source, target, module, module.weight, module.bias)

    feature = emit("conv_1", model.conv_1.eval_conv, source, "feature")

    flowing, first, inner = feature, None, None
    for index, block in enumerate(model.blocks):
        tag = f"b{index}"
        gated = emit(f"{tag}c1", block.c1_r.eval_conv, flowing, f"{tag}_c1")
        inner = graph.unary("Swish", f"{tag}a1", gated, f"{tag}_i")
        mid = emit(f"{tag}c2", block.c2_r.eval_conv, inner, f"{tag}_c2")
        act2 = graph.unary("Swish", f"{tag}a2", mid, f"{tag}_a2")
        out3 = emit(f"{tag}c3", block.c3_r.eval_conv, act2, f"{tag}_c3")

        sigmoid = graph.unary("Sigmoid", f"{tag}sig", out3, f"{tag}_g")
        attention = graph.scalar(f"{tag}att", BINARY_SUB, sigmoid, f"{tag}_t", 0.5)
        summed = graph.binary(f"{tag}add", BINARY_ADD, out3, flowing, f"{tag}_s")
        flowing = graph.binary(f"{tag}mul", BINARY_MUL, summed, attention, f"{tag}_r")
        if index == 0:
            first = flowing

    tail = emit("conv_2", model.conv_2.eval_conv, flowing, "tail")
    graph.add("Concat", "cat", [feature, tail, first, inner], ["cat"], ["0=0"])
    graph.conv(
        "conv_cat",
        "cat",
        "joined",
        model.conv_cat,
        model.conv_cat.weight,
        model.conv_cat.bias,
    )
    emit("up", model.upsampler[0], "joined", "shuffled")
    graph.unary("PixelShuffle", "shuffle", "shuffled", "out", [f"0={scale}", "1=0"])


def write_ncnn_model(state_dict, param_path, bin_path):
    model, scale, size_multiple, min_overlap, description = build_upscaler(state_dict)
    if not isinstance(model, (SRVGGNetCompact, SPAN)):
        raise TypeError(
            f"{description} is not a compact generator. Only those are "
            "converted, because the heavy ones are not worth running on video."
        )
    model.load_state_dict(state_dict, strict=True)

    graph = NcnnGraph()
    if isinstance(model, SPAN):
        model.fuse()
        _write_span_ncnn(graph, model, scale)
    else:
        _write_compact_ncnn(graph, model, state_dict, scale)
    graph.write(param_path, bin_path)

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

        for name, value in ncnn_option_overrides().items():
            if not hasattr(self.net.opt, name):
                print(f"This ncnn has no option '{name}', so it was skipped.")
                continue
            try:
                setattr(self.net.opt, name, value)
                print(f"ncnn option {name} = {value}.")
            except Exception as e:  # noqa: BLE001
                print(f"Could not set ncnn option {name} ({e}).")

        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

        self.on_gpu = bool(self.net.opt.use_vulkan_compute)

        self._extract_into = True

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
            except Exception as e:  # noqa: BLE001
                print(
                    f"Could not hold on to the GPU allocators ({e}). ncnn will "
                    "take and return them every frame instead."
                )

    def __call__(self, frame):
        planes = frame[0].detach()
        if planes.dtype is not torch.float32:
            planes = planes.to(torch.float32)
        planes = planes.contiguous()
        source = planes.numpy()

        extractor = self.net.create_extractor()
        extractor.input("data", self._ncnn.Mat(source))

        if self._extract_into:
            result = self._ncnn.Mat()
            try:
                status = extractor.extract("out", result)
            except TypeError as e:
                print(
                    f"This ncnn has no extract(blob, mat) ({e}), so every "
                    "frame will be copied an extra time on the way out."
                )
                self._extract_into = False
                status, result = extractor.extract("out")
        else:
            status, result = extractor.extract("out")

        if status != 0:
            raise RuntimeError(f"ncnn returned {status} from the network.")

        return torch.from_numpy(numpy.asarray(result)).unsqueeze(0)

    def close(self):
        try:
            self.net.clear()
        except Exception:  # noqa: BLE001, S110
            pass
        for allocator in getattr(self, "_allocators", []):
            try:
                allocator.clear()
            except Exception:  # noqa: BLE001, S110
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
            "CPU backend. The device list is built from what Vulkan enumerates, "
            "which is not the same as what ncnn can open."
        )

    where = f"Vulkan device {gpu}" if model.on_gpu else "CPU"
    precision = "fp16" if (fp16 and model.on_gpu) else "fp32"
    return (
        model,
        scale,
        size_multiple,
        min_overlap,
        f"{description} via ncnn ({where}, {precision})",
    )


def ncnn_devices():
    try:
        import ncnn
    except Exception:  # noqa: BLE001
        return []

    try:
        count = ncnn.get_gpu_count()
    except Exception:  # noqa: BLE001
        return []

    kinds = {0: "discrete", 1: "integrated", 2: "virtual", 3: "software"}
    devices = []
    for index in range(count):
        name, kind = f"device {index}", ""
        try:
            info = ncnn.get_gpu_info(index)
            name = info.device_name()
            kind = kinds.get(info.type(), "")
        except Exception:  # noqa: BLE001, S110
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
        completed = subprocess.run(command, capture_output=True, check=False)
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
        frames = round(duration * fps)

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


def grow_pipe(stream, wanted=PIPE_TARGET_BYTES):
    try:
        import fcntl
    except ImportError:
        return 0

    setter = getattr(fcntl, "F_SETPIPE_SZ", 1031)
    try:
        descriptor = stream.fileno()
    except Exception:  # noqa: BLE001
        return 0

    ceiling = wanted
    try:
        with open("/proc/sys/fs/pipe-max-size") as handle:
            ceiling = min(wanted, int(handle.read().strip()))
    except Exception:  # noqa: BLE001, S110
        pass

    size = max(ceiling, 65536)
    while size >= 65536:
        try:
            return int(fcntl.fcntl(descriptor, setter, size))
        except (OSError, ValueError):
            size //= 2
    return 0


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


STAGE_NAMES = ("decode", "prepare", "network", "convert", "encode")


def new_profile():
    return {name: 0.0 for name in STAGE_NAMES}


def describe_profile(profile, frames):
    if not frames:
        return ""
    parts = [f"{name} {profile[name] / frames * 1e3:.0f} ms" for name in STAGE_NAMES]
    total = sum(profile.values()) / frames * 1e3
    return f"Per frame: {' | '.join(parts)}  (measured total {total:.0f} ms)."


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
    profile=None,
):
    device = torch.device(device)
    frame_bytes = width * height * 3
    memory_format = torch.channels_last if channels_last else torch.contiguous_format

    if profile is None:
        profile = new_profile()

    def depth(each):
        return max(1, min(QUEUE_FRAMES, QUEUE_BYTES // max(each, 1)))

    reads = queue.Queue(maxsize=depth(frame_bytes))
    writes = queue.Queue(maxsize=depth(frame_bytes * scale * scale))
    stopping = threading.Event()
    failures = []
    delivered = [0]

    def offer(destination, item):
        while not stopping.is_set():
            try:
                destination.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def pump():
        try:
            while not stopping.is_set() and (stop_check is None or not stop_check()):
                buffer = bytearray(frame_bytes)
                got = read_exact(source, buffer)
                if got == 0:
                    break
                if got < frame_bytes:
                    print(
                        f"Warning: the decoder stopped {frame_bytes - got} bytes "
                        "into a frame. Treating that as the end of the stream."
                    )
                    break
                if not offer(reads, buffer):
                    return
        except Exception as e:  # noqa: BLE001
            failures.append(e)
        offer(reads, None)

    def drain():
        while True:
            item = writes.get()
            if item is None:
                return
            if failures:
                continue
            try:
                write_exact(sink, item)
                delivered[0] += 1
            except Exception as e:  # noqa: BLE001
                failures.append(e)

    reader = threading.Thread(target=pump, name="animus-decode", daemon=True)
    writer = threading.Thread(target=drain, name="animus-encode", daemon=True)
    reader.start()
    writer.start()

    clock = time.perf_counter
    written = 0

    try:
        while stop_check is None or not stop_check():
            mark = clock()
            buffer = reads.get()
            profile["decode"] += clock() - mark
            if buffer is None or failures:
                break

            mark = clock()
            frame = (
                torch.frombuffer(buffer, dtype=torch.uint8)
                .reshape(height, width, 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous(memory_format=memory_format)
                .to(device=device, dtype=dtype)
                .div_(255.0)
            )
            profile["prepare"] += clock() - mark

            with torch.inference_mode():
                mark = clock()
                result = upscale_frame(
                    model,
                    frame,
                    scale,
                    tile,
                    tile_pad,
                    size_multiple=size_multiple,
                    stop_check=stop_check,
                )
                profile["network"] += clock() - mark
                if result is None:
                    break

                mark = clock()
                planes = result[0].mul_(255.0).round_().clamp_(0.0, 255.0)
                rgb = torch.empty(
                    (planes.shape[1], planes.shape[2], 3), dtype=torch.uint8
                )
                rgb.copy_(planes.permute(1, 2, 0))

                if on_frame is not None:
                    on_frame(written + 1, rgb)
                profile["convert"] += clock() - mark

            del frame, result, planes, buffer

            mark = clock()
            handed = offer(writes, rgb.numpy())
            profile["encode"] += clock() - mark
            del rgb
            if not handed:
                break
            written += 1
    finally:
        while writer.is_alive():
            try:
                writes.put(None, timeout=1.0)
                break
            except queue.Full:
                continue

        writer.join()

        stopping.set()
        reader.join(timeout=0.25)

    if failures:
        raise failures[0]

    return delivered[0]


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
        except Exception:  # noqa: BLE001, S110
            pass
    except Exception:  # noqa: BLE001, S110
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
BF16_IN_HARDWARE = bool(CPU_FLAGS & {"amx_bf16", "avx512_bf16"})


def available_devices():
    devices = ["cpu"]

    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                devices.append(f"cuda:{index}")
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for index in range(xpu.device_count()):
                devices.append(f"xpu:{index}")
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:  # noqa: BLE001, S110
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
    except Exception:  # noqa: BLE001, S110
        pass
    return name


def describe_torch_build():
    parts = [f"torch {torch.__version__}"]
    try:
        parts.append(
            "oneDNN " + ("on" if torch.backends.mkldnn.is_available() else "off")
        )
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        capability = torch.backends.cpu.get_cpu_capability()
        if BF16_IN_HARDWARE:
            capability += "+bf16"
        parts.append(capability)
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        parts.append(
            "OpenMP " + ("on" if torch.backends.openmp.is_available() else "off")
        )
    except Exception:  # noqa: BLE001, S110
        pass
    parts.append(f"{torch.get_num_threads()} threads")
    return ", ".join(parts)


Z_SEQ_MULTIPLE = animus_vulkan.Z_SEQ_MULTIPLE
Z_ADALN_EMBED_DIM = animus_vulkan.Z_ADALN_EMBED_DIM
Z_FREQUENCY_EMBEDDING_SIZE = animus_vulkan.Z_FREQUENCY_EMBEDDING_SIZE
Z_MAX_PERIOD = animus_vulkan.Z_MAX_PERIOD
Z_ROPE_THETA = animus_vulkan.Z_ROPE_THETA
Z_ROPE_AXES_DIMS = animus_vulkan.Z_ROPE_AXES_DIMS
Z_ROPE_AXES_LENS = animus_vulkan.Z_ROPE_AXES_LENS


class ZRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class ZTimestepEmbedder(nn.Module):
    def __init__(
        self,
        out_size,
        mid_size=1024,
        frequency_embedding_size=Z_FREQUENCY_EMBEDDING_SIZE,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=Z_MAX_PERIOD):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight = self.mlp[0].weight
        dtype = weight.dtype if weight.is_floating_point() else torch.float32
        return self.mlp(t_freq.to(dtype))


class ZFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_emb(x, cos, sin):
    x = x.float().unflatten(-1, (-1, 2))
    x_even = x[..., 0]
    x_odd = x[..., 1]
    out = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return out.flatten(-2)


class ZAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, qk_norm=True, eps=1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        if n_heads % n_kv_heads:
            raise ModelError(
                f"{n_heads} query heads do not divide into {n_kv_heads} key "
                "heads, so they cannot be shared between them."
            )

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList(
            [nn.Linear(n_heads * self.head_dim, dim, bias=False)]
        )

        self.norm_q = ZRMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = ZRMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(self, x, cos, sin):
        query = self.to_q(x).unflatten(-1, (self.n_heads, -1))
        key = self.to_k(x).unflatten(-1, (self.n_kv_heads, -1))
        value = self.to_v(x).unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        dtype = value.dtype
        query = apply_rotary_emb(query, cos, sin).to(dtype)
        key = apply_rotary_emb(key, cos, sin).to(dtype)

        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
        )
        out = out.squeeze(0).transpose(0, 1).flatten(1)
        return self.to_out[0](out.to(dtype))


class ZTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads,
        norm_eps,
        qk_norm,
        modulation=True,
        ffn_hidden=None,
    ):
        super().__init__()
        self.modulation = modulation
        self.attention = ZAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = ZFeedForward(
            dim=dim, hidden_dim=ffn_hidden or int(dim / 3 * 8)
        )
        self.attention_norm1 = ZRMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = ZRMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = ZRMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = ZRMSNorm(dim, eps=norm_eps)
        if modulation:
            self.adaLN_modulation = nn.ModuleList(
                [nn.Linear(min(dim, Z_ADALN_EMBED_DIM), 4 * dim, bias=True)]
            )

    def forward(self, x, cos, sin, adaln_input=None):
        if not self.modulation:
            x = x + self.attention_norm2(
                self.attention(self.attention_norm1(x), cos, sin)
            )
            return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation[0](
            adaln_input
        ).chunk(4, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        attn_out = self.attention(self.attention_norm1(x) * scale_msa, cos, sin)
        x = x + gate_msa * self.attention_norm2(attn_out)
        return x + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        )


class ZFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_features):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_features, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, Z_ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        return self.linear(self.norm_final(x) * scale)


class ZRopeEmbedder:
    def __init__(
        self, theta=Z_ROPE_THETA, axes_dims=Z_ROPE_AXES_DIMS, axes_lens=Z_ROPE_AXES_LENS
    ):
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = tuple(axes_lens)
        self.cos = None
        self.sin = None
        self.theta = theta

    def _build(self, device):
        cos, sin = [], []
        for dim, length in zip(self.axes_dims, self.axes_lens):
            freqs = 1.0 / (
                self.theta
                ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)
            )
            positions = torch.arange(length, dtype=torch.float64, device=device)
            angles = torch.outer(positions, freqs).float()
            cos.append(torch.cos(angles))
            sin.append(torch.sin(angles))
        self.cos, self.sin = cos, sin

    def __call__(self, ids):
        if self.cos is None or self.cos[0].device != ids.device:
            self._build(ids.device)
        cos = torch.cat(
            [self.cos[i][ids[:, i]] for i in range(len(self.axes_dims))], -1
        )
        sin = torch.cat(
            [self.sin[i][ids[:, i]] for i in range(len(self.axes_dims))], -1
        )
        return cos.unsqueeze(1), sin.unsqueeze(1)


def _coordinate_grid(size, start, device):
    axes = [
        torch.arange(x0, x0 + span, dtype=torch.long, device=device)
        for x0, span in zip(start, size)
    ]
    return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1).flatten(0, 2)


class ZImageTransformer(nn.Module):
    def __init__(
        self,
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        patch_size=2,
        f_patch_size=1,
        ffn_hidden=None,
        t_embedder_mid=1024,
        rope_theta=Z_ROPE_THETA,
        t_scale=1000.0,
        axes_dims=Z_ROPE_AXES_DIMS,
        axes_lens=Z_ROPE_AXES_LENS,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.t_scale = t_scale
        self.key = f"{patch_size}-{f_patch_size}"
        if ffn_hidden is None:
            ffn_hidden = int(dim / 3 * 8)

        patch_features = f_patch_size * patch_size * patch_size * in_channels
        self.all_x_embedder = nn.ModuleDict(
            {self.key: nn.Linear(patch_features, dim, bias=True)}
        )
        self.all_final_layer = nn.ModuleDict(
            {self.key: ZFinalLayer(dim, patch_features)}
        )

        def block(modulation):
            return ZTransformerBlock(
                dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation, ffn_hidden
            )

        self.noise_refiner = nn.ModuleList(
            [block(True) for _ in range(n_refiner_layers)]
        )
        self.context_refiner = nn.ModuleList(
            [block(False) for _ in range(n_refiner_layers)]
        )
        self.layers = nn.ModuleList([block(True) for _ in range(n_layers)])

        self.t_embedder = ZTimestepEmbedder(
            min(dim, Z_ADALN_EMBED_DIM), mid_size=t_embedder_mid
        )
        self.cap_embedder = nn.Sequential(
            ZRMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.zeros((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.zeros((1, dim)))

        if dim // n_heads != sum(axes_dims):
            raise ModelError(
                f"A head is {dim // n_heads} wide but the rotary axes come "
                f"to {sum(axes_dims)}. They have to be the same number: the "
                "rotation fills the head exactly."
            )
        self.rope_embedder = ZRopeEmbedder(rope_theta, axes_dims, axes_lens)

    @property
    def dtype(self):
        for parameter in self.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    @property
    def device(self):
        return next(self.parameters()).device

    def _patchify(self, image):
        pH = pW = self.patch_size
        pF = self.f_patch_size
        channels, frames, height, width = image.size()
        f_tokens, h_tokens, w_tokens = frames // pF, height // pH, width // pW
        image = image.view(channels, f_tokens, pF, h_tokens, pH, w_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0)
        return (
            image.reshape(f_tokens * h_tokens * w_tokens, pF * pH * pW * channels),
            (f_tokens, h_tokens, w_tokens),
        )

    def _unpatchify(self, tokens, grid):
        pH = pW = self.patch_size
        pF = self.f_patch_size
        f_tokens, h_tokens, w_tokens = grid
        return (
            tokens[: f_tokens * h_tokens * w_tokens]
            .view(f_tokens, h_tokens, w_tokens, pF, pH, pW, self.out_channels)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(self.out_channels, f_tokens * pF, h_tokens * pH, w_tokens * pW)
        )

    @staticmethod
    def _pad_to_multiple(features, multiple):
        extra = (-len(features)) % multiple
        if not extra:
            return features, 0
        return torch.cat([features, features[-1:].repeat(extra, 1)], dim=0), extra

    def forward(self, x, timestep, cap_feats):
        """x: [C, F, H, W]; timestep: scalar tensor; cap_feats: [cap_len, D]."""
        device = x.device
        adaln_input = self.t_embedder((timestep.reshape(1) * self.t_scale).to(device))[
            0
        ]

        cap_len = len(cap_feats)
        cap_feats, cap_pad = self._pad_to_multiple(cap_feats, Z_SEQ_MULTIPLE)
        cap_ids = _coordinate_grid((cap_len + cap_pad, 1, 1), (1, 0, 0), device)

        image, grid = self._patchify(x)
        image_len = len(image)
        image, image_pad = self._pad_to_multiple(image, Z_SEQ_MULTIPLE)
        image_ids = _coordinate_grid(grid, (cap_len + cap_pad + 1, 0, 0), device)
        if image_pad:
            image_ids = torch.cat(
                [image_ids, image_ids.new_zeros((image_pad, 3))], dim=0
            )

        image = self.all_x_embedder[self.key](image)
        if image_pad:
            image[image_len:] = self.x_pad_token.to(image.dtype)
        image_cos, image_sin = self.rope_embedder(image_ids)
        for layer in self.noise_refiner:
            image = layer(image, image_cos, image_sin, adaln_input)

        cap_feats = self.cap_embedder(cap_feats)
        if cap_pad:
            cap_feats[cap_len:] = self.cap_pad_token.to(cap_feats.dtype)
        cap_cos, cap_sin = self.rope_embedder(cap_ids)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_cos, cap_sin)

        unified = torch.cat([image, cap_feats], dim=0)
        cos = torch.cat([image_cos, cap_cos], dim=0)
        sin = torch.cat([image_sin, cap_sin], dim=0)
        for layer in self.layers:
            unified = layer(unified, cos, sin, adaln_input)

        unified = self.all_final_layer[self.key](unified, adaln_input)
        return self._unpatchify(unified, grid)


normalize_z_image_key = animus_vulkan.normalize_key


class ZQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, device=None):
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = torch.float32
        self.loras = {}

    def add_lora(self, adapter, down, up, scale=1.0, factor=1.0):
        self.loras[adapter] = (
            down.to(self.compute_dtype),
            up.to(self.compute_dtype),
            float(scale) * float(factor),
            float(factor),
        )

    def set_lora_scale(self, adapter, scale):
        if adapter in self.loras:
            down, up, _, factor = self.loras[adapter]
            self.loras[adapter] = (down, up, float(scale) * factor, factor)

    def clear_loras(self):
        self.loras = {}

    def _weight(self):
        weight = self.weight
        quant_type = getattr(weight, "quant_type", None)
        if quant_type is None:
            return weight.to(self.compute_dtype)
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

        return dequantize_gguf_tensor(weight).to(self.compute_dtype)

    def forward(self, inputs):
        inputs = inputs.to(self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        output = F.linear(inputs, self._weight(), bias)
        for down, up, scale, _factor in self.loras.values():
            if scale:
                output = output + scale * F.linear(F.linear(inputs, down), up)
        return output


def promote_to_quant_linear(model, path):
    module = model.get_submodule(path)
    if isinstance(module, ZQuantLinear):
        return module
    if not isinstance(module, nn.Linear):
        return None
    replacement = ZQuantLinear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device="meta",
    )
    replacement._parameters["weight"] = module.weight
    if module.bias is not None:
        replacement._parameters["bias"] = module.bias
    owner_path, _, attribute = path.rpartition(".")
    owner = model.get_submodule(owner_path) if owner_path else model
    setattr(owner, attribute, replacement)
    return replacement


def _quantized(tensor):
    return getattr(tensor, "quant_type", None) is not None


def _fits_expected(shape, expected):
    if shape == expected:
        return True
    stripped = tuple(d for d in shape if d != 1)
    return stripped == tuple(d for d in expected if d != 1)


def _dequantize(tensor):
    if not _quantized(tensor):
        return tensor.to(torch.float32) if tensor.is_floating_point() else tensor
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    return dequantize_gguf_tensor(tensor).to(torch.float32)


def _logical_shape(tensor):
    return tuple(getattr(tensor, "quant_shape", None) or tensor.shape)


def split_fused_attention(state):
    fused = [
        name
        for name in state
        if name.endswith((".attention.qkv.weight", ".attention.qkv.bias"))
    ]
    for name in fused:
        prefix, _, suffix = name.rpartition(".")
        prefix = prefix[: -len(".attention.qkv")]
        tensor = state.pop(name)
        rows = int(_logical_shape(tensor)[0])

        norm = state.get(f"{prefix}.attention.norm_q.weight")
        shape = _logical_shape(tensor)
        head_dim = int(_logical_shape(norm)[0]) if norm is not None else 0
        hidden = int(shape[1]) if len(shape) > 1 else 0
        queries = keys = rows // 3
        if head_dim and hidden and hidden % head_dim == 0:
            queries = hidden
            shared = rows - queries
            if shared > 0 and shared % (2 * head_dim) == 0:
                keys = shared // 2
            else:
                queries = keys = rows // 3

        state[f"{prefix}.attention.to_q.{suffix}"] = tensor[:queries]
        state[f"{prefix}.attention.to_k.{suffix}"] = tensor[queries : queries + keys]
        state[f"{prefix}.attention.to_v.{suffix}"] = tensor[queries + keys :]
    return state


def read_z_image_state_dict(path):
    path = str(path)
    if path.lower().endswith(".gguf"):
        from diffusers.models.model_loading_utils import load_gguf_checkpoint

        raw = load_gguf_checkpoint(path)
    else:
        from safetensors.torch import load_file

        raw = load_file(path)

    state = {}
    for name, tensor in raw.items():
        state[normalize_z_image_key(name)] = tensor
    return split_fused_attention(state)


def detect_z_image_config(state):
    def shape_of(key):
        tensor = state.get(key)
        return None if tensor is None else _logical_shape(tensor)

    try:
        config = animus_vulkan.detect_config(state, shape_of)
    except animus_vulkan.VulkanError as e:
        raise ModelError(str(e)) from e

    head_dim = config.pop("head_dim")
    if head_dim != sum(Z_ROPE_AXES_DIMS):
        raise ModelError(
            f"This checkpoint has a head dimension of {head_dim}, but "
            f"Z-Image's rotary embedding is built for "
            f"{sum(Z_ROPE_AXES_DIMS)}. This is not a Z-Image DiT."
        )
    return config


def build_z_image_transformer(state, config=None, progress=None):
    config = config or detect_z_image_config(state)
    with torch.device("meta"):
        model = ZImageTransformer(**config)

    packed = set()
    for name, tensor in state.items():
        if not name.endswith(".weight"):
            continue
        parent = name[: -len(".weight")]
        try:
            module = model.get_submodule(parent)
        except AttributeError:
            continue
        if not isinstance(module, nn.Linear) or isinstance(module, ZQuantLinear):
            continue
        if not (_quantized(tensor) or tensor.dtype != torch.float32):
            continue
        replacement = ZQuantLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
        )
        owner_name, _, attribute = parent.rpartition(".")
        setattr(
            model.get_submodule(owner_name) if owner_name else model,
            attribute,
            replacement,
        )
        packed.add(name)

    expected = {name: tuple(p.shape) for name, p in model.state_dict().items()}
    missing = set(expected)
    unexpected = []

    for name, tensor in state.items():
        if name not in expected:
            unexpected.append(name)
            continue
        shape = tuple(getattr(tensor, "quant_shape", None) or tensor.shape)
        wanted = expected[name]
        if not _fits_expected(shape, wanted):
            raise ModelError(
                f"{name} is {shape} in this checkpoint but the DiT wants "
                f"{wanted}. This file is not the model it claims."
            )
        if name not in packed:
            tensor = _dequantize(tensor)
            if tuple(tensor.shape) != wanted:
                tensor = tensor.reshape(wanted)
        if not isinstance(tensor, nn.Parameter):
            tensor = nn.Parameter(tensor, requires_grad=False)

        owner_path, _, attribute = name.rpartition(".")
        owner = model.get_submodule(owner_path) if owner_path else model
        owner._parameters[attribute] = tensor
        missing.discard(name)

    if missing:
        raise ModelError(
            f"This checkpoint is missing {len(missing)} of the DiT's tensors, "
            f"starting with {min(missing)}."
        )
    if unexpected and progress is not None:
        progress(
            f"Ignored {len(unexpected)} tensors this DiT has no use for, "
            f"starting with {unexpected[0]}."
        )
    model.eval()
    model.requires_grad_(False)
    return model


Z_BASE_IMAGE_SEQ_LEN = 256
Z_MAX_IMAGE_SEQ_LEN = 4096
Z_BASE_SHIFT = 0.5
Z_MAX_SHIFT = 1.15


def z_image_shift(
    image_seq_len,
    base_seq_len=Z_BASE_IMAGE_SEQ_LEN,
    max_seq_len=Z_MAX_IMAGE_SEQ_LEN,
    base_shift=Z_BASE_SHIFT,
    max_shift=Z_MAX_SHIFT,
):
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


class ZImageOutput:
    def __init__(self, images):
        self.images = images


class NativeZImagePipeline:
    def __init__(
        self,
        transformer=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        from diffusers.image_processor import VaeImageProcessor

        factor = 8
        if vae is not None and hasattr(vae, "config"):
            factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.vae_scale_factor = factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=factor)

    def to(self, *args, **kwargs):
        for component in (self.transformer, self.vae, self.text_encoder):
            if component is not None and hasattr(component, "to"):
                component.to(*args, **kwargs)
        return self

    @property
    def components(self):
        return {
            "transformer": self.transformer,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "scheduler": self.scheduler,
        }

    def _format(self, text):
        messages = [{"role": "user", "content": text or ""}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _encode_one(self, text, max_sequence_length):
        inputs = self.tokenizer(
            [self._format(text)],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        mask = inputs.attention_mask.bool()
        outputs = self.text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-2]
        return hidden[0][mask[0]].to(torch.float32)

    def encode_prompt(
        self,
        prompt,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        max_sequence_length=512,
        **kwargs,
    ):
        if self.text_encoder is None:
            raise ModelError("The text encoder is not loaded.")
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ""
        embeds = [self._encode_one(prompt, max_sequence_length)]
        if not do_classifier_free_guidance:
            return embeds, []
        if isinstance(negative_prompt, (list, tuple)):
            negative_prompt = negative_prompt[0] if negative_prompt else ""
        return embeds, [self._encode_one(negative_prompt or "", max_sequence_length)]

    def _set_timesteps(self, num_inference_steps, image_seq_len):
        config = getattr(self.scheduler, "config", {})
        get = config.get if hasattr(config, "get") else (lambda _name, d: d)
        kwargs = {}
        if get("use_dynamic_shifting", False):
            kwargs["mu"] = z_image_shift(
                image_seq_len,
                get("base_image_seq_len", Z_BASE_IMAGE_SEQ_LEN),
                get("max_image_seq_len", Z_MAX_IMAGE_SEQ_LEN),
                get("base_shift", Z_BASE_SHIFT),
                get("max_shift", Z_MAX_SHIFT),
            )
        self.scheduler.set_timesteps(num_inference_steps, **kwargs)
        return self.scheduler.timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        negative_prompt=None,
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=0.0,
        cfg_truncation=1.0,
        max_sequence_length=512,
        generator=None,
        output_type="pil",
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        **kwargs,
    ):
        transformer = self.transformer
        if transformer is None:
            raise ModelError("The diffusion transformer is not loaded.")

        step = self.vae_scale_factor * 2
        if height % step or width % step:
            raise ModelError(
                f"{width}x{height} does not divide by {step}, which is what "
                "the VAE and the DiT's 2x2 patches need between them."
            )

        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt, guidance_scale > 0.0, negative_prompt, max_sequence_length
            )
        cap = (
            prompt_embeds[0]
            if isinstance(prompt_embeds, (list, tuple))
            else prompt_embeds
        )
        negative_cap = None
        if negative_prompt_embeds is not None and len(negative_prompt_embeds):
            negative_cap = (
                negative_prompt_embeds[0]
                if isinstance(negative_prompt_embeds, (list, tuple))
                else negative_prompt_embeds
            )

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latents = torch.randn(
            (1, transformer.in_channels, latent_height, latent_width),
            generator=generator,
            dtype=torch.float32,
        )

        image_seq_len = (latent_height // 2) * (latent_width // 2)
        timesteps = self._set_timesteps(num_inference_steps, image_seq_len)

        do_cfg = float(guidance_scale) > 0.0 and negative_cap is not None
        total = len(timesteps)

        for index, t in enumerate(timesteps):
            if float(t) == 0.0 and index == total - 1:
                if callback_on_step_end is not None:
                    callback_on_step_end(self, index, t, {"latents": latents})
                continue

            timestep = (1000.0 - float(t)) / 1000.0
            timestep_tensor = torch.tensor([timestep], dtype=torch.float32)

            scale = float(guidance_scale)
            if (
                do_cfg
                and cfg_truncation is not None
                and float(cfg_truncation) <= 1
                and timestep > float(cfg_truncation)
            ):
                scale = 0.0

            x = latents[0].unsqueeze(1)
            conditional = transformer(x, timestep_tensor, cap).float()
            pred = conditional
            if do_cfg and scale > 0.0:
                negative = transformer(x, timestep_tensor, negative_cap).float()
                pred = conditional + scale * (conditional - negative)

            noise_pred = -pred.squeeze(1).unsqueeze(0)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_on_step_end(self, index, t, {"latents": latents})

        if output_type == "latent":
            return ZImageOutput([latents])

        image = self.decode_latents(latents)
        return ZImageOutput(
            self.image_processor.postprocess(image, output_type=output_type)
        )

    def decode_latents(self, latents):
        config = self.vae.config
        shift = getattr(config, "shift_factor", None) or 0.0
        scaled = latents.to(torch.float32) / config.scaling_factor + shift
        return self.vae.decode(scaled.to(self.vae.dtype), return_dict=False)[0]


Z_LORA_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora_a.weight")
Z_LORA_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora_b.weight")
Z_LORA_PREFIXES = (
    "transformer.",
    "diffusion_model.",
    "model.diffusion_model.",
    "lora_unet_",
    "lora_transformer_",
    "base_model.model.",
)


def _lora_target(key, suffix, known):
    name = key[: -len(suffix)]
    for prefix in Z_LORA_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = name.replace(".default", "")
    candidates = [normalize_z_image_key(name)]
    if "_" in name and candidates[0] not in known:
        candidates.append(normalize_z_image_key(name.replace("_", ".")))
        candidates.append(
            normalize_z_image_key(
                name.replace("_", ".")
                .replace("to.q", "to_q")
                .replace("to.k", "to_k")
                .replace("to.v", "to_v")
                .replace("to.out", "to_out")
                .replace("norm.q", "norm_q")
                .replace("norm.k", "norm_k")
                .replace("feed.forward", "feed_forward")
                .replace("attention.norm", "attention_norm")
                .replace("ffn.norm", "ffn_norm")
                .replace("noise.refiner", "noise_refiner")
                .replace("context.refiner", "context_refiner")
                .replace("adaLN.modulation", "adaLN_modulation")
            )
        )
    for candidate in candidates:
        if candidate in known:
            return candidate
    return None


def is_vulkan_transformer(model):
    return isinstance(model, animus_vulkan.VulkanZImage)


def apply_z_image_lora(model, state_dict, adapter="lora", scale=1.0):
    on_vulkan = is_vulkan_transformer(model)
    if on_vulkan:
        known = set(model.lora_targets)
    else:
        known = {
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }
    pairs = {}
    for key in state_dict:
        for suffix in Z_LORA_DOWN_SUFFIXES:
            if key.endswith(suffix):
                target = _lora_target(key, suffix, known)
                pairs.setdefault(target or key, {})["down"] = key
                pairs.setdefault(target or key, {})["matched"] = target
                break
        else:
            for suffix in Z_LORA_UP_SUFFIXES:
                if key.endswith(suffix):
                    target = _lora_target(key, suffix, known)
                    pairs.setdefault(target or key, {})["up"] = key
                    pairs.setdefault(target or key, {})["matched"] = target
                    break

    applied, skipped = 0, []
    for target, parts in pairs.items():
        matched = parts.get("matched")
        if matched is None or "down" not in parts or "up" not in parts:
            skipped.append(target)
            continue
        down = state_dict[parts["down"]].to(torch.float32)
        up = state_dict[parts["up"]].to(torch.float32)

        rank = down.shape[0]
        alpha_key = parts["down"].rsplit(".lora", 1)[0] + ".alpha"
        alpha = state_dict.get(alpha_key)
        factor = 1.0
        if alpha is not None and rank:
            factor = float(alpha.item()) / rank

        if on_vulkan:
            wanted = model.lora_shape(matched)
            if wanted is None or (up.shape[0], down.shape[1]) != wanted:
                skipped.append(target)
                continue
            model.add_lora(matched, adapter, down.numpy(), up.numpy(), scale, factor)
            applied += 1
            continue

        module = promote_to_quant_linear(model, matched)
        if module is None:
            skipped.append(target)
            continue
        if down.shape[1] != module.in_features or up.shape[0] != module.out_features:
            skipped.append(target)
            continue
        module.add_lora(adapter, down, up, scale, factor)
        applied += 1

    return applied, skipped


def set_z_image_lora_scale(model, adapter, scale):
    if is_vulkan_transformer(model):
        model.set_lora_scale(adapter, scale)
        return
    for module in model.modules():
        if isinstance(module, ZQuantLinear):
            module.set_lora_scale(adapter, scale)


def clear_z_image_loras(model):
    if is_vulkan_transformer(model):
        model.clear_loras()
        return
    for module in model.modules():
        if isinstance(module, ZQuantLinear):
            module.clear_loras()


def release_transformer(pipe):
    drop_transformer(getattr(pipe, "transformer", None))
    vae = getattr(pipe, "vae", None)
    if isinstance(vae, animus_vulkan.VulkanVAE):
        vae.release()


def drop_transformer(transformer):
    if is_vulkan_transformer(transformer):
        transformer.release()


def zimage_devices():
    try:
        return animus_vulkan.vulkan_devices()
    except Exception as e:  # noqa: BLE001
        print(f"Could not list the Vulkan devices: {e}.")
        return []


def zimage_default_device():
    try:
        return animus_vulkan.preferred_vulkan_device() or ZIMAGE_CPU_DEVICE
    except Exception:  # noqa: BLE001
        return ZIMAGE_CPU_DEVICE


class GeneratePane:
    name = "generate"
    mode_label = "Anima"
    output_label = "Generated Image (Anima)"

    model_name = "Anima"
    model_label = "DiT (GGUF):"
    browse_title = "Select Anima GGUF (diffusion transformer)"
    count_tokenizer = "Qwen/Qwen3-0.6B"
    count_tokenizer_subfolder = None

    default_dit = ANIMA_DEFAULT_DIT
    default_steps = ANIMA_DEFAULT_STEPS
    default_guidance = ANIMA_DEFAULT_GUIDANCE
    default_size = ANIMA_DEFAULT_SIZE
    default_sampler = ANIMA_DEFAULT_SAMPLER
    default_shift = ANIMA_DEFAULT_SHIFT
    samplers = ANIMA_SAMPLERS
    token_limit = ANIMA_TOKEN_LIMIT
    size_multiple = 64
    max_steps = 100
    max_sequence_length = None
    converts_loras = True
    latent_rgb_factors = ANIMA_LATENT_RGB_FACTORS
    latent_rgb_bias = ANIMA_LATENT_RGB_BIAS

    def __init__(self, window):
        self.window = window

        self.pipe = None
        self._base_scheduler = None
        self._loaded_adapters = {}
        self._prompt_cache = None
        self.generating = False
        self.loading_model = False
        self.stop_event = threading.Event()
        self.generation_thread = None
        self.load_thread = None
        self.current_image_path = None
        self.stop_click_count = 0
        self._loading_settings = False
        self.preview_shown = False
        self._total_steps = 0
        self._latent_rgb_weight = torch.tensor(
            self.latent_rgb_factors, dtype=torch.float32
        )
        self._latent_rgb_bias = torch.tensor(self.latent_rgb_bias, dtype=torch.float32)

        self.tokenizer = None
        self._tokenizer_tried = False

    @property
    def busy(self):
        return self.generating or self.loading_model

    def build_controls(self):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_size_request(-1, CONTROLS_HEIGHT)

        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        controls_box.set_border_width(10)
        scrolled.add(controls_box)

        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        model_label = Gtk.Label(label=self.model_label)
        model_label.set_size_request(100, -1)
        model_label.set_xalign(0)
        model_box.pack_start(model_label, False, False, 0)

        self.model_entry = Gtk.Entry()
        self.model_entry.set_text(self.default_dit)
        model_box.pack_start(self.model_entry, True, True, 0)

        model_browse_btn = Gtk.Button(label="Browse...")
        model_browse_btn.connect("clicked", self.on_browse_model)
        model_box.pack_start(model_browse_btn, False, False, 0)

        controls_box.pack_start(model_box, False, False, 0)

        self.lora_entries = []
        self.lora_weight_entries = []

        for i in range(NUM_LORA_SLOTS):
            lora_frame = Gtk.Frame(label=f"LoRA {i + 1}")
            lora_frame_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
            lora_frame_box.set_border_width(5)
            lora_frame.add(lora_frame_box)

            lora_path_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            lora_path_label = Gtk.Label(label="Path or Repository:")
            lora_path_label.set_size_request(80, -1)
            lora_path_label.set_xalign(0)
            lora_path_box.pack_start(lora_path_label, False, False, 0)

            lora_entry = Gtk.Entry()
            lora_path_box.pack_start(lora_entry, True, True, 0)
            self.lora_entries.append(lora_entry)

            browse_btn = Gtk.Button(label="Browse...")
            browse_btn.connect("clicked", self.on_browse_lora, i)
            lora_path_box.pack_start(browse_btn, False, False, 0)

            clear_btn = Gtk.Button(label="Clear")
            clear_btn.connect("clicked", self.on_clear_lora, i)
            lora_path_box.pack_start(clear_btn, False, False, 0)

            lora_frame_box.pack_start(lora_path_box, False, False, 0)

            lora_weight_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

            weight_name_label = Gtk.Label(label="Weight Name:")
            weight_name_label.set_size_request(80, -1)
            weight_name_label.set_xalign(0)
            lora_weight_box.pack_start(weight_name_label, False, False, 0)

            weight_name_entry = Gtk.Entry()
            lora_weight_box.pack_start(weight_name_entry, True, True, 0)

            weight_label = Gtk.Label(label="Strength:")
            weight_label.set_size_request(60, -1)
            lora_weight_box.pack_start(weight_label, False, False, 0)

            weight_spin = Gtk.SpinButton()
            weight_adj = Gtk.Adjustment(
                value=0.5, lower=0.0, upper=2.0, step_increment=0.05, page_increment=0.1
            )
            weight_spin.set_adjustment(weight_adj)
            weight_spin.set_digits(2)
            weight_spin.set_size_request(80, -1)
            lora_weight_box.pack_start(weight_spin, False, False, 0)

            self.lora_weight_entries.append((weight_name_entry, weight_spin))
            lora_frame_box.pack_start(lora_weight_box, False, False, 0)

            controls_box.pack_start(lora_frame, False, False, 0)

        resolution_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        resolution_label = Gtk.Label(label="Size (W x H):")
        resolution_label.set_size_request(100, -1)
        resolution_label.set_xalign(0)
        resolution_box.pack_start(resolution_label, False, False, 0)

        self.width_spin = Gtk.SpinButton()
        width_adj = Gtk.Adjustment(
            value=self.default_size,
            lower=256,
            upper=1536,
            step_increment=self.size_multiple,
            page_increment=self.size_multiple * 2,
        )
        self.width_spin.set_adjustment(width_adj)
        self.width_spin.set_size_request(90, -1)
        resolution_box.pack_start(self.width_spin, False, False, 0)

        self.height_spin = Gtk.SpinButton()
        height_adj = Gtk.Adjustment(
            value=self.default_size,
            lower=256,
            upper=1536,
            step_increment=self.size_multiple,
            page_increment=self.size_multiple * 2,
        )
        self.height_spin.set_adjustment(height_adj)
        self.height_spin.set_size_request(90, -1)
        resolution_box.pack_start(self.height_spin, False, False, 0)
        controls_box.pack_start(resolution_box, False, False, 0)

        steps_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        steps_label = Gtk.Label(label="Steps:")
        steps_label.set_size_request(100, -1)
        steps_label.set_xalign(0)
        steps_box.pack_start(steps_label, False, False, 0)

        self.steps_spin = Gtk.SpinButton()
        steps_adj = Gtk.Adjustment(
            value=self.default_steps,
            lower=1,
            upper=self.max_steps,
            step_increment=1,
            page_increment=5,
        )
        self.steps_spin.set_adjustment(steps_adj)
        self.steps_spin.set_size_request(100, -1)
        steps_box.pack_start(self.steps_spin, False, False, 0)
        controls_box.pack_start(steps_box, False, False, 0)

        guidance_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        guidance_label = Gtk.Label(label="Guidance:")
        guidance_label.set_size_request(100, -1)
        guidance_label.set_xalign(0)
        guidance_box.pack_start(guidance_label, False, False, 0)

        self.guidance_spin = Gtk.SpinButton()
        guidance_adj = Gtk.Adjustment(
            value=self.default_guidance,
            lower=0.0,
            upper=20.0,
            step_increment=0.5,
            page_increment=1.0,
        )
        self.guidance_spin.set_adjustment(guidance_adj)
        self.guidance_spin.set_digits(1)
        self.guidance_spin.set_size_request(100, -1)
        guidance_box.pack_start(self.guidance_spin, False, False, 0)
        controls_box.pack_start(guidance_box, False, False, 0)

        sampler_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        sampler_label = Gtk.Label(label="Sampler:")
        sampler_label.set_size_request(100, -1)
        sampler_label.set_xalign(0)
        sampler_box.pack_start(sampler_label, False, False, 0)

        self.sampler_combo = Gtk.ComboBoxText()
        for sampler_name in self.samplers:
            self.sampler_combo.append(sampler_name, sampler_name)
        self.sampler_combo.set_active_id(self.default_sampler)

        sampler_box.pack_start(self.sampler_combo, False, False, 0)
        controls_box.pack_start(sampler_box, False, False, 0)

        shift_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        shift_label = Gtk.Label(label="Shift:")
        shift_label.set_size_request(100, -1)
        shift_label.set_xalign(0)
        shift_box.pack_start(shift_label, False, False, 0)

        self.shift_spin = Gtk.SpinButton()
        shift_adj = Gtk.Adjustment(
            value=self.default_shift,
            lower=0.10,
            upper=12.0,
            step_increment=0.05,
            page_increment=0.5,
        )
        self.shift_spin.set_adjustment(shift_adj)
        self.shift_spin.set_digits(2)
        self.shift_spin.set_size_request(100, -1)
        shift_box.pack_start(self.shift_spin, False, False, 0)
        controls_box.pack_start(shift_box, False, False, 0)

        self._extra_controls(controls_box)

        seed_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        seed_label = Gtk.Label(label="Seed:")
        seed_label.set_size_request(100, -1)
        seed_label.set_xalign(0)
        seed_box.pack_start(seed_label, False, False, 0)

        self.seed_spin = Gtk.SpinButton()
        seed_adj = Gtk.Adjustment(
            value=ANIMA_DEFAULT_SEED,
            lower=-1,
            upper=ANIMA_SEED_MAX,
            step_increment=1,
            page_increment=1000,
        )
        self.seed_spin.set_adjustment(seed_adj)
        self.seed_spin.set_size_request(150, -1)
        seed_box.pack_start(self.seed_spin, False, False, 0)

        seed_random_btn = Gtk.Button(label="Random")
        seed_random_btn.connect(
            "clicked", lambda _btn: self.seed_spin.set_value(ANIMA_DEFAULT_SEED)
        )
        seed_box.pack_start(seed_random_btn, False, False, 0)
        controls_box.pack_start(seed_box, False, False, 0)

        preview_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        preview_label = Gtk.Label(label="Live Preview:")
        preview_label.set_size_request(100, -1)
        preview_label.set_xalign(0)
        preview_box.pack_start(preview_label, False, False, 0)
        self.preview_check = Gtk.CheckButton(label="Generate a preview each step")
        self.preview_check.set_active(True)
        preview_box.pack_start(self.preview_check, False, False, 0)
        controls_box.pack_start(preview_box, False, False, 0)

        trigger_label = Gtk.Label(label="Trigger:")
        trigger_label.set_xalign(0)
        controls_box.pack_start(trigger_label, False, False, 0)

        trigger_scroll = Gtk.ScrolledWindow()
        trigger_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        trigger_scroll.set_size_request(-1, 50)

        self.trigger_text = Gtk.TextView()
        self.trigger_text.set_wrap_mode(Gtk.WrapMode.WORD)
        self.trigger_text.get_buffer().connect("changed", self.on_text_changed)
        trigger_scroll.add(self.trigger_text)
        controls_box.pack_start(trigger_scroll, False, False, 0)

        self.trigger_token_label = Gtk.Label(label=f"0/{self.token_limit}")
        self.trigger_token_label.set_xalign(1)
        controls_box.pack_start(self.trigger_token_label, False, False, 0)

        prompt_label = Gtk.Label(label="Prompt:")
        prompt_label.set_xalign(0)
        controls_box.pack_start(prompt_label, False, False, 0)

        prompt_scroll = Gtk.ScrolledWindow()
        prompt_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        prompt_scroll.set_size_request(-1, 80)

        self.prompt_text = Gtk.TextView()
        self.prompt_text.set_wrap_mode(Gtk.WrapMode.WORD)
        self.prompt_text.get_buffer().connect("changed", self.on_text_changed)
        prompt_scroll.add(self.prompt_text)
        controls_box.pack_start(prompt_scroll, False, False, 0)

        self.prompt_token_label = Gtk.Label(label=f"0/{self.token_limit}")
        self.prompt_token_label.set_xalign(1)
        controls_box.pack_start(self.prompt_token_label, False, False, 0)

        neg_prompt_label = Gtk.Label(label="Negative Prompt:")
        neg_prompt_label.set_xalign(0)
        controls_box.pack_start(neg_prompt_label, False, False, 0)

        neg_prompt_scroll = Gtk.ScrolledWindow()
        neg_prompt_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        neg_prompt_scroll.set_size_request(-1, 50)

        self.neg_prompt_text = Gtk.TextView()
        self.neg_prompt_text.set_wrap_mode(Gtk.WrapMode.WORD)
        self.neg_prompt_text.get_buffer().connect("changed", self.on_text_changed)
        neg_prompt_scroll.add(self.neg_prompt_text)
        controls_box.pack_start(neg_prompt_scroll, False, False, 0)

        self.neg_prompt_token_label = Gtk.Label(label=f"0/{self.negative_token_limit}")
        self.neg_prompt_token_label.set_xalign(1)
        controls_box.pack_start(self.neg_prompt_token_label, False, False, 0)

        return scrolled

    def build_output_page(self):
        image_scrolled = Gtk.ScrolledWindow()
        image_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        image_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)

        self.image_display = Gtk.Image()
        image_box.pack_start(self.image_display, True, True, 0)

        image_button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        image_button_box.set_halign(Gtk.Align.CENTER)

        self.delete_image_button = Gtk.Button(label="Delete Image")
        self.delete_image_button.connect("clicked", self.on_delete_image_clicked)
        self.delete_image_button.set_sensitive(False)
        image_button_box.pack_start(self.delete_image_button, False, False, 0)

        image_box.pack_start(image_button_box, False, False, 5)

        image_scrolled.add(image_box)

        return image_scrolled

    def build_actions(self):
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

        self.load_button = Gtk.Button(label="Load Model and LoRAs")
        self.load_button.connect("clicked", self.on_load_clicked)
        button_box.pack_start(self.load_button, True, True, 0)

        self.generate_button = Gtk.Button(label="Generate")
        self.generate_button.connect("clicked", self.on_generate_clicked)
        self.generate_button.set_sensitive(False)
        button_box.pack_start(self.generate_button, True, True, 0)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.connect("clicked", self.on_stop_clicked)
        self.stop_button.set_sensitive(False)
        self.stop_button.set_no_show_all(True)
        button_box.pack_start(self.stop_button, True, True, 0)

        self.restore_defaults_button = Gtk.Button(label="Restore Defaults")
        self.restore_defaults_button.connect(
            "clicked", self.on_restore_defaults_clicked
        )
        button_box.pack_start(self.restore_defaults_button, True, True, 0)

        return button_box

    def shutdown(self):
        if self.generating or self.loading_model:
            self.stop_event.set()
            update_status("Waiting for operations to stop...")

        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=GENERATION_THREAD_TIMEOUT)

        if self.load_thread and self.load_thread.is_alive():
            self.load_thread.join(timeout=LOAD_THREAD_TIMEOUT)

    def _counting_tokenizer(self):
        if self.tokenizer is not None or self._tokenizer_tried:
            return self.tokenizer
        self._tokenizer_tried = True
        try:
            from transformers import AutoTokenizer

            kwargs = {}
            if self.count_tokenizer_subfolder:
                kwargs["subfolder"] = self.count_tokenizer_subfolder
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.count_tokenizer, **kwargs
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load the tokenizer for token counting: {e}.")
        return self.tokenizer

    def count_tokens(self, text):
        if not text or self._counting_tokenizer() is None:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:  # noqa: BLE001
            print(f"Error counting tokens: {e}.")
            return 0

    def on_text_changed(self, widget=None):
        trigger_buffer = self.trigger_text.get_buffer()
        trigger_text = trigger_buffer.get_text(
            trigger_buffer.get_start_iter(), trigger_buffer.get_end_iter(), False
        )

        prompt_buffer = self.prompt_text.get_buffer()
        prompt_text = prompt_buffer.get_text(
            prompt_buffer.get_start_iter(), prompt_buffer.get_end_iter(), False
        )

        neg_buffer = self.neg_prompt_text.get_buffer()
        neg_text = neg_buffer.get_text(
            neg_buffer.get_start_iter(), neg_buffer.get_end_iter(), False
        )

        if not (trigger_text or prompt_text or neg_text):
            for label in (
                self.trigger_token_label,
                self.prompt_token_label,
                self.neg_prompt_token_label,
            ):
                self._update_token_label(label, 0)
            return

        if self._counting_tokenizer() is None:
            return

        trigger_tokens = self.count_tokens(trigger_text)

        if trigger_text and prompt_text:
            combined_prompt = f"{trigger_text}, {prompt_text}"
        elif trigger_text:
            combined_prompt = trigger_text
        else:
            combined_prompt = prompt_text
        combined_tokens = self.count_tokens(combined_prompt)

        neg_tokens = self.count_tokens(neg_text)

        self._update_token_label(self.trigger_token_label, trigger_tokens)
        self._update_token_label(self.prompt_token_label, combined_tokens)
        self._update_token_label(
            self.neg_prompt_token_label, neg_tokens, self.negative_token_limit
        )

    @property
    def negative_token_limit(self):
        return self.token_limit

    def _update_token_label(self, label, count, limit=None):
        if limit is None:
            limit = self.token_limit
        warning = int(limit * TOKEN_WARNING_FRACTION)
        if count > limit:
            color = "red"
        elif count > warning:
            color = "orange"
        else:
            color = "green"
        weight = ' weight="bold"' if count > warning else ""
        label.set_markup(f'<span foreground="{color}"{weight}>{count}/{limit}</span>')

    def load_settings(self, settings):
        self._loading_settings = True
        try:
            if "model" in settings:
                self.model_entry.set_text(settings["model"])

            if "loras" in settings:
                for i, lora_data in enumerate(settings["loras"][:NUM_LORA_SLOTS]):
                    if i < len(self.lora_entries):
                        self.lora_entries[i].set_text(lora_data.get("path", ""))
                        weight_name_entry, weight_spin = self.lora_weight_entries[i]
                        weight_name_entry.set_text(lora_data.get("weight_name", ""))
                        weight_spin.set_value(lora_data.get("weight_value", 0.5))

            if "steps" in settings:
                self.steps_spin.set_value(settings["steps"])
            if "guidance" in settings:
                self.guidance_spin.set_value(settings["guidance"])
            if settings.get("sampler") in self.samplers:
                self.sampler_combo.set_active_id(settings["sampler"])
            if "shift" in settings:
                self.shift_spin.set_value(settings["shift"])
            if "seed" in settings:
                self.seed_spin.set_value(settings["seed"])
            if "width" in settings:
                self.width_spin.set_value(settings["width"])
            if "height" in settings:
                self.height_spin.set_value(settings["height"])
            if "preview" in settings:
                self.preview_check.set_active(settings["preview"])
            if "trigger" in settings:
                self.trigger_text.get_buffer().set_text(settings["trigger"])
            if "prompt" in settings:
                self.prompt_text.get_buffer().set_text(settings["prompt"])
            if "negative_prompt" in settings:
                self.neg_prompt_text.get_buffer().set_text(settings["negative_prompt"])
        except Exception as e:  # noqa: BLE001
            print(f"Error loading settings: {e}.")
        finally:
            self._loading_settings = False

        self.on_text_changed(None)

    def collect_settings(self):
        trigger_buffer = self.trigger_text.get_buffer()
        trigger = trigger_buffer.get_text(
            trigger_buffer.get_start_iter(), trigger_buffer.get_end_iter(), False
        )

        prompt_buffer = self.prompt_text.get_buffer()
        prompt = prompt_buffer.get_text(
            prompt_buffer.get_start_iter(), prompt_buffer.get_end_iter(), False
        )

        neg_buffer = self.neg_prompt_text.get_buffer()
        negative_prompt = neg_buffer.get_text(
            neg_buffer.get_start_iter(), neg_buffer.get_end_iter(), False
        )

        settings = {
            "model": self.model_entry.get_text(),
            "loras": [],
            "steps": int(self.steps_spin.get_value()),
            "guidance": spin_value(self.guidance_spin),
            "sampler": self.sampler_combo.get_active_id() or self.default_sampler,
            "shift": spin_value(self.shift_spin),
            "seed": int(self.seed_spin.get_value()),
            "width": int(self.width_spin.get_value()),
            "height": int(self.height_spin.get_value()),
            "preview": self.preview_check.get_active(),
            "trigger": trigger,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }

        for lora_entry, (weight_name_entry, weight_spin) in zip(
            self.lora_entries, self.lora_weight_entries
        ):
            settings["loras"].append(
                {
                    "path": lora_entry.get_text(),
                    "weight_name": weight_name_entry.get_text(),
                    "weight_value": spin_value(weight_spin),
                }
            )

        return settings

    def _copy_to_library(self, src_path, dest_file, description):
        try:
            src_size = src_path.stat().st_size

            if dest_file.exists() and dest_file.stat().st_size == src_size:
                print(f"Already in the library (up to date): {src_path.name}")
                return True

            existing_size = dest_file.stat().st_size if dest_file.exists() else 0

            dest_file.parent.mkdir(parents=True, exist_ok=True)

            free = shutil.disk_usage(dest_file.parent).free
            if free + existing_size < src_size + MIN_FREE_DISK_MARGIN:
                print(
                    f"Not enough disk space to copy {description} "
                    f"({src_path.name}): need "
                    f"{_format_size(src_size + MIN_FREE_DISK_MARGIN)}, only "
                    f"{_format_size(free + existing_size)} available. "
                    "Free up space and try again."
                )
                return False

            print(
                f"Copying {description} to the library: {src_path.name} "
                f"({_format_size(src_size)})..."
            )
            shutil.copy2(src_path, dest_file)

            copied_size = dest_file.stat().st_size
            if copied_size != src_size:
                print(
                    f"Copy verification failed for {description}: expected "
                    f"{_format_size(src_size)}, got {_format_size(copied_size)}."
                )
                self._discard_partial_copy(dest_file)
                return False

            print(
                f"Copied {description} successfully, verified "
                f"{_format_size(copied_size)}: {src_path.name}"
            )
            return True
        except OSError as e:
            if e.errno == errno.ENOSPC:
                print(
                    f"Ran out of disk space while copying {description}: "
                    f"{src_path.name}. Free up space and try again."
                )
            else:
                print(f"Failed to copy {description}: {e}!")
            self._discard_partial_copy(dest_file)
            return False
        except Exception as e:  # noqa: BLE001
            print(f"Failed to copy {description}: {e}!")
            self._discard_partial_copy(dest_file)
            return False

    def _discard_partial_copy(self, dest_file):
        try:
            if dest_file.exists():
                dest_file.unlink()
                print(f"Removed incomplete file: {dest_file}")
        except OSError as e:
            print(f"Warning: could not remove incomplete file {dest_file}: {e}.")

    def on_browse_model(self, button):
        dialog = Gtk.FileChooserDialog(
            title=self.browse_title,
            parent=self.window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        filter_model = Gtk.FileFilter()
        filter_model.set_name("GGUF / checkpoint files")
        for pattern in ("*.gguf", "*.safetensors", "*.ckpt"):
            filter_model.add_pattern(pattern)
        dialog.add_filter(filter_model)

        filter_all = Gtk.FileFilter()
        filter_all.set_name("All files")
        filter_all.add_pattern("*")
        dialog.add_filter(filter_all)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            selected_path = dialog.get_filename()
            if not selected_path:
                dialog.destroy()
                return

            path_obj = Path(selected_path)

            if path_obj.is_file():
                dest_file = DIT_DIR / path_obj.stem / path_obj.name
                if self._copy_to_library(path_obj, dest_file, "model file"):
                    self.model_entry.set_text(str(dest_file))
                else:
                    self.model_entry.set_text(str(path_obj))
            else:
                print(f"Warning: selected path is not a file: {selected_path}.")

        dialog.destroy()

    def on_browse_lora(self, button, lora_index):
        dialog = Gtk.FileChooserDialog(
            title="Select LoRA Weight File",
            parent=self.window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        filter_lora = Gtk.FileFilter()
        filter_lora.set_name("LoRA weight files")
        for pattern in ("*.safetensors", "*.ckpt", "*.pt", "*.bin"):
            filter_lora.add_pattern(pattern)
        dialog.add_filter(filter_lora)

        filter_all = Gtk.FileFilter()
        filter_all.set_name("All files")
        filter_all.add_pattern("*")
        dialog.add_filter(filter_all)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            selected_path = dialog.get_filename()
            if not selected_path:
                dialog.destroy()
                return

            path_obj = Path(selected_path)

            if path_obj.is_file():
                weight_name_entry, _ = self.lora_weight_entries[lora_index]
                parent_dir = path_obj.parent

                if not is_huggingface_repo(str(parent_dir)):
                    dest_file = LORA_DIR / path_obj.stem / path_obj.name

                    if self._copy_to_library(path_obj, dest_file, "LoRA weight file"):
                        self.lora_entries[lora_index].set_text(str(dest_file.parent))
                        weight_name_entry.set_text(path_obj.name)
                    else:
                        weight_name_entry.set_text(path_obj.name)
                        self.lora_entries[lora_index].set_text(str(parent_dir))
                else:
                    weight_name_entry.set_text(path_obj.name)
                    self.lora_entries[lora_index].set_text(str(parent_dir))
            else:
                print(f"Warning: selected path is not a file: {selected_path}.")

        dialog.destroy()

    def on_clear_lora(self, button, lora_index):
        self.lora_entries[lora_index].set_text("")
        weight_name_entry, weight_spin = self.lora_weight_entries[lora_index]
        weight_name_entry.set_text("")
        weight_spin.set_value(0.5)

    def on_load_clicked(self, button):
        if self.busy or not self.window.claim(self):
            return

        self.loading_model = True
        self.stop_event.clear()
        self.stop_click_count = 0
        self.load_button.set_sensitive(False)
        self.generate_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.stop_button.show()

        if self.pipe is not None:
            update_status("Reloading model... Click Stop to cancel.")
        else:
            update_status("Loading model... Click Stop to cancel.")

        self.load_thread = threading.Thread(target=self.load_model_thread, daemon=True)
        self.load_thread.start()

    def _prepare_source(self, model_name):
        model_name = normalize_huggingface_url((model_name or "").strip())
        if not model_name:
            raise ModelError(f"No {self.model_name} DiT specified.")
        return model_name

    def _discard_pipe(self):
        pipe = self.pipe
        self.pipe = None
        if pipe is not None:
            release_transformer(pipe)
            del pipe
        gc.collect()

    def _check_stop_loading(self, cleanup_pipe=False):
        if self.stop_event.is_set():
            update_status("Interrupted by user.")
            if cleanup_pipe and self.pipe is not None:
                self._discard_pipe()
            return True
        return False

    def load_model_thread(self):
        try:
            if self._check_stop_loading():
                return

            model_name = self._prepare_source(self.model_entry.get_text().strip())

            if self.pipe is not None:
                if self._check_stop_loading():
                    return
                update_status("Cleaning up existing model...")
                self._discard_pipe()
                self._base_scheduler = None
                self._loaded_adapters = {}
                self._prompt_cache = None
                gc.collect()

            for released in self.window.release_other_models(self):
                update_status(f"Unloaded {released} to make room. Reload it later.")

            if self._check_stop_loading():
                return

            self.pipe = self._load_pipeline(model_name)
            if self.pipe is None:
                return

            if self._check_stop_loading(cleanup_pipe=True):
                return

            self._load_loras()

            if self._check_stop_loading(cleanup_pipe=True):
                return

            GLib.idle_add(self._enable_generate_and_load)
            update_status("Ready!")

        except KeyboardInterrupt:
            update_status("Interrupted by user.")
            self._discard_pipe()
            GLib.idle_add(self._enable_load)
        except Exception as e:  # noqa: BLE001
            if self.stop_event.is_set():
                update_status("Interrupted by user.")
            else:
                traceback.print_exc()
                update_status(f"Error loading model: {e}!")
            GLib.idle_add(self._enable_load)
        finally:
            self.loading_model = False
            if self.pipe is None:
                GLib.idle_add(self._enable_load)
                GLib.idle_add(self._model_released)
            GLib.idle_add(self._hide_stop_button)

    def _load_pipeline(self, dit_source):
        if not ANIMA_AVAILABLE:
            raise ModelError("Anima requires diffusers >= 0.39.0 and gguf.")

        _install_cosmos_torchvision_shim()
        _install_anima_denoise_hook()

        dit_source = normalize_huggingface_url((dit_source or "").strip())
        if not dit_source:
            raise ModelError("No Anima DiT specified.")

        update_status(f"Loading Anima DiT from {dit_source}...")
        transformer = CosmosTransformer3DModel.from_single_file(
            dit_source,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32),
            config=ANIMA_COMPONENTS_REPO,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )

        if self._check_stop_loading():
            del transformer
            gc.collect()
            return None

        update_status(f"Loading Anima components from {ANIMA_COMPONENTS_REPO}...")
        pipe = ModularPipeline.from_pretrained(ANIMA_COMPONENTS_REPO)
        comp_names = (
            getattr(pipe, "pretrained_component_names", None)
            or getattr(pipe, "component_names", None)
            or []
        )
        other = [n for n in comp_names if n != "transformer"]
        loaded_selectively = False
        try:
            if other:
                pipe.load_components(names=other, torch_dtype=torch.float32)
                loaded_selectively = True
        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: selective component load failed: {e}. Loading all "
                "components, including a float32 transformer that the GGUF "
                "one below replaces. That is a lot of memory to touch twice."
            )
        if not loaded_selectively:
            pipe.load_components(torch_dtype=torch.float32)
            gc.collect()

        if self._check_stop_loading():
            del transformer
            del pipe
            gc.collect()
            return None

        update_status("Injecting the GGUF DiT into the pipeline...")
        pipe.update_components(transformer=transformer)
        gc.collect()

        try:
            vae = getattr(pipe, "vae", None)
            if vae is not None:
                if hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                if hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()
                    update_status("Enabled VAE tiling and slicing.")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: {e}.")

        try:
            pipe.to("cpu")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not move pipeline to CPU: {e}.")

        self._base_scheduler = getattr(pipe, "scheduler", None)

        gc.collect()
        return pipe

    def _lora_local_file(self, path, weight_name):
        try:
            p = Path(path)
            if weight_name:
                candidate = p / weight_name
                if candidate.is_file():
                    return candidate
            if p.is_file():
                return p
        except Exception:  # noqa: BLE001, S110
            pass
        return None

    def _convert_kohya_anima_lora(self, state_dict):
        attn_map = {
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

        groups = {}
        dropped_text_encoder = 0
        dropped_other = 0
        unmapped = set()
        for key, val in state_dict.items():
            if not key.startswith("lora_unet_blocks_"):
                if key.startswith("lora_te"):
                    dropped_text_encoder += 1
                else:
                    dropped_other += 1
                continue
            name_part, _, tail = key.partition(".")
            m = re.match(r"^lora_unet_blocks_(\d+)_(.+)$", name_part)
            if not m:
                dropped_other += 1
                continue
            block_idx, suffix = m.group(1), m.group(2)
            module = attn_map.get(suffix)
            if module is None:
                unmapped.add(suffix)
                continue
            dkey = f"transformer.transformer_blocks.{block_idx}.{module}"
            g = groups.setdefault(dkey, {})
            if tail == "lora_down.weight":
                g["down"] = val
            elif tail == "lora_up.weight":
                g["up"] = val
            elif tail == "alpha":
                g["alpha"] = val

        if not groups:
            return None

        converted = {}
        for dkey, g in groups.items():
            if "down" not in g or "up" not in g:
                continue
            down = g["down"].to(torch.float32)
            up = g["up"].to(torch.float32)
            rank = down.shape[0]
            scale = 1.0
            if "alpha" in g and rank:
                alpha = float(g["alpha"].to(torch.float32).reshape(-1)[0].item())
                scale = alpha / rank
            converted[f"{dkey}.lora_A.weight"] = down
            converted[f"{dkey}.lora_B.weight"] = up * scale

        return converted or None

    def _load_loras(self):
        loras_to_load = []
        for i, (lora_entry, (weight_name_entry, weight_spin)) in enumerate(
            zip(self.lora_entries, self.lora_weight_entries)
        ):
            lora_path = lora_entry.get_text().strip()
            weight_name = weight_name_entry.get_text().strip()
            weight_value = spin_value(weight_spin)
            if lora_path:
                loras_to_load.append(
                    {
                        "path": lora_path,
                        "weight_name": weight_name,
                        "weight_value": weight_value,
                        "adapter_name": f"lora_{i}",
                        "slot": i,
                    }
                )

        if not loras_to_load:
            return

        adapter_weights = {}
        for lora_info in loras_to_load:
            if self._check_stop_loading(cleanup_pipe=True):
                return

            label = lora_info["path"]
            if lora_info["weight_name"]:
                label = f"{label}/{lora_info['weight_name']}"
            update_status(f"Loading LoRA {label}...")
            try:
                converted = None
                local_file = (
                    self._lora_local_file(lora_info["path"], lora_info["weight_name"])
                    if self.converts_loras
                    else None
                )
                if local_file is not None and str(local_file).endswith(".safetensors"):
                    from safetensors.torch import load_file

                    raw = load_file(str(local_file))
                    converted = self._convert_lora(raw)
                    del raw

                if converted is not None:
                    n = sum(1 for k in converted if k.endswith("lora_A.weight"))
                    update_status(f"Converted {n} modules to diffusers format.")
                    self.pipe.load_lora_weights(
                        converted, adapter_name=lora_info["adapter_name"]
                    )
                else:
                    kwargs = {"adapter_name": lora_info["adapter_name"]}
                    if lora_info["weight_name"]:
                        kwargs["weight_name"] = lora_info["weight_name"]
                    self.pipe.load_lora_weights(lora_info["path"], **kwargs)

                adapter_weights[lora_info["adapter_name"]] = lora_info["weight_value"]
                self._loaded_adapters[lora_info["adapter_name"]] = lora_info["slot"]
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                update_status(
                    f"Warning: could not load LoRA {lora_info['adapter_name']} "
                    f"({label}): {e}. Skipping."
                )

        if not adapter_weights:
            return

        try:
            self.pipe.set_adapters(
                list(adapter_weights.keys()),
                adapter_weights=list(adapter_weights.values()),
            )
            try:
                active = self.pipe.get_active_adapters()
            except Exception:  # noqa: BLE001
                active = list(adapter_weights.keys())
            update_status(f"Activated LoRA(s) at runtime: {active}.")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            update_status(f"Warning: could not activate LoRA adapters: {e}.")

    def _enable_generate_and_load(self):
        self.generate_button.set_sensitive(True)
        self.load_button.set_sensitive(True)
        self.load_button.set_label("Reload Model and LoRAs")
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def _enable_generate(self):
        self.generate_button.set_sensitive(True)
        return False

    def _reset_generate_button(self):
        self.generate_button.set_sensitive(True)
        self.generate_button.show()
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        if self.current_image_path and not self.preview_shown:
            self.delete_image_button.set_sensitive(True)
        return False

    def _enable_load(self):
        self.load_button.set_sensitive(True)
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def _model_released(self):
        self.generate_button.set_sensitive(False)
        self.load_button.set_label("Load Model and LoRAs")
        return False

    def release_model(self):
        if self.pipe is None or self.busy:
            return False
        self._discard_pipe()
        self._base_scheduler = None
        self._loaded_adapters = {}
        self._prompt_cache = None
        gc.collect()
        GLib.idle_add(self._model_released)
        return True

    def _hide_stop_button(self):
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def on_generate_clicked(self, button):
        if self.busy or self.pipe is None or not self.window.claim(self):
            return

        self.window.save_settings()

        self.generating = True
        self.stop_event.clear()
        self.stop_click_count = 0
        self.preview_shown = False
        self._total_steps = int(self.steps_spin.get_value())
        self.window.set_progress(0.0, "Starting...")
        self.generate_button.set_sensitive(False)
        self.generate_button.hide()
        self.stop_button.set_sensitive(True)
        self.stop_button.show()
        self.delete_image_button.set_sensitive(False)

        self.generation_thread = threading.Thread(
            target=self.generate_image_thread, daemon=True
        )
        self.generation_thread.start()

    def on_stop_clicked(self, button):
        if self.generating or self.loading_model:
            self.stop_event.set()
            self.stop_click_count += 1

            update_status(f"FORCE STOP #{self.stop_click_count} - Sending interrupt...")
            print(
                f"\n*** STOP BUTTON CLICKED #{self.stop_click_count} - "
                "Sending interrupt... ***"
            )

            target_thread = (
                self.generation_thread if self.generating else self.load_thread
            )
            if target_thread and target_thread.is_alive():
                success = raise_exception_in_thread(target_thread)
                if success:
                    print("  -> Sent KeyboardInterrupt to worker thread")
                else:
                    print("  -> Thread interrupt failed")

            if os.name == "posix":
                print(f"  -> Sending SIGINT to process (PID {os.getpid()}).")
                os.kill(os.getpid(), signal.SIGINT)

    def _build_scheduler(self, sampler, base, shift):
        return build_flow_scheduler(sampler, base, shift)

    def _set_scheduler(self, scheduler):
        self.pipe.update_components(scheduler=scheduler)

    def _apply_sampler(self, sampler, shift=None, generator=None):
        base = self._base_scheduler or getattr(self.pipe, "scheduler", None)
        if base is None or self.pipe is None:
            return
        try:
            scheduler = self._build_scheduler(sampler, base, shift)
            bind_scheduler_generator(scheduler, generator)
            self._set_scheduler(scheduler)
        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: could not select sampler '{sampler}': {e}. "
                "Falling back to the pipeline default."
            )
            try:
                self._set_scheduler(base)
            except Exception:  # noqa: BLE001, S110
                pass

    def _refresh_lora_weights(self):
        if self.pipe is None or not self._loaded_adapters:
            return
        weights = {}
        for adapter_name, slot in self._loaded_adapters.items():
            if slot >= len(self.lora_weight_entries):
                continue
            _, weight_spin = self.lora_weight_entries[slot]
            weights[adapter_name] = spin_value(weight_spin)
        if not weights:
            return
        try:
            self.pipe.set_adapters(
                list(weights.keys()), adapter_weights=list(weights.values())
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not update the LoRA strengths: {e}.")

    def _apply_guidance(self, guidance):
        if self.pipe is None:
            return
        if ClassifierFreeGuidance is None:
            return
        try:
            existing = getattr(self.pipe, "guider", None)
            config = dict(getattr(existing, "config", None) or {})
            if config:
                config["guidance_scale"] = float(guidance)
                guider = ClassifierFreeGuidance.from_config(config)
            else:
                guider = ClassifierFreeGuidance(guidance_scale=float(guidance))
            self.pipe.update_components(guider=guider)
        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: could not apply guidance {guidance} to the guider "
                f"component: {e}. The pipeline's default CFG will be used."
            )

    def _convert_lora(self, state_dict):
        return self._convert_kohya_anima_lora(state_dict)

    def _begin_generation(self):
        _set_anima_stop_check(lambda: self.stop_event.is_set())
        _set_anima_step_hook(self._on_denoise_step)

    def _end_generation(self):
        _set_anima_step_hook(None)
        _set_anima_stop_check(None)

    def _extra_controls(self, controls_box):
        return

    def _extra_metadata(self):
        return {}

    def _prompt_kwargs(self, prompt, negative_prompt, guidance):
        return {"prompt": prompt, "negative_prompt": negative_prompt}

    def _pipeline_kwargs(self, guidance, steps):
        return {}

    def _adjust_size(self, width, height):
        return width, height

    def _png_metadata(self, **params):
        info = PngInfo()
        info.add_text("software", "Animus")
        info.add_text("model", self.model_entry.get_text())
        loras = []
        for lora_entry, (weight_name_entry, weight_spin) in zip(
            self.lora_entries, self.lora_weight_entries
        ):
            path = lora_entry.get_text().strip()
            if path:
                name = weight_name_entry.get_text().strip()
                loras.append(f"{path}/{name}@{spin_text(weight_spin)}")
        if loras:
            info.add_text("loras", ", ".join(loras))
        for key, value in params.items():
            info.add_text(key, str(value))
        return info

    def generate_image_thread(self):
        try:
            prompt_buffer = self.prompt_text.get_buffer()
            user_prompt = prompt_buffer.get_text(
                prompt_buffer.get_start_iter(), prompt_buffer.get_end_iter(), False
            ).strip()

            if not user_prompt:
                update_status("Error: Prompt cannot be empty!")
                GLib.idle_add(self.window.set_progress, 0.0, "Idle")
                GLib.idle_add(self._enable_generate)
                self.generating = False
                return

            neg_buffer = self.neg_prompt_text.get_buffer()
            negative_prompt = neg_buffer.get_text(
                neg_buffer.get_start_iter(), neg_buffer.get_end_iter(), False
            ).strip()

            trigger_buffer = self.trigger_text.get_buffer()
            trigger = trigger_buffer.get_text(
                trigger_buffer.get_start_iter(), trigger_buffer.get_end_iter(), False
            ).strip()

            if trigger:
                full_prompt = f"{trigger}, {user_prompt}"
            else:
                full_prompt = user_prompt

            steps = int(self.steps_spin.get_value())
            guidance = spin_value(self.guidance_spin)
            guidance_text = spin_text(self.guidance_spin)
            width, height = self._adjust_size(
                int(self.width_spin.get_value()), int(self.height_spin.get_value())
            )
            sampler = self.sampler_combo.get_active_id() or self.default_sampler
            shift = spin_value(self.shift_spin)
            shift_text = spin_text(self.shift_spin)
            seed = int(self.seed_spin.get_value())
            if seed < 0:
                seed = random.randint(0, ANIMA_SEED_MAX)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            self._apply_sampler(sampler, shift, generator)
            self._apply_guidance(guidance)
            self._refresh_lora_weights()

            update_status(
                f"Generating with {self.model_name} ({sampler}, shift "
                f"{shift_text}) at {width}x{height} with {steps} steps, "
                f"guidance {guidance_text}, and seed {seed}..."
            )

            self._begin_generation()

            prompt_kwargs = self._prompt_kwargs(full_prompt, negative_prompt, guidance)

            result = None
            with torch.inference_mode():
                result = self.pipe(
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    generator=generator,
                    **prompt_kwargs,
                    **self._pipeline_kwargs(guidance, steps),
                )

                image = getattr(result, "images", [None])[0] if result else None

            if result is not None:
                del result
            gc.collect()

            if self.stop_event.is_set():
                update_status("Interrupted by user.")
                GLib.idle_add(self.window.set_progress, 0.0, "Stopped")
            elif image is not None:
                timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
                output_path = IMAGE_DIR / f"animus_{timestamp}.png"
                info = self._png_metadata(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance=guidance_text,
                    sampler=sampler,
                    shift=shift_text,
                    seed=seed,
                    size=f"{width}x{height}",
                    **self._extra_metadata(),
                )
                image.save(str(output_path), format="PNG", pnginfo=info)

                GLib.idle_add(self._display_image, str(output_path))
                update_status(f"Done! Image saved to {output_path} (seed {seed}).")
                GLib.idle_add(self.window.set_progress, 1.0, "Finished")
            else:
                update_status("No image produced.")
                GLib.idle_add(self.window.set_progress, 0.0, "Failed")

        except KeyboardInterrupt:
            update_status("Interrupted by user.")
            GLib.idle_add(self.window.set_progress, 0.0, "Stopped")
            GLib.idle_add(self._reset_generate_button)
        except Exception as e:  # noqa: BLE001
            if self.stop_event.is_set():
                update_status("Interrupted by user.")
                GLib.idle_add(self.window.set_progress, 0.0, "Stopped")
            else:
                traceback.print_exc()
                update_status(f"Error generating image: {e}")
                GLib.idle_add(self.window.set_progress, 0.0, "Failed")
        finally:
            self._end_generation()
            self.generating = False
            GLib.idle_add(self._reset_generate_button)

    def _on_denoise_step(self, latents, step_index):
        if not self.generating or self.stop_event.is_set():
            return

        if step_index is not None and self._total_steps > 0:
            done = min(int(step_index) + 1, self._total_steps)
            GLib.idle_add(
                self.window.set_progress,
                done / self._total_steps,
                f"Step {done}/{self._total_steps}",
            )

        if latents is None or not self.preview_check.get_active():
            return

        try:
            data = self._preview_bytes(latents, step_index)
            if data is None:
                return
            rgb_bytes, width, height = data
            GLib.idle_add(self._show_preview, rgb_bytes, width, height)
        except Exception as e:  # noqa: BLE001
            print(f"Preview failed: {e}.", file=sys.stderr)

    def _preview_bytes(self, latents, step_index):
        return self._latents_to_rgb_bytes(latents)

    def _latents_to_rgb_bytes(self, latents):
        try:
            with torch.inference_mode():
                lat = latents.detach().to(dtype=torch.float32, device="cpu")
                if lat.dim() == 5:
                    lat = lat[0, :, 0]
                elif lat.dim() == 4:
                    lat = lat[0]
                elif lat.dim() != 3:
                    return None
                channels = self._latent_rgb_weight.shape[0]
                if lat.shape[0] < channels:
                    return None
                lat = lat[:channels]
                rgb = torch.einsum("chw,cr->hwr", lat, self._latent_rgb_weight)
                rgb = rgb + self._latent_rgb_bias
                rgb = ((rgb + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)
                height, width, _ = rgb.shape
                return rgb.contiguous().numpy().tobytes(), width, height
        except Exception as e:  # noqa: BLE001
            print(f"Preview projection failed: {e}.", file=sys.stderr)
            return None

    def _show_preview(self, rgb_bytes, width, height):
        if not self.generating:
            return False
        try:
            gbytes = GLib.Bytes.new(rgb_bytes)
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                gbytes, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width * 3
            )

            longest = max(width, height)
            if longest and longest != PREVIEW_DISPLAY_SIZE:
                scale = PREVIEW_DISPLAY_SIZE / longest
                pixbuf = pixbuf.scale_simple(
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                    GdkPixbuf.InterpType.BILINEAR,
                )

            self.image_display.set_from_pixbuf(pixbuf)

            if not self.preview_shown:
                self.preview_shown = True
                self.window.show_output(self)
        except Exception as e:  # noqa: BLE001
            print(f"Could not display preview: {e}.", file=sys.stderr)

        return False

    def _display_image(self, path):
        try:
            if not Path(path).exists():
                print(f"Error: Image file not found: {path}.", file=sys.stderr)
                self._clear_image_display()
                return False
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
            self.image_display.set_from_pixbuf(pixbuf)
            self.current_image_path = path
            self.delete_image_button.set_sensitive(True)
            self.window.show_output(self)
        except Exception as e:  # noqa: BLE001
            print(f"Error displaying image: {e}.", file=sys.stderr)
        return False

    def _clear_image_display(self):
        self.image_display.clear()
        self.current_image_path = None
        self.delete_image_button.set_sensitive(False)

    def on_delete_image_clicked(self, button):
        if self.current_image_path:
            try:
                if Path(self.current_image_path).exists():
                    os.remove(self.current_image_path)
                    print(f"Deleted image: {self.current_image_path}.")
                    update_status("Image deleted successfully.")
                else:
                    print(f"Image file no longer exists: {self.current_image_path}")
                    update_status("Image file no longer exists.")
                self._clear_image_display()
            except Exception as e:  # noqa: BLE001
                error_msg = f"Error deleting image: {e}."
                print(error_msg, file=sys.stderr)
                update_status(error_msg)

    def on_restore_defaults_clicked(self, button):
        self.model_entry.set_text(self.default_dit)
        self.preview_check.set_active(True)

        for i in range(NUM_LORA_SLOTS):
            self.lora_entries[i].set_text("")
            weight_name_entry, weight_spin = self.lora_weight_entries[i]
            weight_name_entry.set_text("")
            weight_spin.set_value(0.5)

        self.width_spin.set_value(self.default_size)
        self.height_spin.set_value(self.default_size)
        self.steps_spin.set_value(self.default_steps)
        self.guidance_spin.set_value(self.default_guidance)
        self.sampler_combo.set_active_id(self.default_sampler)
        self.shift_spin.set_value(self.default_shift)
        self.seed_spin.set_value(ANIMA_DEFAULT_SEED)

        self.trigger_text.get_buffer().set_text("")
        self.prompt_text.get_buffer().set_text("")
        self.neg_prompt_text.get_buffer().set_text("")

        update_status("Settings restored to defaults.")


class DiffusersPane(GeneratePane):
    converts_loras = False
    pipeline_cls = None
    transformer_cls = None
    components_repo = None
    components_prefix = ""
    component_repos: ClassVar[dict[str, tuple]] = {}
    gguf_repo = None
    gguf_preference = GGUF_QUANT_PREFERENCE
    gguf_prefer = ()
    gguf_avoid = ()
    distilled_tokens = ()
    text_encoder_dtype = torch.bfloat16
    text_encoder_name = "text encoder"
    unquantized_size = "a great deal of"
    cfg_threshold = 1.0
    cfg_trunc_kwarg = "cfg_trunc_ratio"
    default_cfg_trunc = 1.0
    default_sigmas = DEFAULT_SIGMAS
    supplies_explicit_sigmas = True

    def __init__(self, window):
        if (
            self.supplies_explicit_sigmas
            and "UniPC" in self.samplers
            and not _accepts_custom_sigmas(UniPCMultistepScheduler)
        ):
            self.samplers = tuple(s for s in self.samplers if s != "UniPC")
            print(
                "This diffusers' UniPC cannot take a custom sigma schedule, "
                f"which {self.model_name} always supplies. Leaving it out."
            )
        super().__init__(window)

    def _prepare_source(self, model_name):
        model_name = normalize_huggingface_url((model_name or "").strip())
        if not model_name:
            model_name = self.components_repo
        if model_name.startswith(UNQUANTIZED_PREFIX):
            return model_name
        if not model_name.lower().endswith(".gguf") and is_huggingface_repo(model_name):
            resolved = resolve_gguf_in_repo(
                model_name, self.gguf_preference, self.gguf_prefer, self.gguf_avoid
            )
            if resolved is not None:
                self._note_checkpoint(resolved.rsplit("/", 1)[-1], model_name)
                return resolved
        else:
            self._note_checkpoint(model_name.rsplit("/", 1)[-1], None)
        return model_name

    def _note_checkpoint(self, filename, repo):
        if repo is not None:
            update_status(f"Picked {filename} out of {repo}.")
        lowered = filename.lower()
        if not any(token in lowered for token in self.distilled_tokens):
            return
        update_status(
            f"{filename} looks like a distilled checkpoint. Those want very "
            "few steps and little or no guidance, so at the defaults here "
            "they will spend a long time producing a mess."
        )

    def _load_transformer(self, dit_source):
        if dit_source.lower().endswith(".gguf"):
            update_status(f"Loading the {self.model_name} DiT from {dit_source}...")
            with report_loader_complaints():
                return self.transformer_cls.from_single_file(
                    dit_source,
                    quantization_config=GGUFQuantizationConfig(
                        compute_dtype=torch.float32
                    ),
                    config=self.components_repo,
                    subfolder=self._subfolder("transformer"),
                    torch_dtype=torch.float32,
                )

        if dit_source.startswith(UNQUANTIZED_PREFIX):
            dit_source = dit_source[len(UNQUANTIZED_PREFIX) :] or self.components_repo

        dtype = torch.float32
        update_status(
            f"Loading an unquantized {self.model_name} DiT from {dit_source} "
            "in float32. Four bytes a parameter, so check it fits before "
            "waiting on it."
        )
        try:
            return self.transformer_cls.from_pretrained(
                dit_source,
                subfolder=self._subfolder("transformer"),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as e:  # noqa: BLE001
            gc.collect()
            print(f"No transformer subfolder in {dit_source} ({e}) so trying it bare.")
            return self.transformer_cls.from_pretrained(
                dit_source, torch_dtype=dtype, low_cpu_mem_usage=True
            )

    def _subfolder(self, name):
        _repo, prefix = self._component_source(name)
        return f"{prefix}/{name}" if prefix else name

    def _component_source(self, name):
        return self.component_repos.get(
            name, (self.components_repo, self.components_prefix)
        )

    def _component_repo(self, name):
        return self._component_source(name)[0]

    def _check_transformer_weights(self, transformer):
        empty, bad, total, skipped = [], [], 0, 0
        try:
            named = list(transformer.named_parameters())
        except Exception as e:  # noqa: BLE001
            print(f"Could not list the transformer weights: {e}.")
            return

        for name, param in named:
            total += 1
            try:
                data = getattr(param, "data", param)
                if data.numel() == 0:
                    continue
                if data.is_floating_point() and not torch.isfinite(data).all():
                    bad.append(name)
                    continue
                if bool((data == 0).all()):
                    empty.append(name)
            except Exception:  # noqa: BLE001
                skipped += 1

        if skipped:
            print(f"{skipped} of {total} weight tensors could not be inspected.")
        if not total:
            return
        if bad:
            update_status(
                f"{len(bad)} of {total} weight tensors are not finite, starting "
                f"with {bad[0]}. This checkpoint will not produce an image."
            )
        if len(empty) > total // 20:
            update_status(
                f"{len(empty)} of {total} weight tensors are entirely zero, "
                f"starting with {empty[0]}. This usually means the file was "
                "converted for another runtime and few of its keys matched, "
                "which produces noise rather than an error."
            )

    def _available(self):
        raise NotImplementedError

    def _load_pipeline(self, dit_source):
        self._available()

        from transformers import AutoTokenizer

        repo = self.components_repo
        transformer = self._load_transformer(dit_source)
        self._check_transformer_weights(transformer)
        if self._check_stop_loading():
            del transformer
            gc.collect()
            return None

        tokenizer_repo = self._component_repo("tokenizer")
        if tokenizer_repo != repo:
            update_status(f"Taking the tokenizer from {tokenizer_repo}.")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repo, subfolder=self._subfolder("tokenizer")
        )

        update_status(f"Loading the VAE and scheduler from {repo}...")
        vae = AutoencoderKL.from_pretrained(
            self._component_repo("vae"),
            subfolder=self._subfolder("vae"),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self._component_repo("scheduler"), subfolder=self._subfolder("scheduler")
        )

        if self._check_stop_loading():
            del transformer, vae
            gc.collect()
            return None

        pipe = self.pipeline_cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=None,
            tokenizer=tokenizer,
        )

        try:
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
                update_status("Enabled VAE tiling and slicing.")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: {e}.")

        try:
            pipe.to("cpu")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not move pipeline to CPU: {e}.")

        self._align_prompt_dtype(pipe)

        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._measure_token_budget(pipe, tokenizer)
            GLib.idle_add(self.on_text_changed, None)

        self._base_scheduler = pipe.scheduler
        self._prompt_cache = None
        self._adopt_scheduler_defaults(pipe.scheduler)

        gc.collect()
        return pipe

    def _adopt_scheduler_defaults(self, scheduler):
        try:
            config = dict(scheduler.config)
            chosen = spin_value(self.shift_spin)
        except Exception:  # noqa: BLE001
            return

        if config.get("use_dynamic_shifting"):
            update_status(
                f"The {self.model_name} scheduler resolves its own shift. "
                f"Shift is pinning it to {chosen:g} so the control works."
            )
            return

        try:
            shipped = float(config.get("shift", 1.0))
        except (TypeError, ValueError):
            return
        if abs(shipped - chosen) <= 1e-6:
            return

        if abs(chosen - float(self.default_shift)) <= 1e-6:
            GLib.idle_add(self._set_shift, shipped)
            update_status(
                f"The checkpoint asks for shift {shipped:g}. Taking it over "
                f"the default of {float(self.default_shift):g}."
            )
            return

        update_status(
            f"The checkpoint's scheduler asks for shift {shipped:g}, and "
            f"Shift is set to {chosen:g}. Higher values push the image into "
            "the last few steps, so previews stay noise for longer."
        )

    def _set_shift(self, value):
        self.shift_spin.set_value(value)
        return False

    def _load_text_encoder(self):
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            self._component_repo("text_encoder"),
            subfolder=self._subfolder("text_encoder"),
            torch_dtype=self.text_encoder_dtype,
            low_cpu_mem_usage=True,
        )

    def _release_text_encoder(self, pipe=None):
        pipe = pipe if pipe is not None else self.pipe
        if pipe is None or getattr(pipe, "text_encoder", None) is None:
            return
        pipe.text_encoder = None
        gc.collect()
        update_status(
            "Released the text encoder because the denoise loop has no use for it."
        )

    def _ensure_text_encoder(self):
        if getattr(self.pipe, "text_encoder", None) is not None:
            return
        update_status(
            f"Loading the {self.text_encoder_name} from "
            f"{self._component_repo('text_encoder')} to encode this prompt..."
        )
        self.pipe.text_encoder = self._load_text_encoder()
        self._check_vocabularies()

    def _check_vocabularies(self):
        if self._component_repo("tokenizer") == self._component_repo("text_encoder"):
            return
        try:
            rows = self.pipe.text_encoder.get_input_embeddings().weight.shape[0]
            tokens = len(self.pipe.tokenizer)
        except Exception as e:  # noqa: BLE001
            print(f"Could not compare tokenizer and text encoder: {e}.")
            return
        if tokens > rows:
            update_status(
                f"The tokenizer has {tokens} tokens but the text encoder only "
                f"has {rows} embedding rows. They are not the same vocabulary, "
                "and the prompt will not mean what it says."
            )
        else:
            print(f"Tokenizer {tokens} tokens, text encoder {rows} rows: fits.")

    def _measure_token_budget(self, pipe, tokenizer):
        return

    def _align_prompt_dtype(self, pipe):
        dtype = getattr(getattr(pipe, "transformer", None), "dtype", None)
        if dtype is None:
            dtype = torch.float32
        original = pipe.encode_prompt

        def encode_prompt(*args, **kwargs):
            return cast_floats(original(*args, **kwargs), dtype)

        pipe.encode_prompt = encode_prompt

    def _build_scheduler(self, sampler, base, shift):
        return build_flow_scheduler(
            sampler,
            base,
            shift,
            dynamic_shifting=False,
            sigmas=self.sigmas_combo.get_active_id() or self.default_sigmas,
        )

    def _set_scheduler(self, scheduler):
        self.pipe.scheduler = scheduler

    def _apply_guidance(self, guidance):
        return

    def _convert_lora(self, state_dict):
        return None

    def _begin_generation(self):
        return

    def _end_generation(self):
        return

    def _encode_prompts(self, prompt, negative_prompt, guidance):
        do_cfg = float(guidance) > self.cfg_threshold
        key = (prompt, negative_prompt, do_cfg, self.max_sequence_length)

        if self._prompt_cache is None or self._prompt_cache[0] != key:
            self._ensure_text_encoder()
            update_status("Encoding the prompt...")
            with torch.inference_mode():
                self._prompt_cache = (
                    key,
                    self._call_encode_prompt(prompt, negative_prompt, do_cfg),
                )

        self._release_text_encoder()
        return self._prompt_cache[1]

    def _pipeline_kwargs(self, guidance, steps):
        return {
            "guidance_scale": float(guidance),
            self.cfg_trunc_kwarg: spin_value(self.cfg_trunc_spin),
            "max_sequence_length": self.max_sequence_length,
            "callback_on_step_end": self._on_standard_step,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }

    def _on_standard_step(self, pipe, step_index, timestep, callback_kwargs):
        if self.stop_event.is_set():
            raise KeyboardInterrupt()
        self._on_denoise_step(callback_kwargs.get("latents"), step_index)
        return callback_kwargs

    def _preview_bytes(self, latents, step_index):
        every = int(spin_value(self.decode_every_spin))
        if every > 0 and step_index is not None and (int(step_index) + 1) % every == 0:
            decoded = self._decode_preview(latents)
            if decoded is not None:
                return decoded
        return self._latents_to_rgb_bytes(latents)

    def _decode_preview(self, latents):
        vae = getattr(self.pipe, "vae", None)
        processor = getattr(self.pipe, "image_processor", None)
        if vae is None or processor is None or latents is None:
            return None
        try:
            with torch.inference_mode():
                scaled = latents.detach().to(dtype=torch.float32)
                scaled = scaled / vae.config.scaling_factor + vae.config.shift_factor
                image = vae.decode(scaled, return_dict=False)[0]
                image = processor.postprocess(image, output_type="np")[0]
                rgb = (image * 255.0).clip(0, 255).astype(numpy.uint8)
                height, width, _ = rgb.shape
                return numpy.ascontiguousarray(rgb).tobytes(), width, height
        except Exception as e:  # noqa: BLE001
            print(f"Preview decode failed: {e}. Falling back to the projection.")
            return None
        finally:
            gc.collect()

    def _adjust_size(self, width, height):
        multiple = self.size_multiple
        adjusted = tuple(
            max(multiple, (int(value) // multiple) * multiple)
            for value in (width, height)
        )
        if adjusted != (width, height):
            update_status(
                f"Rounded the size to {adjusted[0]}x{adjusted[1]}. Latents pack "
                f"2x2, so both sides have to be a multiple of {multiple}."
            )
        return adjusted

    def _extra_controls(self, controls_box):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        label = Gtk.Label(label="CFG cutoff:")
        label.set_size_request(100, -1)
        label.set_xalign(0)
        box.pack_start(label, False, False, 0)

        self.cfg_trunc_spin = Gtk.SpinButton()
        self.cfg_trunc_spin.set_adjustment(
            Gtk.Adjustment(
                value=self.default_cfg_trunc,
                lower=0.0,
                upper=1.0,
                step_increment=0.05,
                page_increment=0.25,
            )
        )
        self.cfg_trunc_spin.set_digits(2)
        self.cfg_trunc_spin.set_size_request(100, -1)
        box.pack_start(self.cfg_trunc_spin, False, False, 0)
        controls_box.pack_start(box, False, False, 0)

        sigmas_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        sigmas_label = Gtk.Label(label="Sigmas:")
        sigmas_label.set_size_request(100, -1)
        sigmas_label.set_xalign(0)
        sigmas_box.pack_start(sigmas_label, False, False, 0)

        self.sigmas_combo = Gtk.ComboBoxText()
        for name in SIGMA_SCHEDULES:
            self.sigmas_combo.append(name, name)
        self.sigmas_combo.set_active_id(self.default_sigmas)
        sigmas_box.pack_start(self.sigmas_combo, False, False, 0)
        controls_box.pack_start(sigmas_box, False, False, 0)

        decode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        decode_label = Gtk.Label(label="Decode every:")
        decode_label.set_size_request(100, -1)
        decode_label.set_xalign(0)
        decode_box.pack_start(decode_label, False, False, 0)

        self.decode_every_spin = Gtk.SpinButton()
        self.decode_every_spin.set_adjustment(
            Gtk.Adjustment(
                value=DEFAULT_DECODE_EVERY,
                lower=0,
                upper=100,
                step_increment=1,
                page_increment=5,
            )
        )
        self.decode_every_spin.set_size_request(100, -1)
        decode_box.pack_start(self.decode_every_spin, False, False, 0)
        decode_box.pack_start(
            Gtk.Label(label="steps (0 for the rough preview only)"), False, False, 0
        )
        controls_box.pack_start(decode_box, False, False, 0)

    def _extra_metadata(self):
        return {
            self.cfg_trunc_kwarg: spin_text(self.cfg_trunc_spin),
            "sigmas": self.sigmas_combo.get_active_id() or self.default_sigmas,
        }

    def collect_settings(self):
        settings = super().collect_settings()
        settings["cfg_trunc"] = spin_value(self.cfg_trunc_spin)
        settings["decode_every"] = int(spin_value(self.decode_every_spin))
        settings["sigmas"] = self.sigmas_combo.get_active_id() or self.default_sigmas
        return settings

    def load_settings(self, settings):
        super().load_settings(settings)
        try:
            for key in ("cfg_trunc", "cfg_trunc_ratio"):
                if key in settings:
                    self.cfg_trunc_spin.set_value(float(settings[key]))
                    break
            if "decode_every" in settings:
                self.decode_every_spin.set_value(int(settings["decode_every"]))
            if settings.get("sigmas") in SIGMA_SCHEDULES:
                self.sigmas_combo.set_active_id(settings["sigmas"])
        except Exception as e:  # noqa: BLE001
            print(f"Error loading settings: {e}.")

    def on_restore_defaults_clicked(self, button):
        self.cfg_trunc_spin.set_value(self.default_cfg_trunc)
        self.decode_every_spin.set_value(DEFAULT_DECODE_EVERY)
        self.sigmas_combo.set_active_id(self.default_sigmas)
        super().on_restore_defaults_clicked(button)


class NativeZImagePane(DiffusersPane):
    pipeline_cls = NativeZImagePipeline
    transformer_cls = None
    text_encoder_name = "Qwen3 text encoder"
    unquantized_size = "roughly 24 gigabytes against about 4 for"
    model_label = "DiT (GGUF):"
    cfg_threshold = 0.0
    cfg_trunc_kwarg = "cfg_truncation"
    latent_rgb_factors = FLUX_LATENT_RGB_FACTORS
    latent_rgb_bias = FLUX_LATENT_RGB_BIAS
    size_multiple = ZIMAGE_SIZE_MULTIPLE
    samplers = ZIMAGE_SAMPLERS
    supplies_explicit_sigmas = False
    expects_distilled = True
    distilled_tokens = ("turbo", "distill", "lightning", "-step", "_step")

    def _available(self):
        if not VAE_AVAILABLE:
            raise ModelError(
                f"{self.model_name} needs diffusers for the VAE, the "
                "scheduler and the GGUF unpacking."
            )

    def _note_checkpoint(self, filename, repo):
        if repo is not None:
            update_status(f"Picked {filename} out of {repo}.")
        lowered = filename.lower()
        distilled = any(token in lowered for token in self.distilled_tokens)
        if distilled == self.expects_distilled:
            return
        steps = int(spin_value(self.steps_spin))
        if distilled:
            update_status(
                f"{filename} is a distilled checkpoint, but this tab is set "
                f"up for one that is not. Distilled Z-Image wants about "
                f"{ZIMAGE_DEFAULT_STEPS} steps and no guidance at all. At "
                f"{steps} steps it spends the rest making the image worse."
            )
        else:
            update_status(
                f"{filename} does not look distilled, but this tab is set up "
                f"for one that is. Undistilled Z-Image wants nearer fifty "
                f"steps and guidance around five, so at {steps} steps and "
                "this tab's guidance it will produce a smear."
            )

    def _fetch_dit(self, source):
        path = Path(source)
        try:
            if path.is_file():
                return path
        except OSError:
            pass
        if not is_direct_url(source):
            raise ModelError(
                f"{source} is neither a file on disk nor somewhere to "
                "download one from."
            )

        if "huggingface.co/" in source:
            tail = source.split("huggingface.co/", 1)[1].strip("/")
            parts = tail.split("/")
            if len(parts) >= 3:
                repo = "/".join(parts[:2])
                filename = "/".join(parts[2:])
                update_status(f"Fetching {filename} from {repo}...")
                from huggingface_hub import hf_hub_download

                return Path(hf_hub_download(repo_id=repo, filename=filename))

        destination = DIT_DIR / source.rsplit("/", 1)[-1]
        if destination.is_file():
            return destination

        def report(done, total):
            if total:
                GLib.idle_add(
                    self.window.set_progress,
                    done / total,
                    f"Downloading {_format_size(done)} of {_format_size(total)}",
                )

        update_status(f"Downloading {destination.name}...")
        download_model(
            source,
            destination,
            progress=report,
            stop_check=lambda: self.stop_event.is_set(),
        )
        return destination

    def _unquantized_files(self, source):
        if not is_huggingface_repo(source):
            raise ModelError(
                f"{source} is not a repo this can take an unquantized "
                "transformer out of."
            )
        update_status(
            f"Fetching an unquantized transformer from {source}. That is "
            f"{self.unquantized_size} a quantized one, so check it fits."
        )
        from huggingface_hub import snapshot_download

        local = (
            Path(snapshot_download(source, allow_patterns=["transformer/*"]))
            / "transformer"
        )
        shards = sorted(local.glob("*.safetensors"))
        if not shards:
            raise ModelError(f"{source} has no transformer/*.safetensors.")
        return shards

    def _selected_device(self):
        combo = getattr(self, "device_combo", None)
        chosen = combo.get_active_id() if combo is not None else None
        return chosen or ZIMAGE_CPU_DEVICE

    def _dit_files(self, dit_source):
        if dit_source.startswith(UNQUANTIZED_PREFIX):
            repo = dit_source[len(UNQUANTIZED_PREFIX) :] or self.components_repo
            return self._unquantized_files(repo)
        local = self._fetch_dit(dit_source)
        if local.suffix.lower() == ".gguf":
            try:
                import gguf  # noqa: F401
            except Exception as e:
                raise ModelError(
                    f"{local.name} is a GGUF file, which needs the gguf "
                    f"package to unpack ({e})."
                ) from e
        return [local]

    @staticmethod
    def _describe_config(config):
        update_status(
            f"{config['n_layers']} layers of {config['dim']} across "
            f"{config['n_heads']} heads, {config['n_refiner_layers']} refiner "
            f"layers, caption features {config['cap_feat_dim']} wide."
        )

    def _load_transformer(self, dit_source):
        files = self._dit_files(dit_source)
        device = self._selected_device()
        if device != ZIMAGE_CPU_DEVICE:
            return self._load_vulkan_transformer(files, device)

        state = {}
        for path in files:
            update_status(f"Reading the {self.model_name} DiT from {path.name}...")
            state.update(read_z_image_state_dict(path))

        if self._check_stop_loading():
            return None

        config = detect_z_image_config(state)
        self._describe_config(config)
        with report_loader_complaints():
            return build_z_image_transformer(state, config, progress=update_status)

    def _load_vulkan_transformer(self, files, device):
        try:
            gpu = animus_vulkan.get_device(device)
        except (animus_vulkan.VulkanUnavailable, animus_vulkan.VulkanError) as e:
            raise ModelError(f"Could not open {device}: {e}") from e
        update_status(f"Using {gpu.describe()}.")

        names = ", ".join(path.name for path in files)
        update_status(f"Reading the {self.model_name} DiT from {names}...")
        tensors = animus_vulkan.read_checkpoints(files)
        config = animus_vulkan.detect_config(tensors)
        self._describe_config(config)

        weights = sum(t.nbytes for t in tensors.values() if len(t.shape) > 1)
        free, total = gpu.budget()
        if weights > free * 0.85:
            update_status(
                f"The weights are {_format_size(weights)} and {gpu.name} has "
                f"about {_format_size(free)} free of {_format_size(total)}. "
                "Whatever does not fit stays in host memory and is read over "
                "the bus, which is slower. A smaller quantization would fit "
                "outright."
            )

        if self._check_stop_loading():
            return None

        try:
            return animus_vulkan.VulkanZImage(
                gpu,
                tensors,
                config,
                progress=update_status,
                stop_check=lambda: self.stop_event.is_set(),
            )
        except animus_vulkan.VulkanError as e:
            raise ModelError(str(e)) from e

    def _check_transformer_weights(self, transformer):
        return

    def _load_pipeline(self, dit_source):
        self._available()

        from transformers import AutoTokenizer

        repo = self.components_repo
        transformer = self._load_transformer(dit_source)
        if transformer is None or self._check_stop_loading():
            drop_transformer(transformer)
            del transformer
            gc.collect()
            return None

        tokenizer_repo = self._component_repo("tokenizer")
        if tokenizer_repo != repo:
            update_status(f"Taking the tokenizer from {tokenizer_repo}.")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repo, subfolder=self._subfolder("tokenizer")
        )

        update_status(f"Loading the VAE and scheduler from {repo}...")
        vae = AutoencoderKL.from_pretrained(
            self._component_repo("vae"),
            subfolder=self._subfolder("vae"),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self._component_repo("scheduler"), subfolder=self._subfolder("scheduler")
        )

        if self._check_stop_loading():
            drop_transformer(transformer)
            del transformer, vae
            gc.collect()
            return None

        pipe = self.pipeline_cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=None,
            tokenizer=tokenizer,
        )

        for name, enable in (
            ("slicing", "enable_slicing"),
            ("tiling", "enable_tiling"),
        ):
            try:
                if hasattr(vae, enable):
                    getattr(vae, enable)()
            except Exception as e:  # noqa: BLE001
                print(f"Warning: could not turn on VAE {name}: {e}.")

        if is_vulkan_transformer(transformer):
            try:
                pipe.vae = animus_vulkan.VulkanVAE(
                    transformer.device, vae, progress=update_status, fallback=vae
                )
            except animus_vulkan.VulkanError as e:
                update_status(f"The VAE decode stays on the CPU: {e}")

        self._align_prompt_dtype(pipe)

        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._measure_token_budget(pipe, tokenizer)
            GLib.idle_add(self.on_text_changed, None)

        self._base_scheduler = pipe.scheduler
        self._prompt_cache = None
        self._loaded_adapters = {}
        self._adopt_scheduler_defaults(pipe.scheduler)

        gc.collect()
        return pipe

    def _measure_token_budget(self, pipe, tokenizer):
        if not getattr(tokenizer, "chat_template", None):
            update_status(
                "This tokenizer has no chat template. Z-Image wraps every "
                "prompt in one before encoding, so without it the text "
                "encoder sees something it was never trained on and the "
                "image will show it. Point the tab at a repo whose tokenizer "
                "has one."
            )
            return

        try:
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False,
                add_generation_prompt=True,
            )
            overhead = len(tokenizer(wrapped, add_special_tokens=False).input_ids)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not measure the chat template: {e}.")
            return

        if overhead < QWEN_TEMPLATE_MIN_TOKENS:
            update_status(
                f"The chat template here is only {overhead} tokens, which is "
                "too short to be Qwen3's. The text encoder is being handed "
                "something it was not trained on, and the image will show it."
            )
            return

        limit = max(1, self.max_sequence_length - overhead)
        if limit == self.token_limit:
            return
        self.token_limit = limit
        update_status(
            f"Qwen3's chat template takes {overhead} of "
            f"{self.max_sequence_length} tokens, leaving {limit} for yours."
        )

    def _call_encode_prompt(self, prompt, negative_prompt, do_cfg):
        return self.pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            max_sequence_length=self.max_sequence_length,
        )

    def _prompt_kwargs(self, prompt, negative_prompt, guidance):
        embeds, negative = self._encode_prompts(prompt, negative_prompt, guidance)
        return {
            "prompt_embeds": embeds,
            "negative_prompt_embeds": negative or None,
        }

    def _pipeline_kwargs(self, guidance, steps):
        return {
            "guidance_scale": float(guidance),
            "cfg_truncation": spin_value(self.cfg_trunc_spin),
            "max_sequence_length": self.max_sequence_length,
            "callback_on_step_end": self._on_standard_step,
        }

    def _lora_state_dict(self, path, weight_name):
        local = self._lora_local_file(path, weight_name)
        if local is None:
            try:
                found = sorted(Path(path).glob("*.safetensors"))
            except OSError:
                found = []
            if found:
                local = found[0]
                if len(found) > 1:
                    update_status(
                        f"{path} holds {len(found)} LoRAs; taking "
                        f"{local.name}. Name one in Weight file to choose."
                    )
        if local is None and is_huggingface_repo(path):
            from huggingface_hub import hf_hub_download

            if weight_name:
                local = Path(hf_hub_download(repo_id=path, filename=weight_name))
            else:
                from huggingface_hub import list_repo_files

                names = [
                    name
                    for name in list_repo_files(path)
                    if name.endswith(".safetensors")
                ]
                if not names:
                    raise ModelError(f"{path} has no .safetensors in it.")
                local = Path(hf_hub_download(repo_id=path, filename=min(names)))
        if local is None:
            raise ModelError(f"Could not find a LoRA at {path}.")

        from safetensors.torch import load_file

        return load_file(str(local))

    def _load_loras(self):
        transformer = getattr(self.pipe, "transformer", None)
        if transformer is None:
            return
        clear_z_image_loras(transformer)
        self._loaded_adapters = {}

        for index, (entry, (weight_name_entry, weight_spin)) in enumerate(
            zip(self.lora_entries, self.lora_weight_entries)
        ):
            path = entry.get_text().strip()
            if not path:
                continue
            if self._check_stop_loading(cleanup_pipe=True):
                return

            weight_name = weight_name_entry.get_text().strip()
            label = f"{path}/{weight_name}" if weight_name else path
            adapter = f"lora_{index}"
            update_status(f"Loading LoRA {label}...")
            try:
                state = self._lora_state_dict(path, weight_name)
                applied, skipped = apply_z_image_lora(
                    transformer, state, adapter, spin_value(weight_spin)
                )
                del state
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                update_status(f"Warning: could not load LoRA {label}: {e}. Skipping.")
                continue

            if not applied:
                update_status(
                    f"Warning: none of {label} matched this DiT, so it is "
                    "doing nothing. It was probably trained for another model."
                )
                continue
            self._loaded_adapters[adapter] = index
            if skipped:
                update_status(
                    f"Applied {label} to {applied} modules. {len(skipped)} of "
                    "its entries matched nothing here."
                )
            else:
                update_status(f"Applied {label} to {applied} modules.")
        gc.collect()

    def _refresh_lora_weights(self):
        transformer = getattr(self.pipe, "transformer", None)
        if transformer is None or not self._loaded_adapters:
            return
        for adapter, slot in self._loaded_adapters.items():
            if slot >= len(self.lora_weight_entries):
                continue
            _, weight_spin = self.lora_weight_entries[slot]
            set_z_image_lora_scale(transformer, adapter, spin_value(weight_spin))

    def _extra_controls(self, controls_box):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        label = Gtk.Label(label="Device:")
        label.set_size_request(100, -1)
        label.set_xalign(0)
        box.pack_start(label, False, False, 0)

        self.device_combo = Gtk.ComboBoxText()
        self.device_combo.append(ZIMAGE_CPU_DEVICE, "CPU via torch")
        for ident, description, _kind in zimage_devices():
            self.device_combo.append(ident, description)
        self.device_combo.set_active_id(zimage_default_device())
        box.pack_start(self.device_combo, True, True, 0)
        controls_box.pack_start(box, False, False, 0)

        super()._extra_controls(controls_box)

    def _extra_metadata(self):
        metadata = super()._extra_metadata()
        metadata["device"] = self._selected_device()
        return metadata

    def collect_settings(self):
        settings = super().collect_settings()
        settings["device"] = self._selected_device()
        return settings

    def load_settings(self, settings):
        super().load_settings(settings)
        wanted = settings.get("device")
        if not isinstance(wanted, str) or not wanted:
            return
        if not self.device_combo.set_active_id(wanted):
            self.device_combo.set_active_id(zimage_default_device())
            update_status(
                f"The saved {self.model_name} device {wanted} is not here, so "
                f"the DiT will run on {self._selected_device()}."
            )

    def on_restore_defaults_clicked(self, button):
        self.device_combo.set_active_id(zimage_default_device())
        super().on_restore_defaults_clicked(button)


class ZImagePane(NativeZImagePane):
    name = "zimage"
    mode_label = "Z-Image"
    output_label = "Generated Image (Z-Image)"

    model_name = "Z-Image"
    browse_title = "Select a Z-Image GGUF (diffusion transformer)"
    count_tokenizer = ZIMAGE_COMPONENTS_REPO
    count_tokenizer_subfolder = "tokenizer"

    components_repo = ZIMAGE_COMPONENTS_REPO
    text_encoder_dtype = ZIMAGE_TEXT_ENCODER_DTYPE

    default_dit = ZIMAGE_DEFAULT_DIT
    default_steps = ZIMAGE_DEFAULT_STEPS
    default_guidance = ZIMAGE_DEFAULT_GUIDANCE
    default_size = ZIMAGE_DEFAULT_SIZE
    default_sampler = ZIMAGE_DEFAULT_SAMPLER
    default_shift = ZIMAGE_DEFAULT_SHIFT
    default_cfg_trunc = ZIMAGE_DEFAULT_CFG_TRUNC
    default_sigmas = ZIMAGE_DEFAULT_SIGMAS
    token_limit = ZIMAGE_TOKEN_LIMIT
    max_sequence_length = ZIMAGE_MAX_SEQUENCE_LENGTH


class UpscalePane:
    name = "upscale"
    mode_label = "Upscale"
    output_label = "Upscaled Frame"

    def __init__(self, window):
        self.window = window

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

    @property
    def busy(self):
        return self.working

    def build_controls(self):
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_size_request(-1, CONTROLS_HEIGHT)

        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        controls_box.set_border_width(10)
        scrolled.add(controls_box)

        controls_box.pack_start(self._build_source_area(), False, False, 0)
        controls_box.pack_start(self._build_output_row(), False, False, 0)
        controls_box.pack_start(self._build_model_rows(), False, False, 0)
        controls_box.pack_start(self._build_advanced(), False, False, 0)

        return scrolled

    def shutdown(self):
        if self.working:
            self.stop_event.set()
            terminate_process(self.decoder)
            terminate_process(self.encoder)

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=WORKER_THREAD_TIMEOUT)

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
            "A .pth or .safetensors SPAN / Real-ESRGAN / ESRGAN checkpoint"
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
        self.output_entry.set_text(str(VIDEO_DIR))
        grid.attach(self.output_entry, 1, row, 4, 1)

        output_browse = Gtk.Button(label="Browse...")
        output_browse.connect("clicked", self.on_browse_output_dir)
        grid.attach(output_browse, 5, row, 1, 1)

        return expander

    def _label(self, text):
        label = Gtk.Label(label=text)
        label.set_xalign(0)
        return label

    def build_output_page(self):
        preview_scrolled = Gtk.ScrolledWindow()
        preview_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.preview_image = Gtk.Image()
        preview_box.pack_start(self.preview_image, True, True, 0)

        self.preview_note = Gtk.Label(label="The newest upscaled frame appears here.")
        preview_box.pack_start(self.preview_note, False, False, 5)

        preview_scrolled.add(preview_box)
        return preview_scrolled

    def build_actions(self):
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

    def report_environment(self):
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

    def load_settings(self, settings):
        self._loading_settings = True
        try:
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
        except Exception as e:  # noqa: BLE001
            print(f"Error loading settings: {e}.")
        finally:
            self._loading_settings = False

        remembered = settings.get("user_set")
        if isinstance(remembered, list):
            automatic = self._auto_knobs()
            self._user_set.update(key for key in remembered if key in automatic)

        self.on_model_changed()
        self.on_encoder_changed()
        self.on_output_changed()
        self.on_device_changed()

    def collect_settings(self):
        return {
            "preset": self.preset_combo.get_active_id() or DEFAULT_PRESET,
            "out_width": int(self.out_width_spin.get_value()),
            "out_height": int(self.out_height_spin.get_value()),
            "model": self.model_combo.get_active_id() or DEFAULT_MODEL,
            "model_path": self.model_entry.get_text(),
            "device": self.device_combo.get_active_id() or "cpu",
            "threads": int(self.threads_spin.get_value()),
            "tile": int(self.tile_spin.get_value()),
            "tile_pad": int(self.tile_pad_spin.get_value()),
            "user_set": sorted(self._user_set),
            "channels_last": self.channels_last_check.get_active(),
            "deinterlace": self.deinterlace_check.get_active(),
            "compile": self.compile_check.get_active(),
            "precision": (self.precision_combo.get_active_id() or DEFAULT_PRECISION),
            "filters_pre": self.filters_pre_entry.get_text(),
            "filters_post": self.filters_post_entry.get_text(),
            "encoder": self.encoder_combo.get_active_id() or DEFAULT_ENCODER,
            "encoder_preset": (
                self.preset_encoder_combo.get_active_id() or DEFAULT_ENCODER_PRESET
            ),
            "crf": int(self.crf_spin.get_value()),
            "container": (self.container_combo.get_active_id() or DEFAULT_CONTAINER),
            "audio": self.audio_combo.get_active_id() or DEFAULT_AUDIO,
            "start": spin_value(self.start_spin),
            "limit": spin_value(self.limit_spin),
            "output_dir": self.output_entry.get_text(),
        }

    def on_source_entry_activate(self, entry):
        text = entry.get_text().strip()
        if text:
            self.set_source(text)

    def on_browse_source(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select a video",
            parent=self.window,
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
        except Exception as e:  # noqa: BLE001
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

        local = UPSCALER_DIR / model
        if local.exists():
            self.model_note.set_text(
                f"{_format_size(local.stat().st_size)} in {UPSCALER_DIR}"
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
            parent=self.window,
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
            parent=self.window,
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
        update_status("Settings restored to defaults.")

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
        self.output_entry.set_text(str(VIDEO_DIR))

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
                update_status("Choose a weights file first.")
                return None
            model_path = Path(weights).expanduser()
            model_url = None
        else:
            model_path = UPSCALER_DIR / model_id
            model_url = next(u for _label, f, u in BUILTIN_MODELS if f == model_id)

        encoder = self.encoder_combo.get_active_id() or DEFAULT_ENCODER
        container = self.container_combo.get_active_id() or DEFAULT_CONTAINER
        output_dir = Path(
            self.output_entry.get_text().strip() or str(VIDEO_DIR)
        ).expanduser()
        stem = Path(info["path"]).stem
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        dest = output_dir / (
            f"{stem}_{out_width}x{out_height}_{timestamp}."
            f"{container_for(encoder, container)}"
        )

        device = self.device_combo.get_active_id() or "cpu"
        precision = self.precision_combo.get_active_id() or DEFAULT_PRECISION
        if device.startswith("ncnn:") and precision == "bfloat16":
            print(
                "ncnn's Vulkan path has no bfloat16 here. Using float16, "
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
        if self.working or self.source is None or not self.window.claim(self):
            return

        job = self._collect_job()
        if job is None:
            return

        self.window.save_settings()

        self.working = True
        self.stop_event.clear()
        self.start_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.delete_button.set_sensitive(False)
        self.window.set_progress(0.0, "Starting...")

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
                update_status(f"Deleted {target}.")
            else:
                update_status(f"{target} is already gone.")
        except OSError as e:
            update_status(f"Could not delete {target}: {e}")
            return
        self.current_output = None
        self.delete_button.set_sensitive(False)

    def on_stop_clicked(self, button):
        if not self.working:
            return
        self.stop_event.set()
        update_status(
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
            except Exception:  # noqa: BLE001, S110
                pass
            finally:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001, S110
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
            return max(round(span * info["fps"]), 0)
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
                update_status(f"Downloading {model_path.name}...")
                download_model(
                    job["model_url"],
                    model_path,
                    progress=self._download_progress,
                    stop_check=self.stop_event.is_set,
                )
                update_status(f"Saved {model_path}.")
                terms = MODEL_LICENSES.get(model_path.name)
                if terms:
                    update_status(f"{model_path.name} is under {terms}.")

            update_status(f"Loading {model_path.name}...")
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
                update_status(f"{description}.")
            else:
                update_status(
                    f"{description}, running on {job['device']} in "
                    f"{str(torch_dtype).replace('torch.', '')}."
                )

            if job["compile"] and not on_ncnn:
                update_status(
                    "Compiling with torch.compile. The first few frames pay "
                    "for it so the rest should be faster."
                )
                try:
                    model = torch.compile(model)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"torch.compile is not usable here ({e}). Carrying on "
                        "without it."
                    )

            tile = job["tile"]
            tile_pad = job["tile_pad"]
            if tile > 0 and tile_pad < min_overlap:
                if min_overlap * 2 <= tile:
                    tile_pad = min_overlap
                    update_status(
                        f"Raised the tile overlap to {tile_pad} px, the radius "
                        "this network actually reads. Tiled output now matches "
                        "whole-frame output exactly."
                    )
                else:
                    update_status(
                        f"Note: this network reads {min_overlap} px around every "
                        f"output pixel but the tiles only carry {tile_pad} px of "
                        "context, so tile seams may show and shimmer between "
                        "frames. Set Tile to 0 to process whole frames, or raise "
                        "Tile and Overlap if there is memory for it."
                    )

            if tile > 0:
                update_status(
                    f"Working in {tile} px tiles with {tile_pad} px of overlap."
                )
                if on_ncnn:
                    update_status(
                        "On a GPU that is usually the wrong trade. Every tile "
                        "is a separate upload, dispatch and download, and the "
                        f"{tile_pad} px of overlap is computed twice along "
                        "every seam. Set Tile to 0 unless the card runs out of "
                        "memory."
                    )
            else:
                update_status("Working on whole frames.")

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
                    update_status(
                        f"'{filters}' resizes the source to "
                        f"{source_width}x{source_height} so upscaling from that."
                    )

            model_width = source_width * native_scale
            model_height = source_height * native_scale
            out_width, out_height = job["out_width"], job["out_height"]
            if (out_width, out_height) != (model_width, model_height):
                update_status(
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
                    update_status(
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

            self.current_output = job["dest"]

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
            inbound = grow_pipe(self.decoder.stdout)
            outbound = grow_pipe(self.encoder.stdin)
            if inbound or outbound:
                print(
                    f"Pipe buffers: {_format_size(inbound)} in from the decoder, "
                    f"{_format_size(outbound)} out to the encoder."
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

            profile = new_profile()
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
                    profile=profile,
                )
            except BrokenPipeError:
                raise RuntimeError(
                    "FFmpeg closed the pipe:\n" + "".join(encoder_errors).strip()
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
                    f"FFmpeg exited with {encoder_status} while encoding:\n"
                    + "".join(encoder_errors).strip()
                )
            if not stopped and decoder_status not in (0, -13, 255):
                print(f"Note: the decoder exited with {decoder_status}.")

            elapsed = time.monotonic() - started
            size = job["dest"].stat().st_size if job["dest"].exists() else 0
            rate = frames_done / elapsed if elapsed > 0 else 0.0

            breakdown = describe_profile(profile, frames_done)
            if breakdown:
                print(breakdown)
            if stopped:
                update_status(
                    f"Stopped after {frames_done} frames. {job['dest']} holds "
                    f"what was finished ({_format_size(size)})."
                )
                GLib.idle_add(self._set_progress, 0.0, "Stopped")
            else:
                update_status(
                    f"Done. {frames_done} frames in {_format_duration(elapsed)} "
                    f"({rate:.2f} fps), {_format_size(size)} written to "
                    f"{job['dest']}."
                )
                GLib.idle_add(self._set_progress, 1.0, "Finished")

        except KeyboardInterrupt:
            update_status("Stopped.")
            GLib.idle_add(self._set_progress, 0.0, "Stopped")
        except Exception as e:  # noqa: BLE001
            if self.stop_event.is_set():
                update_status("Stopped.")
            else:
                traceback.print_exc()
                update_status(f"Error: {e}")
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
        except Exception as e:  # noqa: BLE001
            print(f"Preview failed: {e}.", file=sys.stderr)
            return None

    def _set_progress(self, fraction, text):
        return self.window.set_progress(fraction, text)

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
        except Exception as e:  # noqa: BLE001
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


def _ncnn_probe_models():
    torch.manual_seed(0)
    yield "compact", SRVGGNetCompact(num_feat=16, num_conv=3, upscale=2)
    for norm in (False, True):
        model = SPAN(feature_channels=16, num_block=2, upscale=2, norm=norm)
        for parameter in model.parameters():
            nn.init.normal_(parameter, std=0.05)
        yield f"span norm={str(norm).lower()}", model


def _self_test_ncnn(check):
    try:
        import ncnn  # noqa: F401
    except Exception:  # noqa: BLE001
        return

    import tempfile

    for architecture, reference in _ncnn_probe_models():
        _self_test_ncnn_model(check, architecture, reference, tempfile)


def _self_test_ncnn_model(check, architecture, reference, tempfile):
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    frame = torch.rand(1, 3, 24, 32)

    with tempfile.TemporaryDirectory(prefix="animus-ncnn-") as workdir:
        weights = Path(workdir) / "probe.pth"
        torch.save({"params": reference.state_dict()}, weights)
        if hasattr(reference, "fuse"):
            reference.fuse()
        with torch.inference_mode():
            want = reference(frame)

        targets = [(None, "CPU")]
        targets += [
            (int(ident.partition(":")[2]), label)
            for ident, label, _kind in ncnn_devices()
        ]

        for gpu, label in targets:
            label = f"{architecture} {label}"
            for fp16 in (False,) if gpu is None else (False, True):
                try:
                    model, _scale, _multiple, _overlap, _d = load_ncnn_upscaler(
                        weights, gpu=gpu, threads=2, fp16=fp16
                    )
                    got = model(frame)
                except Exception as e:  # noqa: BLE001
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
                spread = max(1.0, want.abs().max().item())
                limit = (6e-2 if fp16 else 1e-4) * spread
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
    center = size // 2
    center -= center % max(size_multiple, 1)

    base = torch.rand(1, 3, size, size)
    probed = base.clone()
    probed[0, :, center, center] += 10.0

    with torch.inference_mode():
        difference = (model(probed) - model(base)).abs().sum(dim=1)[0] != 0

    rows = torch.nonzero(difference.any(dim=1)).flatten()
    cols = torch.nonzero(difference.any(dim=0)).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None, False

    top, bottom = int(rows[0]) // scale, int(rows[-1]) // scale
    left, right = int(cols[0]) // scale, int(cols[-1]) // scale
    reach = max(center - top, bottom - center, center - left, right - center)
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
        except Exception as e:  # noqa: BLE001
            print(f"Could not read {target}: {e}\nFalling back to 640x480.\n")

    installed = [
        (label, UPSCALER_DIR / filename)
        for label, filename, _url in BUILTIN_MODELS
        if (UPSCALER_DIR / filename).exists()
    ]
    if not installed:
        print(
            f"No weights in {UPSCALER_DIR} yet. Run one model from the window "
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

        probe = frame if device == "cpu" else frame.contiguous()

        def once():
            return upscale_frame(model, probe, scale, tile, overlap, multiple)

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
            except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
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


def _self_test_schedulers(check):
    if not ANIMA_AVAILABLE:
        check("Schedulers: diffusers", False, "not installed")
        return

    base = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    for sampler in FLOW_SAMPLERS:
        for spacing in SIGMA_SCHEDULES:
            label = f"Schedulers: {sampler} + {spacing}"
            try:
                scheduler = build_flow_scheduler(sampler, base, 3.0, sigmas=spacing)
                scheduler.set_timesteps(8)
                sigmas = scheduler.sigmas
                timesteps = scheduler.timesteps
            except Exception as e:  # noqa: BLE001
                check(label, False, str(e))
                continue
            usable = (
                bool(torch.isfinite(sigmas).all())
                and float(sigmas.max()) <= 1.001
                and len(timesteps) == 8
            )
            check(label, usable, f"sigma max {float(sigmas.max()):.3f}")


def _self_test_zimage(check):
    if not VAE_AVAILABLE:
        check("Z-Image: diffusers", False, "not installed")
        return

    torch.manual_seed(0)
    shape = {
        "in_channels": 4,
        "dim": 256,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "cap_feat_dim": 24,
    }
    try:
        reference = ZImageTransformer(**shape).eval()
    except Exception as e:  # noqa: BLE001
        check("Z-Image: builds", False, str(e))
        return
    check(
        "Z-Image: builds",
        True,
        f"{sum(p.numel() for p in reference.parameters())} parameters",
    )

    state = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    try:
        detected = detect_z_image_config(state)
    except Exception as e:  # noqa: BLE001
        check("Z-Image: shape detected", False, str(e))
        return
    wanted = {key: shape[key] for key in shape}
    check(
        "Z-Image: shape detected",
        all(detected.get(key) == value for key, value in wanted.items()),
        ", ".join(f"{k}={detected.get(k)}" for k in ("dim", "n_layers", "n_heads")),
    )

    try:
        rebuilt = build_z_image_transformer(state, detected)
    except Exception as e:  # noqa: BLE001
        check("Z-Image: loads", False, str(e))
        return
    check("Z-Image: loads", True, "weights assigned without a copy")

    caption = torch.randn(13, 24)
    latent = torch.randn(4, 1, 12, 20)
    timestep = torch.tensor([0.42])
    try:
        with torch.inference_mode():
            want = reference(latent, timestep, caption)
            got = rebuilt(latent, timestep, caption)
    except Exception as e:  # noqa: BLE001
        check("Z-Image: forward", False, str(e))
        return

    check(
        "Z-Image: output shape",
        tuple(got.shape) == tuple(latent.shape),
        f"got {tuple(got.shape)}",
    )
    check("Z-Image: finite", bool(torch.isfinite(got).all()))
    difference = (want - got).abs().max().item()
    check(
        "Z-Image: reload matches", difference < 1e-6, f"max difference {difference:.2e}"
    )

    flattened = {
        name: (tensor.reshape(-1) if 1 in tuple(tensor.shape) else tensor)
        for name, tensor in reference.state_dict().items()
    }
    try:
        flat_model = build_z_image_transformer(flattened, detected)
        with torch.inference_mode():
            flat_out = flat_model(latent, timestep, caption)
        check(
            "Z-Image: flattened pad tokens",
            tuple(flat_model.cap_pad_token.shape) == (1, detected["dim"])
            and (want - flat_out).abs().max().item() < 1e-6,
        )
    except Exception as e:  # noqa: BLE001
        check("Z-Image: flattened pad tokens", False, str(e))

    sdcpp = {}
    attentions = sorted(
        name[: -len(".attention.to_q.weight")]
        for name in state
        if name.endswith(".attention.to_q.weight")
    )
    for name, tensor in state.items():
        if any(name.startswith(f"{block}.attention.to_") for block in attentions):
            continue
        sdcpp[
            name.replace("all_x_embedder.2-1.", "x_embedder.").replace(
                "all_final_layer.2-1.", "final_layer."
            )
        ] = tensor
    for block in attentions:
        sdcpp[f"{block}.attention.qkv.weight"] = torch.cat(
            [state[f"{block}.attention.to_{part}.weight"] for part in "qkv"], dim=0
        )
        sdcpp[f"{block}.attention.out.weight"] = state[
            f"{block}.attention.to_out.0.weight"
        ]
        for part in ("q", "k"):
            sdcpp[f"{block}.attention.{part}_norm.weight"] = state[
                f"{block}.attention.norm_{part}.weight"
            ]
    try:
        converted = {normalize_z_image_key(k): v for k, v in sdcpp.items()}
        converted = split_fused_attention(converted)
        sdcpp_model = build_z_image_transformer(
            converted, detect_z_image_config(converted)
        )
        with torch.inference_mode():
            sdcpp_out = sdcpp_model(latent, timestep, caption)
        check(
            "Z-Image: fused qkv and sd.cpp names",
            (want - sdcpp_out).abs().max().item() < 1e-6,
            f"{len(attentions)} attentions split",
        )
    except Exception as e:  # noqa: BLE001
        check("Z-Image: fused qkv and sd.cpp names", False, str(e))

    wrong = dict(reference.state_dict())
    wrong["cap_embedder.1.weight"] = torch.zeros(detected["dim"], 48)
    try:
        build_z_image_transformer(wrong, detected)
        check("Z-Image: wrong shape refused", False, "it was accepted")
    except ModelError:
        check("Z-Image: wrong shape refused", True)

    try:
        shared = ZImageTransformer(
            in_channels=4,
            dim=256,
            n_layers=1,
            n_refiner_layers=1,
            n_heads=2,
            n_kv_heads=1,
            cap_feat_dim=24,
        ).eval()
        with torch.inference_mode():
            grouped = shared(latent, timestep, caption)
        check(
            "Z-Image: grouped-query attention",
            tuple(grouped.shape) == tuple(latent.shape)
            and bool(torch.isfinite(grouped).all()),
        )
    except Exception as e:  # noqa: BLE001
        check("Z-Image: grouped-query attention", False, str(e))

    target = "layers.0.attention.to_q"
    lora = {
        f"transformer.{target}.lora_A.weight": torch.randn(8, 256) * 0.05,
        f"transformer.{target}.lora_B.weight": torch.randn(256, 8) * 0.05,
    }
    applied, skipped = apply_z_image_lora(rebuilt, lora, "self_test", 1.0)
    with torch.inference_mode():
        changed = rebuilt(latent, timestep, caption)
    check(
        "Z-Image: LoRA applies",
        applied == 1 and not skipped,
        f"{applied} modules, {len(skipped)} unmatched",
    )
    check("Z-Image: LoRA changes the output", (changed - got).abs().max().item() > 1e-4)
    clear_z_image_loras(rebuilt)
    with torch.inference_mode():
        restored = rebuilt(latent, timestep, caption)
    check("Z-Image: LoRA clears", (restored - got).abs().max().item() < 1e-9)


def _self_test_vulkan(check):
    stale = animus_vulkan.stale_shaders()
    check(
        "Vulkan: shipped SPIR-V matches the GLSL",
        not stale,
        f"{len(stale)} stale, starting with {stale[0]}" if stale else "all current",
    )
    if not animus_vulkan.vulkan_devices():
        print("note  no Vulkan device, so the GPU backend was not tested")
        return

    device = animus_vulkan.self_test(check)
    if device is None:
        return

    torch.manual_seed(0)
    caption = torch.randn(13, 24)
    latent = torch.randn(4, 1, 12, 20)
    timestep = torch.tensor([0.42])
    base = {
        "in_channels": 4,
        "dim": 256,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "cap_feat_dim": 24,
    }
    grouped = dict(base, n_layers=1, n_kv_heads=1)

    for label, shape, quantize, tolerance in (
        ("float16", base, None, 1e-3),
        ("Q8_0", base, "Q8_0", 5e-3),
        ("grouped-query", grouped, None, 1e-3),
    ):
        stage = f"Vulkan: DiT with {label} weights matches torch"
        model = None
        try:
            torch.manual_seed(0)
            reference = ZImageTransformer(**shape).eval()
            with torch.no_grad():
                for name, parameter in reference.named_parameters():
                    if parameter.dim() == 1 and "bias" not in name:
                        parameter.add_(torch.randn_like(parameter) * 0.1)
            arrays = {
                name: tensor.detach().to(torch.float32).numpy()
                for name, tensor in reference.state_dict().items()
            }
            tensors = animus_vulkan.tensors_from_arrays(arrays, quantize)
            rounded = {
                name: torch.from_numpy(tensor.float32().copy())
                for name, tensor in tensors.items()
            }
            expected = build_z_image_transformer(
                rounded, detect_z_image_config(rounded)
            )
            with torch.inference_mode():
                want = expected(latent, timestep, caption)
            model = animus_vulkan.VulkanZImage(device, tensors)
            got = model(latent, timestep, caption)
            error = (got - want).abs().max().item() / max(1.0, want.abs().max().item())
            check(
                stage,
                tuple(got.shape) == tuple(want.shape) and error < tolerance,
                f"max relative error {error:.2e}",
            )
            if quantize is None and shape is base:
                stage = "Vulkan: LoRA with alpha matches torch"
                down = torch.randn(8, 256) * 0.05
                up = torch.randn(256, 8) * 0.05
                lora = {
                    "transformer.layers.0.attention.to_q.lora_A.weight": down,
                    "transformer.layers.0.attention.to_q.lora_B.weight": up,
                    "transformer.layers.0.attention.to_q.alpha": torch.tensor(4.0),
                }
                apply_z_image_lora(expected, lora, "self_test", 0.7)
                set_z_image_lora_scale(expected, "self_test", 0.9)
                applied, skipped = apply_z_image_lora(model, lora, "self_test", 0.7)
                set_z_image_lora_scale(model, "self_test", 0.9)
                with torch.inference_mode():
                    want_lora = expected(latent, timestep, caption)
                got_lora = model(latent, timestep, caption)
                error = (got_lora - want_lora).abs().max().item() / max(
                    1.0, want_lora.abs().max().item()
                )
                moved = (got_lora - got).abs().max().item()
                check(
                    stage,
                    applied == 1 and not skipped and error < tolerance and moved > 1e-4,
                    f"{applied} applied, max relative error {error:.2e}",
                )
                clear_z_image_loras(model)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            check(stage, False, str(e))
        finally:
            if model is not None:
                model.release()

    _self_test_vulkan_vae(check, device)


def _self_test_vulkan_vae(check, device):
    if not VAE_AVAILABLE:
        return
    stage = "Vulkan: VAE decode matches diffusers"
    gpu_vae = None
    try:
        torch.manual_seed(0)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(32, 64, 64, 64),
            layers_per_block=2,
            latent_channels=4,
            norm_num_groups=32,
            mid_block_add_attention=True,
            use_quant_conv=False,
            use_post_quant_conv=False,
        ).eval()
        with torch.no_grad():
            for _name, parameter in vae.named_parameters():
                if parameter.dim() == 1:
                    parameter.add_(torch.randn_like(parameter) * 0.2)
                else:
                    parameter.copy_(parameter.to(torch.float16).float())
        gpu_vae = animus_vulkan.VulkanVAE(device, vae)
        latent = torch.randn(1, 4, 6, 10)
        with torch.inference_mode():
            want = vae.decode(latent, return_dict=False)[0]
        got = gpu_vae.decode(latent, return_dict=False)[0]
        error = (got - want).abs().max().item() / max(1.0, want.abs().max().item())
        check(
            stage,
            tuple(got.shape) == tuple(want.shape) and error < 1e-3,
            f"max relative error {error:.2e}",
        )

        stage = "Vulkan: tiled VAE decode matches diffusers"
        vae.tile_latent_min_size, vae.tile_sample_min_size = 8, 64
        vae.enable_tiling()
        gpu_vae.tile_latent_min_size = 8
        gpu_vae.enable_tiling()
        latent = torch.randn(1, 4, 12, 20)
        with torch.inference_mode():
            want = vae.decode(latent, return_dict=False)[0]
        got = gpu_vae.decode(latent, return_dict=False)[0]
        error = (got - want).abs().max().item() / max(1.0, want.abs().max().item())
        check(
            stage,
            tuple(got.shape) == tuple(want.shape) and error < 1e-3,
            f"max relative error {error:.2e}",
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(stage, False, str(e))
    finally:
        if gpu_vae is not None:
            gpu_vae.release()


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
        (
            "SPAN x2",
            SPAN(feature_channels=16, num_block=2, upscale=2, norm=False),
            2,
            1,
        ),
        (
            "SPAN x4 normalized",
            SPAN(feature_channels=16, num_block=3, upscale=4, norm=True),
            4,
            1,
        ),
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
        except Exception as e:  # noqa: BLE001
            check(f"{label}: detected", False, str(e))
            continue

        check(f"{label}: architecture", type(model) is type(reference), description)
        check(f"{label}: scale", scale == want_scale, f"got {scale}")
        check(f"{label}: size multiple", multiple == want_multiple, f"got {multiple}")

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:  # noqa: BLE001
            check(f"{label}: strict load", False, str(e))
            continue
        check(f"{label}: strict load", True, f"overlap {overlap} px")

        model.eval()
        if hasattr(model, "fuse"):
            model.fuse()
            reference.fuse()
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
        spread = max(1.0, whole.abs().max().item())
        check(
            f"{label}: tiling with full overlap matches whole frames",
            delta < 1e-4 * spread,
            f"max |diff| = {delta:.2e}",
        )

        reach, clipped = measure_reach(model, scale, multiple, overlap)
        if reach is None:
            check(f"{label}: reach measurable", False, "the probe did not propagate")
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

    builtin = {filename for _label, filename, _url in BUILTIN_MODELS}
    check(
        "the default model is one of the built-in ones",
        DEFAULT_MODEL in builtin,
        DEFAULT_MODEL,
    )
    check(
        "the default model is under a license that restricts nobody",
        MODEL_LICENSES.get(DEFAULT_MODEL) in PERMISSIVE_LICENSES,
        MODEL_LICENSES.get(DEFAULT_MODEL, "unrecorded"),
    )
    check(
        "every model offered has its license recorded",
        builtin <= set(MODEL_LICENSES),
        ", ".join(sorted(builtin - set(MODEL_LICENSES))) or "all recorded",
    )
    check(
        "no license is recorded for a model that is not offered",
        set(MODEL_LICENSES) <= builtin,
        ", ".join(sorted(set(MODEL_LICENSES) - builtin)) or "none stale",
    )

    _self_test_schedulers(check)

    _self_test_zimage(check)

    _self_test_vulkan(check)

    _self_test_ncnn(check)

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        _self_test_pipeline(check)
    else:
        print(
            "note  ffmpeg and ffprobe are not in the PATH, so the video pipeline "
            "was not tested"
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


def _add_css(css, description):
    screen = Gdk.Screen.get_default()
    if screen is None:
        return

    try:
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            screen,
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Could not apply the {description} style: {e}.")


def _stored_settings():
    settings = {}

    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                settings = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Error loading settings: {e}.")
        settings = {}

    if not isinstance(settings, dict):
        return {}

    if not isinstance(settings.get(GeneratePane.name), dict) and not isinstance(
        settings.get(UpscalePane.name), dict
    ):
        settings = {GeneratePane.name: settings}

    if not isinstance(settings.get(UpscalePane.name), dict):
        try:
            if LEGACY_UPSCALE_FILE.exists():
                with open(LEGACY_UPSCALE_FILE, "r") as f:
                    stored = json.load(f)
                if isinstance(stored, dict):
                    settings[UpscalePane.name] = stored
                    print(f"Carried the settings over from {LEGACY_UPSCALE_FILE}.")
        except (OSError, ValueError) as e:
            print(f"Warning: could not read {LEGACY_UPSCALE_FILE}: {e}.")

    return settings


class AnimusWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title=WINDOW_TITLE)
        warnings.filterwarnings("ignore")
        self.set_wmclass("Animus", "Animus")
        self.set_default_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.set_border_width(10)

        _add_css(CONSOLE_CSS, "console font")
        _add_css(PROGRESS_CSS, "progress bar")

        for directory in (
            CONFIG_DIR,
            IMAGE_DIR,
            LORA_DIR,
            DIT_DIR,
            VIDEO_DIR,
            UPSCALER_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.panes = []
        self._output_pages = {}
        self._settings = {}
        self._remembered_mode = None

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(main_box)

        self.mode_notebook = Gtk.Notebook()
        main_box.pack_start(self.mode_notebook, False, False, 0)

        self.output_notebook = Gtk.Notebook()
        main_box.pack_start(self.output_notebook, True, True, 0)

        console_scrolled = Gtk.ScrolledWindow()
        console_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.console_text = Gtk.TextView()
        self.console_text.set_editable(False)
        self.console_text.set_wrap_mode(Gtk.WrapMode.CHAR)
        self.console_text.set_cursor_visible(False)
        self.console_text.get_style_context().add_class(CONSOLE_CSS_CLASS)
        console_scrolled.add(self.console_text)

        self.console_page = self.output_notebook.append_page(
            console_scrolled, Gtk.Label(label="Console Output")
        )

        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(True)
        self.progress.set_text("Idle")
        main_box.pack_start(self.progress, False, False, 0)

        self.action_stack = Gtk.Stack()
        main_box.pack_start(self.action_stack, False, False, 0)

        for pane in (
            GeneratePane(self),
            ZImagePane(self),
            UpscalePane(self),
        ):
            self._add_pane(pane)

        self.generate, self.zimage, self.upscale = self.panes

        self.output_notebook.set_current_page(self.console_page)
        self.mode_notebook.connect("switch-page", self.on_mode_switched)

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text, self.original_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.original_stderr)

        self.upscale.report_environment()
        self.load_settings()

    def _add_pane(self, pane):
        self.mode_notebook.append_page(
            pane.build_controls(), Gtk.Label(label=pane.mode_label)
        )
        self._output_pages[pane.name] = self.output_notebook.append_page(
            pane.build_output_page(), Gtk.Label(label=pane.output_label)
        )
        self.action_stack.add_named(pane.build_actions(), pane.name)
        self.panes.append(pane)

    def current_pane(self):
        index = self.mode_notebook.get_current_page()
        if 0 <= index < len(self.panes):
            return self.panes[index]
        return self.panes[0]

    def select_mode(self, pane):
        self.mode_notebook.set_current_page(self.panes.index(pane))

    def on_mode_switched(self, notebook, page, index):
        if not 0 <= index < len(self.panes):
            return

        pane = self.panes[index]
        self.action_stack.set_visible_child_name(pane.name)

        shown = self.output_notebook.get_current_page()
        if shown not in (self.console_page, self._output_pages[pane.name]):
            self.output_notebook.set_current_page(self.console_page)

    def show_output(self, pane):
        self.output_notebook.set_current_page(self._output_pages[pane.name])
        return False

    def set_progress(self, fraction, text):
        self.progress.set_fraction(max(0.0, min(1.0, fraction)))
        self.progress.set_text(text)
        return False

    def release_other_models(self, pane):
        released = []
        for other in self.panes:
            if other is pane or not hasattr(other, "release_model"):
                continue
            if other.release_model():
                released.append(other.model_name)
        return released

    def claim(self, pane):
        for other in self.panes:
            if other is not pane and other.busy:
                update_status(
                    f"The {other.mode_label} tab is still working. Stop it "
                    "first, or wait for it to finish."
                )
                return False
        return True

    def load_settings(self):
        self._settings = _stored_settings()

        for pane in self.panes:
            section = self._settings.get(pane.name)
            pane.load_settings(section if isinstance(section, dict) else {})

        self._remembered_mode = self._settings.get("mode")

    def restore_mode(self):
        for index, pane in enumerate(self.panes):
            if pane.name == self._remembered_mode:
                self.mode_notebook.set_current_page(index)
                break

        self.on_mode_switched(
            self.mode_notebook, None, self.mode_notebook.get_current_page()
        )
        return False

    def save_settings(self):
        known = {pane.name for pane in self.panes}
        settings = {k: v for k, v in self._settings.items() if k in known}
        settings["mode"] = self.current_pane().name

        for pane in self.panes:
            try:
                settings[pane.name] = pane.collect_settings()
            except Exception as e:  # noqa: BLE001
                print(f"Error collecting the {pane.mode_label} settings: {e}!")

        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(settings, f, indent=2)
        except OSError as e:
            print(f"Error saving settings: {e}!")
            return

        self._settings = settings

    def on_window_close(self, widget, event):
        self.save_settings()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        for pane in self.panes:
            pane.shutdown()

        for pane in self.panes:
            release = getattr(pane, "release_model", None)
            if release is not None and not pane.busy:
                release()

        return False


def _zimage_argument(arguments, name, fallback=None):
    flag = f"--{name}"
    for index, argument in enumerate(arguments):
        if argument == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
    return fallback


def _zimage_fetch(source, preference=GGUF_QUANT_PREFERENCE):
    source = normalize_huggingface_url((source or "").strip())
    if Path(source).is_file():
        return Path(source)

    if not source.lower().endswith(".gguf") and is_huggingface_repo(source):
        resolved = resolve_gguf_in_repo(source, preference)
        if resolved is None:
            raise ModelError(f"No GGUF transformer found in {source}.")
        print(f"Picked {resolved.rsplit('/', 1)[-1]} out of {source}.")
        source = resolved

    if "huggingface.co/" in source:
        tail = source.split("huggingface.co/", 1)[1].strip("/").split("/")
        if len(tail) >= 3:
            from huggingface_hub import hf_hub_download

            repo, filename = "/".join(tail[:2]), "/".join(tail[2:])
            print(f"Fetching {filename} from {repo}...")
            return Path(hf_hub_download(repo_id=repo, filename=filename))

    if is_direct_url(source):
        destination = DIT_DIR / source.rsplit("/", 1)[-1]
        if not destination.is_file():
            print(f"Downloading {destination.name}...")
            download_model(
                source,
                destination,
                progress=lambda done, total: print(
                    f"  {_format_size(done)}"
                    + (f" of {_format_size(total)}" if total else ""),
                    end="\r",
                ),
            )
            print()
        return destination

    raise ModelError(f"{source} is not a file, a URL or a Hugging Face repo.")


def zimage_render(arguments):
    if not VAE_AVAILABLE:
        print("Z-Image needs diffusers for the VAE and the scheduler.")
        return 1

    dit = _zimage_argument(arguments, "dit", ZIMAGE_DEFAULT_DIT)
    repo = _zimage_argument(arguments, "repo", ZIMAGE_COMPONENTS_REPO)
    prompt = _zimage_argument(
        arguments,
        "prompt",
        "a close-up photograph of a pair of hands holding an open book",
    )
    negative = _zimage_argument(arguments, "negative", "")
    steps = int(_zimage_argument(arguments, "steps", ZIMAGE_DEFAULT_STEPS))
    guidance = float(_zimage_argument(arguments, "guidance", ZIMAGE_DEFAULT_GUIDANCE))
    size = int(_zimage_argument(arguments, "size", ZIMAGE_DEFAULT_SIZE))
    width = int(_zimage_argument(arguments, "width", size))
    height = int(_zimage_argument(arguments, "height", size))
    seed = int(_zimage_argument(arguments, "seed", 42))
    shift = float(_zimage_argument(arguments, "shift", ZIMAGE_DEFAULT_SHIFT))
    output = Path(_zimage_argument(arguments, "out", "zimage.png"))
    device = _zimage_argument(arguments, "device", "auto").strip().lower()
    if device == "auto":
        device = zimage_default_device()

    print(describe_torch_build() + ".")
    print(
        f"Z-Image: {width}x{height}, {steps} steps, guidance {guidance:g}, "
        f"shift {shift:g}, seed {seed}."
    )

    from transformers import AutoModel, AutoTokenizer

    local = _zimage_fetch(dit)
    print(f"Reading the DiT from {local}...")
    if device != ZIMAGE_CPU_DEVICE:
        try:
            gpu = animus_vulkan.get_device(device)
        except (animus_vulkan.VulkanUnavailable, animus_vulkan.VulkanError) as e:
            print(f"Could not open {device}: {e} Try --device cpu.")
            return 1
        print(f"Using {gpu.describe()}.")
        tensors = animus_vulkan.read_checkpoints([local])
        config = animus_vulkan.detect_config(tensors)
        print(f"Detected: {config}.")
        transformer = animus_vulkan.VulkanZImage(gpu, tensors, config, progress=print)
        del tensors
    else:
        state = read_z_image_state_dict(local)
        config = detect_z_image_config(state)
        print(f"Detected: {config}.")
        transformer = build_z_image_transformer(state, config, progress=print)
        del state
    gc.collect()

    print(f"Loading the VAE, scheduler and tokenizer from {repo}...")
    vae = AutoencoderKL.from_pretrained(
        repo, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo, subfolder="scheduler"
    )
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")

    pipe = NativeZImagePipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=tokenizer,
    )
    pipe.scheduler = build_flow_scheduler(
        "Euler", scheduler, shift, dynamic_shifting=False
    )
    if device != ZIMAGE_CPU_DEVICE:
        pipe.vae = animus_vulkan.VulkanVAE(gpu, vae, progress=print, fallback=vae)

    print("Loading the Qwen3 text encoder...")
    pipe.text_encoder = AutoModel.from_pretrained(
        repo,
        subfolder="text_encoder",
        torch_dtype=ZIMAGE_TEXT_ENCODER_DTYPE,
        low_cpu_mem_usage=True,
    )
    print("Encoding the prompt...")
    with torch.inference_mode():
        embeds, negative_embeds = pipe.encode_prompt(
            prompt,
            guidance > 0.0,
            negative,
            max_sequence_length=ZIMAGE_MAX_SEQUENCE_LENGTH,
        )
    print(f"Caption is {len(embeds[0])} tokens of {embeds[0].shape[-1]}.")
    pipe.text_encoder = None
    gc.collect()

    started = time.monotonic()

    def report(pipeline, index, timestep, kwargs):
        done = index + 1
        print(
            f"  step {done}/{steps}  t={float(timestep):7.2f}  "
            f"{_format_duration(time.monotonic() - started)} elapsed",
            flush=True,
        )
        return kwargs

    with torch.inference_mode():
        result = pipe(
            prompt_embeds=embeds,
            negative_prompt_embeds=negative_embeds or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            cfg_truncation=ZIMAGE_DEFAULT_CFG_TRUNC,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            callback_on_step_end=report,
        )

    result.images[0].save(str(output), format="PNG")
    print(f"Wrote {output} in {_format_duration(time.monotonic() - started)}.")
    return 0


def main():
    global _window

    arguments = sys.argv[1:]

    if "--self-test" in arguments:
        sys.exit(self_test())

    if "--benchmark" in arguments:
        rest = [a for a in arguments if a != "--benchmark"]
        sys.exit(benchmark(rest[0] if rest else None))

    if "--zimage" in arguments:
        sys.exit(zimage_render(arguments))

    if any(a in ("-h", "--help") for a in arguments):
        print("Usage: animus [--self-test] [--benchmark] [--zimage] [VIDEO]")
        print()
        print("  --self-test   check the upscaling networks and the ffmpeg pipeline")
        print("  --benchmark   time the installed upscaling models")
        print("  --zimage      render one image with Z-Image and exit")
        print()
        print("  Z-Image options: --dit --repo --prompt --negative --steps")
        print("                   --guidance --size --width --height --seed")
        print("                   --shift --out --device (cpu, auto or vulkan:N)")
        print()
        print("A video argument opens the Upscale tab with that file loaded.")
        sys.exit(0)

    def sigint_handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    window = AnimusWindow()
    _window = window
    window.connect("delete-event", window.on_window_close)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.set_focus(None)
    window.generate.model_entry.select_region(0, 0)
    window.restore_mode()

    source = next((a for a in arguments if not a.startswith("-")), None)
    if source:
        window.select_mode(window.upscale)
        window.upscale.set_source(source)

    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received - shutting down...")
        if _window is not None:
            for pane in _window.panes:
                if pane.busy:
                    print("Stopping operations...")
                    pane.stop_event.set()
            terminate_process(_window.upscale.decoder)
            terminate_process(_window.upscale.encoder)
        sys.exit(0)


if __name__ == "__main__":
    main()
