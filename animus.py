#!/usr/bin/env python3

# autopep8: off

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

import shutil
import errno
from pathlib import Path
import json
from datetime import datetime
import threading
import ctypes
import re
import random


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
from gi.repository import Gtk, GdkPixbuf, GLib
import warnings

warnings.filterwarnings(
    "ignore", message=".*device_type of 'cuda', but CUDA is not available.*"
)
from diffusers import logging
import torch

try:
    from diffusers import (
        ModularPipeline,
        CosmosTransformer3DModel,
        GGUFQuantizationConfig,
        FlowMatchEulerDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    ANIMA_AVAILABLE = True
except Exception:  # pragma: Requires diffusers >= 0.39.0.
    ModularPipeline = None
    CosmosTransformer3DModel = None
    GGUFQuantizationConfig = None
    FlowMatchEulerDiscreteScheduler = None
    UniPCMultistepScheduler = None
    ANIMA_AVAILABLE = False

try:
    from diffusers import ClassifierFreeGuidance
except Exception:
    try:
        from diffusers.guiders import ClassifierFreeGuidance
    except Exception:
        ClassifierFreeGuidance = None

from PIL import Image as _PILImage
from PIL.PngImagePlugin import PngInfo

_PILImage.preinit()

import gc
import signal
import traceback

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
except Exception as e:
    print(f"Warning: unexpected error while patching tqdm: {e}.", file=sys.stderr)

CONFIG_FILE = Path.home() / ".config" / "animus" / "settings.json"
OUTPUT_DIR = Path.home() / ".config" / "animus" / "outputs"
LORA_DIR = Path.home() / ".config" / "animus" / "loras"
MODEL_DIR = Path.home() / ".config" / "animus" / "models"
EMBEDDING_DIR = Path.home() / ".config" / "animus" / "embeddings"

MIN_FREE_DISK_MARGIN = 256 * 1024 * 1024

GENERATION_THREAD_TIMEOUT = 5.0
LOAD_THREAD_TIMEOUT = 3.0

NUM_LORA_SLOTS = 4
NUM_EMBEDDING_SLOTS = 2

ANIMA_COMPONENTS_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
ANIMA_DEFAULT_DIT = (
    "https://huggingface.co/Abiray/Anima-turbo-v1.0-GGUF/"
    "anima-turbo-v1.0-Q4_K_M.gguf"
)
ANIMA_DEFAULT_STEPS = 8
ANIMA_DEFAULT_GUIDANCE = 1.5
ANIMA_DEFAULT_SIZE = 512

ANIMA_SAMPLERS = ("Euler", "Euler Ancestral", "UniPC")
ANIMA_DEFAULT_SAMPLER = "Euler"
ANIMA_DEFAULT_SHIFT = 3.0
ANIMA_DEFAULT_SEED = -1
ANIMA_SEED_MAX = 2**32 - 1

# XXX: Real value here?
ANIMA_TOKEN_LIMIT = 512
ANIMA_TOKEN_WARNING = 448

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


def build_anima_scheduler(sampler, base_scheduler, shift=None):
    config = dict(base_scheduler.config)

    if shift is None:
        shift = config.get("shift", 1.0)
    try:
        shift = float(shift)
    except (TypeError, ValueError):
        shift = 1.0
    if shift <= 0.0:
        shift = 1.0

    if sampler == "Euler Ancestral":
        return FlowMatchEulerDiscreteScheduler.from_config(
            config, shift=shift, stochastic_sampling=True
        )

    if sampler == "UniPC":
        return UniPCMultistepScheduler(
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            solver_order=2,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
        )

    return FlowMatchEulerDiscreteScheduler.from_config(config, shift=shift)


_gui_instance = None
_anima_step_hook = None
_anima_stop_check = None

# autopep8: on


def raise_exception_in_thread(thread_obj):
    if thread_obj is None or not thread_obj.is_alive():
        return False

    thread_id = None
    try:
        for tid, tobj in threading._active.items():
            if tobj is thread_obj:
                thread_id = tid
                break
    except Exception as e:
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

        return  # Nothing to shim.
    except Exception:
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
    except Exception as e:
        print(f"Warning: could not install the Cosmos torchvision shim: {e}.")


def _set_anima_step_hook(fn):
    global _anima_step_hook
    _anima_step_hook = fn


def _set_anima_stop_check(fn):
    global _anima_stop_check
    _anima_stop_check = fn


def _install_anima_denoise_hook():
    # Live preview patch.
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
                    except Exception:
                        pass
                    hook(getattr(bs, "latents", None), kwargs.get("i"))
            except Exception:
                pass
            return result

        wrapper.loop_step = _patched_loop_step
        wrapper._animus_hooked = True
    except Exception as e:
        print(f"Warning: could not install the preview hook: {e}.")


def _format_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def is_huggingface_repo(path_str):
    if not path_str or not isinstance(path_str, str):
        return False

    path_obj = Path(path_str)

    if path_obj.exists():
        return False

    if len(path_str) >= 2 and path_str[1] == ":" and path_str[0].isalpha():
        return False

    parts = path_str.split("/")
    if len(parts) == 2 and not path_str.startswith((".", "/")):
        if not any(c in path_str for c in ["\\", "~"]):
            return True

    return False


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(text):
    return _ANSI_ESCAPE.sub("", text)


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


class AnimusGUI(Gtk.Window):
    def __init__(self):
        super().__init__(title="Animus")
        warnings.filterwarnings("ignore")
        self.set_wmclass("Animus", "Animus")
        self.set_default_size(800, 900)
        self.set_border_width(10)

        self.pipe = None
        self._base_scheduler = None
        self.generating = False
        self.loading_model = False
        self.stop_event = threading.Event()
        self.generation_thread = None
        self.load_thread = None
        self.current_image_path = None
        self.stop_click_count = 0
        self._loading_settings = False
        self.preview_shown = False
        self._latent_rgb_weight = torch.tensor(
            ANIMA_LATENT_RGB_FACTORS, dtype=torch.float32
        )
        self._latent_rgb_bias = torch.tensor(ANIMA_LATENT_RGB_BIAS, dtype=torch.float32)

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception as e:
            print(f"Warning: Could not load the tokenizer for token counting: {e}.")
            self.tokenizer = None

        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LORA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(main_box)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_size_request(-1, 400)

        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        controls_box.set_border_width(10)
        scrolled.add(controls_box)
        main_box.pack_start(scrolled, False, False, 0)

        top_spacer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        top_spacer.set_size_request(-1, 20)
        controls_box.pack_start(top_spacer, False, False, 0)

        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        model_label = Gtk.Label(label="DiT (GGUF):")
        model_label.set_size_request(100, -1)
        model_label.set_xalign(0)
        model_box.pack_start(model_label, False, False, 0)

        self.model_entry = Gtk.Entry()
        self.model_entry.set_text(ANIMA_DEFAULT_DIT)
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

        self.embedding_entries = []
        self.embedding_token_entries = []

        for i in range(NUM_EMBEDDING_SLOTS):
            embedding_frame = Gtk.Frame(label=f"Negative Embedding {i + 1}")
            embedding_frame_box = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL, spacing=5
            )
            embedding_frame_box.set_border_width(5)
            embedding_frame.add(embedding_frame_box)

            embedding_path_box = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL, spacing=5
            )
            embedding_path_label = Gtk.Label(label="File:")
            embedding_path_label.set_size_request(80, -1)
            embedding_path_label.set_xalign(0)
            embedding_path_box.pack_start(embedding_path_label, False, False, 0)

            embedding_entry = Gtk.Entry()
            embedding_path_box.pack_start(embedding_entry, True, True, 0)
            self.embedding_entries.append(embedding_entry)

            embedding_browse_btn = Gtk.Button(label="Browse...")
            embedding_browse_btn.connect("clicked", self.on_browse_embedding, i)
            embedding_path_box.pack_start(embedding_browse_btn, False, False, 0)

            embedding_clear_btn = Gtk.Button(label="Clear")
            embedding_clear_btn.connect("clicked", self.on_clear_embedding, i)
            embedding_path_box.pack_start(embedding_clear_btn, False, False, 0)

            embedding_frame_box.pack_start(embedding_path_box, False, False, 0)

            embedding_token_box = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL, spacing=5
            )
            embedding_token_label = Gtk.Label(label="Token:")
            embedding_token_label.set_size_request(80, -1)
            embedding_token_label.set_xalign(0)
            embedding_token_box.pack_start(embedding_token_label, False, False, 0)

            embedding_token_entry = Gtk.Entry()
            embedding_token_box.pack_start(embedding_token_entry, True, True, 0)
            self.embedding_token_entries.append(embedding_token_entry)

            embedding_frame_box.pack_start(embedding_token_box, False, False, 0)

            controls_box.pack_start(embedding_frame, False, False, 0)

        resolution_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        resolution_label = Gtk.Label(label="Size (W x H):")
        resolution_label.set_size_request(100, -1)
        resolution_label.set_xalign(0)
        resolution_box.pack_start(resolution_label, False, False, 0)

        self.width_spin = Gtk.SpinButton()
        width_adj = Gtk.Adjustment(
            value=ANIMA_DEFAULT_SIZE,
            lower=256,
            upper=1536,
            step_increment=64,
            page_increment=128,
        )
        self.width_spin.set_adjustment(width_adj)
        self.width_spin.set_size_request(90, -1)
        resolution_box.pack_start(self.width_spin, False, False, 0)

        self.height_spin = Gtk.SpinButton()
        height_adj = Gtk.Adjustment(
            value=ANIMA_DEFAULT_SIZE,
            lower=256,
            upper=1536,
            step_increment=64,
            page_increment=128,
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
            value=ANIMA_DEFAULT_STEPS,
            lower=1,
            upper=100,
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
            value=ANIMA_DEFAULT_GUIDANCE,
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
        for sampler_name in ANIMA_SAMPLERS:
            self.sampler_combo.append(sampler_name, sampler_name)
        self.sampler_combo.set_active_id(ANIMA_DEFAULT_SAMPLER)

        sampler_box.pack_start(self.sampler_combo, False, False, 0)
        controls_box.pack_start(sampler_box, False, False, 0)

        shift_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        shift_label = Gtk.Label(label="Shift:")
        shift_label.set_size_request(100, -1)
        shift_label.set_xalign(0)
        shift_box.pack_start(shift_label, False, False, 0)

        self.shift_spin = Gtk.SpinButton()
        shift_adj = Gtk.Adjustment(
            value=ANIMA_DEFAULT_SHIFT,
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
        preview_label = Gtk.Label(label="Live preview:")
        preview_label.set_size_request(100, -1)
        preview_label.set_xalign(0)
        preview_box.pack_start(preview_label, False, False, 0)
        self.preview_check = Gtk.CheckButton(label="Generate a rough preview each step")
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

        self.trigger_token_label = Gtk.Label(label=f"0/{ANIMA_TOKEN_LIMIT}")
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

        self.prompt_token_label = Gtk.Label(label=f"0/{ANIMA_TOKEN_LIMIT}")
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

        self.neg_prompt_token_label = Gtk.Label(label=f"0/{ANIMA_TOKEN_LIMIT}")
        self.neg_prompt_token_label.set_xalign(1)
        controls_box.pack_start(self.neg_prompt_token_label, False, False, 0)

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

        controls_box.pack_start(button_box, False, False, 0)

        self.notebook = Gtk.Notebook()
        main_box.pack_start(self.notebook, True, True, 0)

        console_scrolled = Gtk.ScrolledWindow()
        console_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.console_text = Gtk.TextView()
        self.console_text.set_editable(False)
        self.console_text.set_wrap_mode(Gtk.WrapMode.CHAR)
        self.console_text.set_cursor_visible(False)

        try:
            css_provider = Gtk.CssProvider()
            css_provider.load_from_data(b"""
                textview {
                    font-family: monospace;
                    font-size: 12pt;
                }
            """)
            style_context = self.console_text.get_style_context()
            style_context.add_provider(
                css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
        except Exception as e:
            print(f"Could not set monospace font: {e}.")

        self.console_buffer = self.console_text.get_buffer()
        console_scrolled.add(self.console_text)

        console_label = Gtk.Label(label="Console Output")
        self.notebook.append_page(console_scrolled, console_label)

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

        image_label = Gtk.Label(label="Generated Image")
        self.notebook.append_page(image_scrolled, image_label)

        self.notebook.set_current_page(0)

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text, self.original_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.original_stderr)

        self.load_settings()

    def update_status(self, message):
        if hasattr(sys.stdout, "write_with_newline"):
            sys.stdout.write_with_newline(f"{message}\n")
        else:
            print(f"{message}\n", end="")

    def count_tokens(self, text):
        if not self.tokenizer or not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            print(f"Error counting tokens: {e}.")
            return 0

    def on_text_changed(self, widget=None):
        if not self.tokenizer:
            return

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
        self._update_token_label(self.neg_prompt_token_label, neg_tokens)

    def _update_token_label(self, label, count):
        if count > ANIMA_TOKEN_LIMIT:
            color = "red"
        elif count > ANIMA_TOKEN_WARNING:
            color = "orange"
        else:
            color = "green"
        weight = ' weight="bold"' if count > ANIMA_TOKEN_WARNING else ""
        label.set_markup(
            f'<span foreground="{color}"{weight}>' f"{count}/{ANIMA_TOKEN_LIMIT}</span>"
        )

    def load_settings(self):
        self._loading_settings = True
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r") as f:
                    settings = json.load(f)

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
                if settings.get("sampler") in ANIMA_SAMPLERS:
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
                    self.neg_prompt_text.get_buffer().set_text(
                        settings["negative_prompt"]
                    )

                embeddings = settings.get("embeddings")
                if embeddings is None and "embedding" in settings:
                    embeddings = [settings["embedding"]]
                if embeddings:
                    for i, embedding_data in enumerate(
                        embeddings[:NUM_EMBEDDING_SLOTS]
                    ):
                        self.embedding_entries[i].set_text(
                            embedding_data.get("path", "")
                        )
                        self.embedding_token_entries[i].set_text(
                            embedding_data.get("token", "")
                        )
        except Exception as e:
            print(f"Error loading settings: {e}.")
        finally:
            self._loading_settings = False

        self.on_text_changed(None)

    def save_settings(self):
        try:
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
                "guidance": float(self.guidance_spin.get_value()),
                "sampler": self.sampler_combo.get_active_id() or ANIMA_DEFAULT_SAMPLER,
                "shift": float(self.shift_spin.get_value()),
                "seed": int(self.seed_spin.get_value()),
                "width": int(self.width_spin.get_value()),
                "height": int(self.height_spin.get_value()),
                "preview": self.preview_check.get_active(),
                "trigger": trigger,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "embeddings": [
                    {
                        "path": embedding_entry.get_text(),
                        "token": embedding_token_entry.get_text(),
                    }
                    for embedding_entry, embedding_token_entry in zip(
                        self.embedding_entries, self.embedding_token_entries
                    )
                ],
            }

            for lora_entry, (weight_name_entry, weight_spin) in zip(
                self.lora_entries, self.lora_weight_entries
            ):
                settings["loras"].append(
                    {
                        "path": lora_entry.get_text(),
                        "weight_name": weight_name_entry.get_text(),
                        "weight_value": weight_spin.get_value(),
                    }
                )

            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            with open(CONFIG_FILE, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}!")

    def _copy_to_config(self, src_path, dest_file, description):
        try:
            src_size = src_path.stat().st_size

            if dest_file.exists() and dest_file.stat().st_size == src_size:
                print(f"Already in config (up to date): {src_path.name}")
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
                f"Copying {description} to config: {src_path.name} "
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
        except Exception as e:
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
            title="Select Anima GGUF (diffusion transformer)",
            parent=self,
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
                dest_file = MODEL_DIR / path_obj.stem / path_obj.name
                if self._copy_to_config(path_obj, dest_file, "model file"):
                    self.model_entry.set_text(str(dest_file))
                else:
                    self.model_entry.set_text(str(path_obj))
            else:
                print(f"Warning: selected path is not a file: {selected_path}.")

        dialog.destroy()

    def on_browse_lora(self, button, lora_index):
        dialog = Gtk.FileChooserDialog(
            title="Select LoRA Weight File",
            parent=self,
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

                    if self._copy_to_config(path_obj, dest_file, "LoRA weight file"):
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

    def on_browse_embedding(self, button, embedding_index):
        dialog = Gtk.FileChooserDialog(
            title="Select Negative Embedding File",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        filter_embedding = Gtk.FileFilter()
        filter_embedding.set_name("Embedding files")
        for pattern in ("*.safetensors", "*.pt", "*.bin"):
            filter_embedding.add_pattern(pattern)
        dialog.add_filter(filter_embedding)

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
                dest_file = EMBEDDING_DIR / path_obj.stem / path_obj.name

                if self._copy_to_config(path_obj, dest_file, "embedding file"):
                    self.embedding_entries[embedding_index].set_text(str(dest_file))
                else:
                    self.embedding_entries[embedding_index].set_text(str(path_obj))

                self.embedding_token_entries[embedding_index].set_text(path_obj.stem)
            else:
                print(f"Warning: selected path is not a file: {selected_path}.")

        dialog.destroy()

    def on_clear_embedding(self, button, embedding_index):
        self.embedding_entries[embedding_index].set_text("")
        self.embedding_token_entries[embedding_index].set_text("")

    def on_load_clicked(self, button):
        if self.generating or self.loading_model:
            return

        self.loading_model = True
        self.stop_event.clear()
        self.stop_click_count = 0
        self.load_button.set_sensitive(False)
        self.generate_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.stop_button.show()

        if self.pipe is not None:
            self.update_status("Reloading model... Click Stop to cancel.")
        else:
            self.update_status("Loading model... Click Stop to cancel.")

        self.load_thread = threading.Thread(target=self.load_model_thread, daemon=True)
        self.load_thread.start()

    def _check_stop_loading(self, cleanup_pipe=False):
        if self.stop_event.is_set():
            self.update_status("Interrupted by user.")
            if cleanup_pipe and self.pipe is not None:
                self.pipe = None
                gc.collect()
            return True
        return False

    def load_model_thread(self):
        try:
            if self._check_stop_loading():
                return

            model_name = self.model_entry.get_text().strip()

            if self.pipe is not None:
                if self._check_stop_loading():
                    return
                self.update_status("Cleaning up existing model...")
                del self.pipe
                self.pipe = None
                self._base_scheduler = None
                gc.collect()

            if self._check_stop_loading():
                return

            self.pipe = self._load_anima_pipeline(model_name)
            if self.pipe is None:
                return

            if self._check_stop_loading(cleanup_pipe=True):
                return

            self._load_loras()

            if self._check_stop_loading(cleanup_pipe=True):
                return

            self._load_textual_inversions()

            if self._check_stop_loading(cleanup_pipe=True):
                return

            GLib.idle_add(self._enable_generate_and_load)
            self.update_status("Ready!")

        except KeyboardInterrupt:
            self.update_status("Interrupted by user.")
            if self.pipe is not None:
                self.pipe = None
                gc.collect()
            GLib.idle_add(self._enable_load)
        except Exception as e:
            if self.stop_event.is_set():
                self.update_status("Interrupted by user.")
            else:
                traceback.print_exc()
                self.update_status(f"Error loading model: {str(e)}!")
            GLib.idle_add(self._enable_load)
        finally:
            self.loading_model = False
            GLib.idle_add(self._hide_stop_button)

    def _load_anima_pipeline(self, dit_source):
        if not ANIMA_AVAILABLE:
            raise Exception("Anima requires diffusers >= 0.39.0 and gguf.")

        _install_cosmos_torchvision_shim()
        _install_anima_denoise_hook()

        dit_source = normalize_huggingface_url((dit_source or "").strip())
        if not dit_source:
            raise Exception("No Anima DiT specified.")

        self.update_status(f"Loading Anima DiT " f"from {dit_source}...")
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

        self.update_status(
            f"Loading Anima components " f"from {ANIMA_COMPONENTS_REPO}..."
        )
        pipe = ModularPipeline.from_pretrained(ANIMA_COMPONENTS_REPO)
        # We supply our own GGUF transformer via update_components() below.
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
        except Exception as e:
            print(
                f"Warning: selective component load failed: {e}. "
                "Loading all components."
            )
        if not loaded_selectively:
            pipe.load_components(torch_dtype=torch.float32)

        if self._check_stop_loading():
            del transformer
            del pipe
            gc.collect()
            return None

        self.update_status("Injecting the GGUF DiT into the pipeline...")
        pipe.update_components(transformer=transformer)

        try:
            vae = getattr(pipe, "vae", None)
            if vae is not None:
                if hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                if hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()
                    self.update_status("Enabled VAE tiling and slicing.")
        except Exception as e:
            print(f"Warning: {e}.")

        try:
            pipe.to("cpu")
        except Exception as e:
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
        except Exception:
            pass
        return None

    # XXX: Finish me.
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
        for key, val in state_dict.items():
            if not key.startswith("lora_unet_blocks_"):
                continue
            name_part, _, tail = key.partition(".")
            m = re.match(r"^lora_unet_blocks_(\d+)_(.+)$", name_part)
            if not m:
                continue
            block_idx, suffix = m.group(1), m.group(2)
            module = attn_map.get(suffix)
            if module is None:
                print(f"Warning: skipping unmapped module '{suffix}'.")
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
            weight_value = weight_spin.get_value()
            if lora_path:
                loras_to_load.append(
                    {
                        "path": lora_path,
                        "weight_name": weight_name,
                        "weight_value": weight_value,
                        "adapter_name": f"lora_{i}",
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
            self.update_status(f"Loading LoRA {label}...")
            try:
                converted = None
                local_file = self._lora_local_file(
                    lora_info["path"], lora_info["weight_name"]
                )
                if local_file is not None and str(local_file).endswith(".safetensors"):
                    from safetensors.torch import load_file

                    raw = load_file(str(local_file))
                    converted = self._convert_kohya_anima_lora(raw)

                if converted is not None:
                    n = sum(1 for k in converted if k.endswith("lora_A.weight"))
                    self.update_status(f"Converted {n} modules to diffusers format.")
                    self.pipe.load_lora_weights(
                        converted, adapter_name=lora_info["adapter_name"]
                    )
                else:
                    kwargs = {"adapter_name": lora_info["adapter_name"]}
                    if lora_info["weight_name"]:
                        kwargs["weight_name"] = lora_info["weight_name"]
                    self.pipe.load_lora_weights(lora_info["path"], **kwargs)

                adapter_weights[lora_info["adapter_name"]] = lora_info["weight_value"]
            except Exception as e:
                traceback.print_exc()
                self.update_status(
                    f"Warning: could not load LoRA {lora_info['adapter_name']} "
                    f"({label}): {str(e)}. Skipping."
                )

        if not adapter_weights:
            return

        # https://github.com/huggingface/diffusers/issues/12047
        try:
            self.pipe.set_adapters(
                list(adapter_weights.keys()),
                adapter_weights=list(adapter_weights.values()),
            )
            try:
                active = self.pipe.get_active_adapters()
            except Exception:
                active = list(adapter_weights.keys())
            self.update_status(f"Activated LoRA(s) at runtime: {active}.")
        except Exception as e:
            traceback.print_exc()
            self.update_status(f"Warning: could not activate LoRA adapters: {str(e)}.")

    def _extract_embedding_tensor(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            s2p = obj.get("string_to_param")
            if isinstance(s2p, dict):
                for val in s2p.values():
                    if isinstance(val, torch.Tensor):
                        return val
            for key in ("emb_params", "embedding", "clip_l", "weight", "*"):
                val = obj.get(key)
                if isinstance(val, torch.Tensor):
                    return val
            tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            two_d = [t for t in tensors if t.dim() == 2]
            if len(two_d) == 1:
                return two_d[0]
            if len(tensors) == 1:
                return tensors[0]
        return None

    def _load_embedding_file(self, path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Error: file not found: {path}.")

        if p.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            raw = load_file(str(p))
        else:
            raw = torch.load(str(p), map_location="cpu", weights_only=True)

        emb = self._extract_embedding_tensor(raw)
        if emb is None:
            raise ValueError("no embedding tensor found in the file")

        emb = emb.detach().to(dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if emb.dim() != 2:
            raise ValueError(f"unexpected embedding shape {tuple(emb.shape)}")
        return emb

    def _load_textual_inversions(self):
        embeddings_to_load = []
        for embedding_entry, embedding_token_entry in zip(
            self.embedding_entries, self.embedding_token_entries
        ):
            path = embedding_entry.get_text().strip()
            token = embedding_token_entry.get_text().strip()
            if path and token:
                embeddings_to_load.append((path, token))

        if not embeddings_to_load:
            return

        tokenizer = getattr(self.pipe, "tokenizer", None)
        text_encoder = getattr(self.pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            self.update_status(
                "Warning: this pipeline exposes no tokenizer or text_encoder. "
                "Skipping."
            )
            return

        try:
            encoder_dim = text_encoder.get_input_embeddings().weight.shape[1]
        except Exception as e:
            self.update_status(
                f"Warning: could not inspect the text encoder embeddings: {e}. "
                "Skipping."
            )
            return

        for path, token in embeddings_to_load:
            if self._check_stop_loading(cleanup_pipe=True):
                return

            self.update_status(f"Loading negative embedding '{token}'...")
            try:
                emb = self._load_embedding_file(path)
            except Exception as e:
                self.update_status(
                    f"Warning: could not read embedding '{token}' ({path}): "
                    f"{e}. Skipping."
                )
                continue

            num_vectors, dim = emb.shape[0], emb.shape[1]
            if dim != encoder_dim:
                self.update_status(
                    f"Warning: embedding '{token}' has dimension {dim} but the "
                    f"Anima text encoder expects {encoder_dim}. It was likely "
                    "trained for a different model. Skipping."
                )
                continue

            tokens = [token] + [f"{token}_{i}" for i in range(1, num_vectors)]
            num_added = tokenizer.add_tokens(tokens)
            if num_added != len(tokens):
                self.update_status(
                    f"Warning: '{token}' already exists in the " "tokenizer. Skipping."
                )
                continue

            try:
                text_encoder.resize_token_embeddings(len(tokenizer))
                input_embeddings = text_encoder.get_input_embeddings()
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                with torch.no_grad():
                    for i, token_id in enumerate(token_ids):
                        input_embeddings.weight[token_id] = emb[i].to(
                            dtype=input_embeddings.weight.dtype,
                            device=input_embeddings.weight.device,
                        )
            except Exception as e:
                traceback.print_exc()
                self.update_status(
                    f"Warning: could not inject embedding '{token}': {e}. Skipping."
                )
                continue

            if num_vectors == 1:
                self.update_status(f"Loaded negative embedding '{token}' (dim {dim}).")
            else:
                self.update_status(
                    f"Loaded negative embedding '{token}' as {num_vectors} "
                    f"tokens {tokens} (dim {dim})."
                )

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
        return False

    def _enable_load(self):
        self.load_button.set_sensitive(True)
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def _hide_stop_button(self):
        self.stop_button.hide()
        self.stop_button.set_sensitive(False)
        return False

    def on_generate_clicked(self, button):
        if self.generating or self.loading_model or self.pipe is None:
            return

        self.save_settings()

        self.generating = True
        self.stop_event.clear()
        self.stop_click_count = 0
        self.generate_button.set_sensitive(False)
        self.generate_button.hide()
        self.stop_button.set_sensitive(True)
        self.stop_button.show()

        self.generation_thread = threading.Thread(
            target=self.generate_image_thread, daemon=True
        )
        self.generation_thread.start()

    def on_stop_clicked(self, button):
        if self.generating or self.loading_model:
            self.stop_event.set()
            self.stop_click_count += 1

            self.update_status(
                f"FORCE STOP #{self.stop_click_count} - Sending interrupt..."
            )
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

    def _apply_sampler(self, sampler, shift=None):
        base = self._base_scheduler or getattr(self.pipe, "scheduler", None)
        if base is None or self.pipe is None:
            return
        try:
            scheduler = build_anima_scheduler(sampler, base, shift)
            self.pipe.update_components(scheduler=scheduler)
        except Exception as e:
            print(
                f"Warning: could not select sampler '{sampler}': {e}. "
                "Falling back to the pipeline default."
            )
            try:
                self.pipe.update_components(scheduler=base)
            except Exception:
                pass

    def _apply_guidance(self, guidance):
        if self.pipe is None:
            return
        if ClassifierFreeGuidance is None:
            # Trouble ahead.
            return
        try:
            self.pipe.update_components(
                guider=ClassifierFreeGuidance(guidance_scale=float(guidance))
            )
            try:
                active = self.pipe.guider.guidance_scale
            except Exception:
                active = guidance
        except Exception as e:
            print(
                f"Warning: could not apply guidance {guidance} to the guider "
                f"component: {e}. The pipeline's default CFG will be used."
            )

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
                loras.append(f"{path}/{name}@{weight_spin.get_value():g}")
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
                self.update_status("Error: Prompt cannot be empty!")
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
            guidance = float(self.guidance_spin.get_value())
            width = int(self.width_spin.get_value())
            height = int(self.height_spin.get_value())
            sampler = self.sampler_combo.get_active_id() or ANIMA_DEFAULT_SAMPLER
            shift = float(self.shift_spin.get_value())
            seed = int(self.seed_spin.get_value())
            if seed < 0:
                seed = random.randint(0, ANIMA_SEED_MAX)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            self._apply_sampler(sampler, shift)
            self._apply_guidance(guidance)

            self.update_status(
                f"Generating with Anima ({sampler}, shift {shift:g}) at "
                f"{width}x{height} with {steps} steps, guidance {guidance}, "
                f"and seed {seed}..."
            )

            self.preview_shown = False
            _set_anima_stop_check(lambda: self.stop_event.is_set())
            if self.preview_check.get_active():
                _set_anima_step_hook(self._on_denoise_step)

            result = None
            with torch.inference_mode():
                result = self.pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    generator=generator,
                )

                image = getattr(result, "images", [None])[0] if result else None

            if result is not None:
                del result
            gc.collect()

            if self.stop_event.is_set():
                self.update_status("Interrupted by user.")
            elif image is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"animus_{timestamp}.png"
                info = self._png_metadata(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance=guidance,
                    sampler=sampler,
                    shift=shift,
                    seed=seed,
                    size=f"{width}x{height}",
                )
                image.save(str(output_path), format="PNG", pnginfo=info)

                GLib.idle_add(self._display_image, str(output_path))
                self.update_status(f"Done! Image saved to {output_path} (seed {seed}).")
            else:
                self.update_status("No image produced.")

        except KeyboardInterrupt:
            self.update_status("Interrupted by user.")
            GLib.idle_add(self._reset_generate_button)
        except Exception as e:
            if self.stop_event.is_set():
                self.update_status("Interrupted by user.")
            else:
                traceback.print_exc()
                self.update_status(f"Error generating image: {str(e)}")
        finally:
            _set_anima_step_hook(None)
            _set_anima_stop_check(None)
            self.generating = False
            GLib.idle_add(self._reset_generate_button)

    def _on_denoise_step(self, latents, step_index):
        if latents is None or not self.generating or self.stop_event.is_set():
            return
        try:
            data = self._latents_to_rgb_bytes(latents)
            if data is None:
                return
            rgb_bytes, width, height = data
            GLib.idle_add(self._show_preview, rgb_bytes, width, height)
        except Exception as e:
            print(f"Preview failed: {e}.", file=sys.stderr)

    def _latents_to_rgb_bytes(self, latents):
        try:
            with torch.inference_mode():
                lat = latents.detach().to(dtype=torch.float32, device="cpu")
                if lat.dim() == 5:  # [B, C, T, H, W]
                    lat = lat[0, :, 0]
                elif lat.dim() == 4:  # [B, C, H, W]
                    lat = lat[0]
                elif lat.dim() != 3:  # [C, H, W] expected
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
        except Exception as e:
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
            if longest < PREVIEW_DISPLAY_SIZE:
                scale = PREVIEW_DISPLAY_SIZE / longest
                pixbuf = pixbuf.scale_simple(
                    int(round(width * scale)),
                    int(round(height * scale)),
                    GdkPixbuf.InterpType.BILINEAR,
                )

            self.image_display.set_from_pixbuf(pixbuf)

            if not self.preview_shown:
                self.preview_shown = True
                self.notebook.set_current_page(1)
        except Exception as e:
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
            self.notebook.set_current_page(1)
        except Exception as e:
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
                    print(f"Deleted image: {self.current_image_path}")
                    self.update_status("Image deleted successfully.")
                else:
                    print(f"Image file no longer exists: " f"{self.current_image_path}")
                    self.update_status("Image file no longer exists.")
                self._clear_image_display()
            except Exception as e:
                error_msg = f"Error deleting image: {e}."
                print(error_msg, file=sys.stderr)
                self.update_status(error_msg)

    def on_restore_defaults_clicked(self, button):
        self.model_entry.set_text(ANIMA_DEFAULT_DIT)
        self.preview_check.set_active(True)

        for i in range(NUM_LORA_SLOTS):
            self.lora_entries[i].set_text("")
            weight_name_entry, weight_spin = self.lora_weight_entries[i]
            weight_name_entry.set_text("")
            weight_spin.set_value(0.5)

        for embedding_entry, embedding_token_entry in zip(
            self.embedding_entries, self.embedding_token_entries
        ):
            embedding_entry.set_text("")
            embedding_token_entry.set_text("")

        self.width_spin.set_value(ANIMA_DEFAULT_SIZE)
        self.height_spin.set_value(ANIMA_DEFAULT_SIZE)
        self.steps_spin.set_value(ANIMA_DEFAULT_STEPS)
        self.guidance_spin.set_value(ANIMA_DEFAULT_GUIDANCE)
        self.sampler_combo.set_active_id(ANIMA_DEFAULT_SAMPLER)
        self.shift_spin.set_value(ANIMA_DEFAULT_SHIFT)
        self.seed_spin.set_value(ANIMA_DEFAULT_SEED)

        self.trigger_text.get_buffer().set_text("")
        self.prompt_text.get_buffer().set_text("")
        self.neg_prompt_text.get_buffer().set_text("")

        self.update_status("Settings restored to defaults.")

    def on_window_close(self, widget, event):
        self.save_settings()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self._cleanup_threads()
        return False

    def _cleanup_threads(self):
        if self.generating or self.loading_model:
            self.stop_event.set()
            self.update_status("Waiting for operations to stop...")

        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=GENERATION_THREAD_TIMEOUT)

        if self.load_thread and self.load_thread.is_alive():
            self.load_thread.join(timeout=LOAD_THREAD_TIMEOUT)


def main():
    global _gui_instance

    def sigint_handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    win = AnimusGUI()
    _gui_instance = win
    win.connect("delete-event", win.on_window_close)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    win.set_focus(None)
    win.model_entry.select_region(0, 0)

    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received - shutting down...")
        if _gui_instance:
            if _gui_instance.generating or _gui_instance.loading_model:
                print("Stopping operations...")
                _gui_instance.stop_event.set()
        sys.exit(0)


if __name__ == "__main__":
    main()
