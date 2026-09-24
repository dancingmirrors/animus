#!/usr/bin/env python3

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy

SHADER_DIR = Path(__file__).resolve().parent / "shaders"
SHADER_CACHE_DIR = (
    Path(
        os.environ.get("XDG_CACHE_HOME")
        if os.environ.get("XDG_CACHE_HOME")
        and os.path.isabs(os.environ["XDG_CACHE_HOME"])
        else Path.home() / ".cache"
    )
    / "animus"
    / "shaders"
)
MANIFEST_NAME = "manifest.json"
GLSL_COMPILERS = ("glslangValidator", "glslc")

QUANT_TYPES = (
    "Q8_0",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
)

# name -> (source, defines); one SPIR-V blob each.
SHADER_VARIANTS = {}
for _q in QUANT_TYPES:
    SHADER_VARIANTS[f"dequant_{_q.lower()}"] = ("dequant.comp", (_q,))
for _b in ("F16", "F32"):
    for _l in ("NK", "KN"):
        for _s in ("LARGE", "SMALL"):
            SHADER_VARIANTS[f"matmul_{_b.lower()}_{_l.lower()}_{_s.lower()}"] = (
                "matmul.comp",
                (f"B_{_b}", f"B_{_l}", _s),
            )
SHADER_VARIANTS["norm"] = ("norm.comp", ())
SHADER_VARIANTS["head_norm_rope"] = ("head_norm_rope.comp", ())
SHADER_VARIANTS["softmax"] = ("softmax.comp", ())
SHADER_VARIANTS["silu_mul"] = ("silu_mul.comp", ())
SHADER_VARIANTS["im2col"] = ("im2col.comp", ())
SHADER_VARIANTS["gn_stats"] = ("gn_stats.comp", ())
SHADER_VARIANTS["gn_finalize"] = ("gn_finalize.comp", ())
SHADER_VARIANTS["gn_apply"] = ("gn_apply.comp", ())

SHADER_BINDINGS = {
    "dequant": 2,
    "matmul": 4,
    "norm": 4,
    "head_norm_rope": 4,
    "softmax": 1,
    "silu_mul": 2,
    "im2col": 2,
    "gn_stats": 3,
    "gn_finalize": 2,
    "gn_apply": 5,
}

PUSH_CONSTANT_BYTES = 64

NORM_WEIGHT = 1
NORM_SCALE = 2
NORM_RESIDUAL = 4
NORM_GATE = 8
NORM_CENTER = 16

MATMUL_ACCUMULATE = 1
MATMUL_BIAS = 2

ROPE_NORM = 1
ROPE_ROTATE = 2

MATMUL_LARGE_TILE = 128
MATMUL_SMALL_TILE = 32

# ggml block geometry: values per block, bytes per block.
BLOCK_SIZES = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
}

DEVICE_KINDS = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}

CHUNK_BYTES = 256 * 1024 * 1024
STAGING_BYTES = 64 * 1024 * 1024


def _env_megabytes(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(float(raw) * 1024 * 1024)
    except ValueError:
        return None


SCORES_MAX_BYTES = _env_megabytes("ANIMUS_VULKAN_SCORES_MB") or 1024 * 1024 * 1024
SCORES_MAX_UNIFIED_BYTES = (
    _env_megabytes("ANIMUS_VULKAN_SCORES_MB") or 512 * 1024 * 1024
)
HEAP_MARGIN_BYTES = _env_megabytes("ANIMUS_VULKAN_MARGIN_MB")
HEAP_MARGIN_FRACTION = 0.08
LARGE_TILE_SHARED_BYTES = 2 * 16 * (MATMUL_LARGE_TILE + 1) * 4
RESERVE_TOKENS = 4096 + 512
IM2COL_BYTES = 96 * 1024 * 1024
GN_BLOCK_PIXELS = 64
GN_MAX_CHANNELS = 1024
VAE_TILE_CHOICES = (128, 96, 64, 48, 32, 16)
ALIGNMENT = 256
DESCRIPTOR_SETS_PER_POOL = 4096
FENCE_TIMEOUT_NS = 300 * 10**9

_TRUTHY = ("1", "true", "yes", "on")
PROFILE = os.environ.get("ANIMUS_VULKAN_PROFILE", "").lower() in _TRUTHY


class VulkanUnavailable(Exception):
    pass


class VulkanError(Exception):
    pass


def _vk():
    try:
        import vulkan
    except ImportError as e:
        raise VulkanUnavailable(
            f"The vulkan Python package is not installed ({e}). It is in "
            "requirements.txt, and needs libvulkan (the Vulkan loader) on "
            "the system."
        ) from e
    except OSError as e:
        raise VulkanUnavailable(
            f"The Vulkan loader could not be opened ({e}). Install the "
            "system's Vulkan loader package (libvulkan1 on Debian and "
            "Ubuntu, vulkan-loader on Alpine)."
        ) from e
    return vulkan


def format_bytes(count):
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _source_hash(source_path, defines):
    digest = hashlib.sha256()
    digest.update(source_path.read_bytes())
    digest.update("\n".join(defines).encode())
    return digest.hexdigest()[:16]


def find_glsl_compiler():
    for name in GLSL_COMPILERS:
        found = shutil.which(name)
        if found:
            return found
    return None


def compile_shader(compiler, source, defines, output):
    command = [compiler]
    if Path(compiler).name.startswith("glslc"):
        command += ["-fshader-stage=compute", "--target-env=vulkan1.1"]
        command += [f"-D{define}" for define in defines]
        command += ["-o", str(output), str(source)]
    else:
        command += ["-V", "--target-env", "vulkan1.1"]
        command += [f"-D{define}" for define in defines]
        command += ["-o", str(output), str(source)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise VulkanError(
            f"Compiling {source.name} with {' '.join(defines) or 'no defines'} "
            f"failed:\n{result.stdout}{result.stderr}"
        )


def _read_manifest(directory):
    try:
        with open(directory / MANIFEST_NAME, "r") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def build_shaders(output_dir=SHADER_DIR, force=False, report=print):
    compiler = find_glsl_compiler()
    if compiler is None:
        raise VulkanError(
            "No GLSL compiler was found. Install glslang (glslangValidator) "
            "or shaderc (glslc) to rebuild the shaders."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {} if force else _read_manifest(output_dir)
    manifest = {name: manifest[name] for name in SHADER_VARIANTS if name in manifest}
    built = 0
    for name, (source_name, defines) in SHADER_VARIANTS.items():
        source = SHADER_DIR / source_name
        digest = _source_hash(source, defines)
        target = output_dir / f"{name}.spv"
        if not force and manifest.get(name) == digest and target.is_file():
            continue
        compile_shader(compiler, source, defines, target)
        manifest[name] = digest
        built += 1
    with open(output_dir / MANIFEST_NAME, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    report(
        f"Built {built} shader variant(s) with {Path(compiler).name} into {output_dir}."
        if built
        else f"The shaders in {output_dir} are already up to date."
    )
    return built


def stale_shaders(directory=SHADER_DIR):
    manifest = _read_manifest(directory)
    stale = []
    for name, (source_name, defines) in SHADER_VARIANTS.items():
        source = SHADER_DIR / source_name
        if not source.is_file():
            continue
        if manifest.get(name) != _source_hash(source, defines):
            stale.append(name)
        elif not (Path(directory) / f"{name}.spv").is_file():
            stale.append(name)
    return stale


_shader_dir_resolved = None


def shader_directory(report=print):
    global _shader_dir_resolved
    if _shader_dir_resolved is not None:
        return _shader_dir_resolved

    stale = stale_shaders(SHADER_DIR)
    if not stale:
        _shader_dir_resolved = SHADER_DIR
        return SHADER_DIR

    if find_glsl_compiler() is not None:
        try:
            build_shaders(SHADER_CACHE_DIR, report=report)
            if not stale_shaders(SHADER_CACHE_DIR):
                _shader_dir_resolved = SHADER_CACHE_DIR
                return SHADER_CACHE_DIR
        except VulkanError as e:
            report(f"Could not rebuild the shaders: {e}")

    missing = [name for name in stale if not (SHADER_DIR / f"{name}.spv").is_file()]
    if missing:
        raise VulkanError(
            f"{len(missing)} shader(s) have no SPIR-V, starting with "
            f"{missing[0]}, and no GLSL compiler is installed to build them. "
            "Run 'make shaders' where glslangValidator is available."
        )
    report(
        f"{len(stale)} shader source(s) are newer than the SPIR-V shipped "
        f"with them, starting with {stale[0]}, and there is no GLSL compiler "
        "to rebuild them. Using the shipped SPIR-V as-is."
    )
    _shader_dir_resolved = SHADER_DIR
    return SHADER_DIR


def shader_code(name):
    path = shader_directory() / f"{name}.spv"
    try:
        return path.read_bytes()
    except OSError as e:
        raise VulkanError(f"Could not read the shader {path}: {e}.") from e


_instance = None
_devices = {}


def _api_version(vk):
    return vk.VK_MAKE_VERSION(1, 1, 0)


def _get_instance():
    global _instance
    if _instance is not None:
        return _instance
    vk = _vk()
    application = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="animus",
        applicationVersion=1,
        pEngineName="animus",
        engineVersion=1,
        apiVersion=_api_version(vk),
    )
    layers = []
    if os.environ.get("ANIMUS_VULKAN_VALIDATE", "").lower() in _TRUTHY:
        try:
            available = {
                layer.layerName for layer in vk.vkEnumerateInstanceLayerProperties()
            }
        except Exception:  # noqa: BLE001
            available = set()
        if "VK_LAYER_KHRONOS_validation" in available:
            layers.append("VK_LAYER_KHRONOS_validation")
            print("Vulkan validation layer enabled.")
        else:
            print(
                "ANIMUS_VULKAN_VALIDATE is set but the validation layer is not installed."
            )
    info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=application,
        enabledLayerCount=len(layers),
        ppEnabledLayerNames=layers or None,
    )
    try:
        _instance = vk.vkCreateInstance(info, None)
    except Exception as e:  # noqa: BLE001
        raise VulkanUnavailable(f"Vulkan could not create an instance ({e}).") from e
    return _instance


def _physical_devices():
    vk = _vk()
    instance = _get_instance()
    try:
        return list(vk.vkEnumeratePhysicalDevices(instance))
    except Exception as e:  # noqa: BLE001
        raise VulkanUnavailable(f"Vulkan found no devices ({e}).") from e


def vulkan_devices():
    try:
        vk = _vk()
        physical = _physical_devices()
    except VulkanUnavailable:
        return []
    devices = []
    for index, handle in enumerate(physical):
        try:
            properties = vk.vkGetPhysicalDeviceProperties(handle)
            name = properties.deviceName
            kind = DEVICE_KINDS.get(properties.deviceType, "other")
        except Exception:  # noqa: BLE001
            name, kind = f"device {index}", "other"
        label = f"Vulkan - {name} [{kind}]"
        devices.append((f"vulkan:{index}", label, kind))
    return devices


def preferred_vulkan_device(allow_cpu=False):
    found = vulkan_devices()
    wanted = ["discrete", "integrated", "virtual", "other"]
    if allow_cpu:
        wanted.append("cpu")
    for kind in wanted:
        for ident, _label, found_kind in found:
            if found_kind == kind:
                return ident
    return None


def device_index(ident):
    if isinstance(ident, int):
        return ident
    ident = str(ident).strip().lower()
    if ident.startswith("vulkan:"):
        ident = ident[len("vulkan:") :]
    try:
        return int(ident)
    except ValueError as e:
        raise VulkanError(f"{ident!r} is not a Vulkan device index.") from e


def get_device(ident=None):
    if ident is None:
        ident = preferred_vulkan_device(allow_cpu=True)
        if ident is None:
            raise VulkanUnavailable("No Vulkan device was found.")
    index = device_index(ident)
    device = _devices.get(index)
    if device is None:
        device = Device(index)
        _devices[index] = device
    return device


class Buffer:
    _next_id = 1

    def __init__(
        self,
        device,
        size,
        allocation,
        host_visible,
        device_local,
        memory_type,
        handle,
        memory,
        mapped,
    ):
        self.device = device
        self.size = size
        self.allocation = allocation
        self.host_visible = host_visible
        self.device_local = device_local
        self.memory_type = memory_type
        self.handle = handle
        self.memory = memory
        self.mapped = mapped
        self.ident = Buffer._next_id
        Buffer._next_id += 1
        self.freed = False

    def view(self, offset=0, size=None):
        if size is None:
            size = self.size - offset
        return View(self, offset, size)

    def free(self):
        if not self.freed:
            self.device._free_buffer(self)
            self.freed = True


class View:
    __slots__ = ("buffer", "offset", "size")

    def __init__(self, buffer, offset, size):
        if offset < 0 or size < 0 or offset + size > buffer.size:
            raise VulkanError(
                f"A view of {size} bytes at {offset} does not fit a buffer of "
                f"{buffer.size} bytes."
            )
        self.buffer = buffer
        self.offset = offset
        self.size = size

    def key(self):
        return (self.buffer.ident, self.offset, self.size)

    def sub(self, offset, size):
        return View(self.buffer, self.offset + offset, size)


class Chunk:
    def __init__(self, buffer):
        self.buffer = buffer
        self.used = 0

    def take(self, size):
        alignment = self.buffer.device.offset_alignment
        start = (self.used + alignment - 1) // alignment * alignment
        if start + size > self.buffer.size:
            return None
        self.used = start + size
        return View(self.buffer, start, size)


class Arena:
    def __init__(self, device, chunk_bytes=CHUNK_BYTES, prefer_device=True):
        self.device = device
        self.chunk_bytes = chunk_bytes
        self.prefer_device = prefer_device
        self.chunks = []
        self.spilled = 0
        self.total = 0
        self.remaining = 0

    def plan(self, total_bytes):
        self.remaining = int(total_bytes)

    def allocate(self, size):
        size = int(size)
        for chunk in self.chunks:
            view = chunk.take(size)
            if view is not None:
                return view
        planned = self.remaining + self.device.offset_alignment
        capacity = max(size, min(self.chunk_bytes, planned))
        prefer = self.prefer_device and self.device.fits(capacity)
        buffer = self.device.create_buffer(capacity, prefer_device=prefer)
        if self.prefer_device and not buffer.device_local:
            self.spilled += capacity
        chunk = Chunk(buffer)
        self.chunks.append(chunk)
        self.total += capacity
        view = chunk.take(size)
        if view is None:
            raise VulkanError(f"Could not place {size} bytes in a new chunk.")
        self.remaining = max(0, self.remaining - size)
        return view

    def upload(self, array):
        array = numpy.ascontiguousarray(array)
        view = self.allocate(max(array.nbytes, 4))
        self.device.upload(view, array)
        return view

    def free(self):
        for chunk in self.chunks:
            chunk.buffer.free()
        self.chunks = []
        self.total = 0
        self.spilled = 0


class Pipeline:
    def __init__(self, name, bindings, handle, layout, set_layout):
        self.name = name
        self.bindings = bindings
        self.handle = handle
        self.layout = layout
        self.set_layout = set_layout


class Device:
    def __init__(self, index):
        vk = _vk()
        self.vk = vk
        self.index = index
        physical = _physical_devices()
        if not 0 <= index < len(physical):
            raise VulkanError(
                f"There is no Vulkan device {index}; {len(physical)} were found."
            )
        self.physical = physical[index]
        properties = vk.vkGetPhysicalDeviceProperties(self.physical)
        self.name = properties.deviceName
        self.kind = DEVICE_KINDS.get(properties.deviceType, "other")
        limits = properties.limits
        self.max_storage_range = limits.maxStorageBufferRange
        self.max_workgroups = tuple(limits.maxComputeWorkGroupCount)
        self.max_push = limits.maxPushConstantsSize
        self.offset_alignment = max(
            ALIGNMENT, int(limits.minStorageBufferOffsetAlignment)
        )
        self.large_tiles = limits.maxComputeSharedMemorySize >= LARGE_TILE_SHARED_BYTES
        self.api_version = properties.apiVersion
        self.reserve = 0
        if self.api_version < _api_version(vk):
            raise VulkanError(f"{self.name} only speaks Vulkan 1.0; 1.1 is needed.")

        if self.max_push < PUSH_CONSTANT_BYTES:
            raise VulkanError(
                f"{self.name} only allows {self.max_push} bytes of push "
                f"constants, and the kernels need {PUSH_CONSTANT_BYTES}."
            )

        storage16 = vk.VkPhysicalDevice16BitStorageFeatures(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
        )
        features = vk.VkPhysicalDeviceFeatures2(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=storage16
        )
        holder = vk.ffi.new("VkPhysicalDeviceFeatures2*", features)
        vk.vkGetPhysicalDeviceFeatures2(self.physical, holder)
        if not storage16.storageBuffer16BitAccess:
            raise VulkanError(
                f"{self.name} cannot read 16-bit values from buffers "
                "(storageBuffer16BitAccess), which the float16 weights need."
            )

        try:
            extensions = {
                e.extensionName
                for e in vk.vkEnumerateDeviceExtensionProperties(self.physical, None)
            }
        except Exception:  # noqa: BLE001
            extensions = set()
        enabled = [name for name in ("VK_EXT_memory_budget",) if name in extensions]
        self.has_budget = "VK_EXT_memory_budget" in enabled

        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical)
        self.queue_family = None
        for family_index, family in enumerate(families):
            if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                if self.queue_family is None or not (
                    family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
                ):
                    self.queue_family = family_index
                    if not family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                        break
        if self.queue_family is None:
            raise VulkanError(f"{self.name} has no compute queue.")

        wanted16 = vk.VkPhysicalDevice16BitStorageFeatures(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
            storageBuffer16BitAccess=True,
        )
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=wanted16,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(enabled),
            ppEnabledExtensionNames=enabled or None,
        )
        self.handle = vk.vkCreateDevice(self.physical, device_info, None)
        self.queue = vk.vkGetDeviceQueue(self.handle, self.queue_family, 0)

        self.memory = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)
        self.memory_types = [
            (
                self.memory.memoryTypes[i].propertyFlags,
                self.memory.memoryTypes[i].heapIndex,
            )
            for i in range(self.memory.memoryTypeCount)
        ]
        self.heap_sizes = [
            self.memory.memoryHeaps[i].size for i in range(self.memory.memoryHeapCount)
        ]
        local_flag = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        host_flag = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        self.unified = all(
            flags & host_flag
            for flags, _heap in self.memory_types
            if flags & local_flag
        )
        self.device_local_bytes = sum(
            self.memory.memoryHeaps[i].size
            for i in range(self.memory.memoryHeapCount)
            if self.memory.memoryHeaps[i].flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
        )

        self.command_pool = vk.vkCreateCommandPool(
            self.handle,
            vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex=self.queue_family,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            ),
            None,
        )
        self.command_buffer = vk.vkAllocateCommandBuffers(
            self.handle,
            vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        self.fence = vk.vkCreateFence(
            self.handle,
            vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO),
            None,
        )
        self._recording = False
        self._recorded = 0
        self._pending = False
        self._barrier = vk.VkMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT
            | vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
            | vk.VK_ACCESS_SHADER_WRITE_BIT
            | vk.VK_ACCESS_TRANSFER_READ_BIT
            | vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        )
        self._stages = (
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.VK_PIPELINE_STAGE_TRANSFER_BIT
        )

        self.pipelines = {}
        self.descriptor_pools = []
        self._pool_sets_left = 0
        self._dead_sets = 0
        self.descriptor_sets = {}
        self.buffers = {}
        self.allocated_bytes = 0
        self.staging = None
        self.dispatches = 0
        self.submissions = 0

    def describe(self):
        return (
            f"{self.name} [{self.kind}], Vulkan {self.api_version >> 22}."
            f"{(self.api_version >> 12) & 0x3FF}, {format_bytes(self.device_local_bytes)} "
            f"device memory{' (shared with the host)' if self.unified else ''}"
        )

    def budget(self):
        vk = self.vk
        total = self.device_local_bytes
        if self.has_budget:
            try:
                budget = vk.VkPhysicalDeviceMemoryBudgetPropertiesEXT(
                    sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
                )
                properties = vk.VkPhysicalDeviceMemoryProperties2(
                    sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
                    pNext=budget,
                )
                holder = vk.ffi.new("VkPhysicalDeviceMemoryProperties2*", properties)
                vk.vkGetPhysicalDeviceMemoryProperties2(self.physical, holder)
                free = 0
                for i in range(self.memory.memoryHeapCount):
                    if (
                        self.memory.memoryHeaps[i].flags
                        & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                    ):
                        free += max(0, budget.heapBudget[i] - budget.heapUsage[i])
                return free, total
            except Exception:  # noqa: BLE001, S110
                pass
        return max(0, total - self.allocated_bytes), total

    def margin(self):
        """Device memory left alone so the driver never has to evict."""
        if HEAP_MARGIN_BYTES is not None:
            return HEAP_MARGIN_BYTES
        return int(self.device_local_bytes * HEAP_MARGIN_FRACTION)

    def fits(self, nbytes):
        """Whether nbytes fit in device memory with the reserve and margin left."""
        if self.unified:
            return True
        free, _total = self.budget()
        return free - self.reserve - self.margin() >= nbytes

    def _find_memory_type(self, allowed, required, forbidden=0):
        for index, (flags, _heap) in enumerate(self.memory_types):
            if not (allowed >> index) & 1:
                continue
            if (flags & required) == required and not (flags & forbidden):
                return index
        return None

    def create_buffer(self, size, prefer_device=True, cached=False):
        vk = self.vk
        size = max(int(size), 4)
        usage = (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )
        handle = vk.vkCreateBuffer(
            self.handle,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size,
                usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.handle, handle)
        allowed = requirements.memoryTypeBits
        local = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        visible = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        coherent = vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        hostcached = vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT

        candidates = []
        if prefer_device:
            candidates.append(self._find_memory_type(allowed, local, visible))
            candidates.append(
                self._find_memory_type(allowed, local | visible | coherent)
            )
        if cached:
            candidates.append(
                self._find_memory_type(allowed, visible | coherent | hostcached, local)
            )
        candidates.append(self._find_memory_type(allowed, visible | coherent, local))
        candidates.append(self._find_memory_type(allowed, visible | coherent))
        candidates = list(dict.fromkeys(c for c in candidates if c is not None))
        if not candidates:
            vk.vkDestroyBuffer(self.handle, handle, None)
            raise VulkanError(f"{self.name} has no memory type for a buffer.")

        memory = None
        error = None
        for memory_type in candidates:
            try:
                memory = vk.vkAllocateMemory(
                    self.handle,
                    vk.VkMemoryAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        allocationSize=requirements.size,
                        memoryTypeIndex=memory_type,
                    ),
                    None,
                )
                break
            except Exception as e:  # noqa: BLE001
                error = e
                memory = None
        if memory is None:
            vk.vkDestroyBuffer(self.handle, handle, None)
            raise VulkanError(
                f"{self.name} could not allocate {format_bytes(size)} ({error})."
            )
        vk.vkBindBufferMemory(self.handle, handle, memory, 0)
        flags = self.memory_types[memory_type][0]
        host_visible = bool(flags & visible)
        mapped = None
        if host_visible:
            mapped = vk.vkMapMemory(self.handle, memory, 0, requirements.size, 0)
        buffer = Buffer(
            self,
            size,
            requirements.size,
            host_visible,
            bool(flags & local),
            memory_type,
            handle,
            memory,
            mapped,
        )
        self.buffers[buffer.ident] = buffer
        if flags & local:
            self.allocated_bytes += requirements.size
        return buffer

    def _free_buffer(self, buffer):
        vk = self.vk
        self.wait_idle()
        stale = [
            key
            for key in self.descriptor_sets
            if any(k[0] == buffer.ident for k in key[1])
        ]
        for key in stale:
            del self.descriptor_sets[key]
        self._dead_sets += len(stale)
        if self._dead_sets >= DESCRIPTOR_SETS_PER_POOL:
            self._reset_descriptor_pools()
        if buffer.mapped is not None:
            vk.vkUnmapMemory(self.handle, buffer.memory)
            buffer.mapped = None
        vk.vkDestroyBuffer(self.handle, buffer.handle, None)
        vk.vkFreeMemory(self.handle, buffer.memory, None)
        if (
            self.memory_types[buffer.memory_type][0]
            & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        ):
            self.allocated_bytes -= buffer.allocation
        self.buffers.pop(buffer.ident, None)

    def _staging(self):
        if self.staging is None:
            self.staging = self.create_buffer(
                STAGING_BYTES, prefer_device=False, cached=True
            )
        return self.staging

    def _copy(self, source, source_offset, target, target_offset, size):
        vk = self.vk
        self.flush()
        self.begin()
        vk.vkCmdCopyBuffer(
            self.command_buffer,
            source.handle,
            target.handle,
            1,
            [
                vk.VkBufferCopy(
                    srcOffset=source_offset, dstOffset=target_offset, size=size
                )
            ],
        )
        self._memory_barrier()
        self._recorded += 1
        self.flush()

    def upload(self, view, data):
        """Copy bytes or an ndarray into a view."""
        if isinstance(data, numpy.ndarray):
            data = numpy.ascontiguousarray(data)
            raw = memoryview(data).cast("B")
        else:
            raw = memoryview(data).cast("B")
        size = raw.nbytes
        if size > view.size:
            raise VulkanError(f"{size} bytes do not fit a view of {view.size} bytes.")
        buffer = view.buffer
        if buffer.host_visible:
            self.flush()
            target = numpy.frombuffer(buffer.mapped, dtype=numpy.uint8)
            target[view.offset : view.offset + size] = numpy.frombuffer(
                raw, dtype=numpy.uint8
            )
            return
        staging = self._staging()
        stage = numpy.frombuffer(staging.mapped, dtype=numpy.uint8)
        source = numpy.frombuffer(raw, dtype=numpy.uint8)
        done = 0
        while done < size:
            piece = min(staging.size, size - done)
            stage[:piece] = source[done : done + piece]
            self._copy(staging, 0, buffer, view.offset + done, piece)
            done += piece

    def download(self, view, dtype=numpy.float32, count=None):
        itemsize = numpy.dtype(dtype).itemsize
        size = view.size if count is None else count * itemsize
        buffer = view.buffer
        self.flush()
        if buffer.host_visible:
            source = numpy.frombuffer(buffer.mapped, dtype=numpy.uint8)
            raw = source[view.offset : view.offset + size].copy()
            return raw.view(dtype)
        staging = self._staging()
        out = numpy.empty(size, dtype=numpy.uint8)
        stage = numpy.frombuffer(staging.mapped, dtype=numpy.uint8)
        done = 0
        while done < size:
            piece = min(staging.size, size - done)
            self._copy(buffer, view.offset + done, staging, 0, piece)
            out[done : done + piece] = stage[:piece]
            done += piece
        return out.view(dtype)

    def pipeline(self, name):
        pipeline = self.pipelines.get(name)
        if pipeline is not None:
            return pipeline
        vk = self.vk
        if name not in SHADER_VARIANTS:
            raise VulkanError(f"There is no shader called {name}.")
        if name.startswith(("dequant", "matmul")):
            family = name.split("_")[0]
        else:
            family = name
        bindings = SHADER_BINDINGS[family]
        code = shader_code(name)
        module = vk.vkCreateShaderModule(
            self.handle,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code),
                pCode=code,
            ),
            None,
        )
        layout_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for i in range(bindings)
        ]
        set_layout = vk.vkCreateDescriptorSetLayout(
            self.handle,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=bindings,
                pBindings=layout_bindings,
            ),
            None,
        )
        layout = vk.vkCreatePipelineLayout(
            self.handle,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                        offset=0,
                        size=PUSH_CONSTANT_BYTES,
                    )
                ],
            ),
            None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=module,
            pName="main",
        )
        handle = vk.vkCreateComputePipelines(
            self.handle,
            None,
            1,
            [
                vk.VkComputePipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    stage=stage,
                    layout=layout,
                )
            ],
            None,
        )[0]
        vk.vkDestroyShaderModule(self.handle, module, None)
        pipeline = Pipeline(name, bindings, handle, layout, set_layout)
        self.pipelines[name] = pipeline
        return pipeline

    def _new_descriptor_pool(self):
        vk = self.vk
        pool = vk.vkCreateDescriptorPool(
            self.handle,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=DESCRIPTOR_SETS_PER_POOL,
                poolSizeCount=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=DESCRIPTOR_SETS_PER_POOL * 4,
                    )
                ],
            ),
            None,
        )
        self.descriptor_pools.append(pool)
        self._pool_sets_left = DESCRIPTOR_SETS_PER_POOL
        return pool

    def _reset_descriptor_pools(self):
        vk = self.vk
        for pool in self.descriptor_pools[1:]:
            vk.vkDestroyDescriptorPool(self.handle, pool, None)
        self.descriptor_pools = self.descriptor_pools[:1]
        for pool in self.descriptor_pools:
            vk.vkResetDescriptorPool(self.handle, pool, 0)
        self._pool_sets_left = DESCRIPTOR_SETS_PER_POOL if self.descriptor_pools else 0
        self.descriptor_sets = {}
        self._dead_sets = 0

    def _descriptor_set(self, pipeline, views):
        key = (pipeline.name, tuple(view.key() for view in views))
        found = self.descriptor_sets.get(key)
        if found is not None:
            return found
        vk = self.vk
        if not self.descriptor_pools or self._pool_sets_left <= 0:
            self._new_descriptor_pool()
        pool = self.descriptor_pools[-1]
        descriptor_set = vk.vkAllocateDescriptorSets(
            self.handle,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=pool,
                descriptorSetCount=1,
                pSetLayouts=[pipeline.set_layout],
            ),
        )[0]
        self._pool_sets_left -= 1
        infos = [
            vk.VkDescriptorBufferInfo(
                buffer=view.buffer.handle, offset=view.offset, range=view.size
            )
            for view in views
        ]
        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=i,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[infos[i]],
            )
            for i in range(len(views))
        ]
        vk.vkUpdateDescriptorSets(self.handle, len(writes), writes, 0, None)
        self.descriptor_sets[key] = descriptor_set
        return descriptor_set

    def _wait(self):
        vk = self.vk
        try:
            vk.vkWaitForFences(self.handle, 1, [self.fence], True, FENCE_TIMEOUT_NS)
        except vk.VkTimeout as e:
            raise VulkanError(f"{self.name} did not finish its work in time.") from e
        finally:
            self._pending = False

    def begin(self):
        if self._recording:
            return
        if self._pending:
            self._wait()
        vk = self.vk
        vk.vkResetCommandBuffer(self.command_buffer, 0)
        vk.vkBeginCommandBuffer(
            self.command_buffer,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        self._recording = True
        self._recorded = 0

    def _memory_barrier(self):
        vk = self.vk
        vk.vkCmdPipelineBarrier(
            self.command_buffer,
            self._stages,
            self._stages,
            0,
            1,
            [self._barrier],
            0,
            None,
            0,
            None,
        )

    def dispatch(self, name, views, push, groups):
        """Record one compute dispatch, with a barrier after it."""
        vk = self.vk
        pipeline = self.pipeline(name)
        if len(views) != pipeline.bindings:
            raise VulkanError(
                f"{name} takes {pipeline.bindings} buffers, not {len(views)}."
            )
        for view in views:
            if view.size > self.max_storage_range:
                raise VulkanError(
                    f"{self.name} cannot bind {format_bytes(view.size)} at "
                    f"once (its limit is {format_bytes(self.max_storage_range)})."
                )
        x, y, z = groups
        if (
            x > self.max_workgroups[0]
            or y > self.max_workgroups[1]
            or z > self.max_workgroups[2]
        ):
            raise VulkanError(
                f"A dispatch of {groups} exceeds {self.name}'s limit of "
                f"{self.max_workgroups}."
            )
        if len(push) > PUSH_CONSTANT_BYTES:
            raise VulkanError(f"{len(push)} bytes of push constants is too many.")
        self.begin()
        descriptor_set = self._descriptor_set(pipeline, views)
        vk.vkCmdBindPipeline(
            self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle
        )
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout,
            0,
            1,
            [descriptor_set],
            0,
            None,
        )
        vk.vkCmdPushConstants(
            self.command_buffer,
            pipeline.layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            len(push),
            vk.ffi.from_buffer("char[]", push),
        )
        vk.vkCmdDispatch(self.command_buffer, x, y, z)
        self._memory_barrier()
        self._recorded += 1
        self.dispatches += 1

    def abort(self):
        """Drop whatever was recorded but not submitted."""
        if not self._recording:
            return
        try:
            self.vk.vkEndCommandBuffer(self.command_buffer)
        except Exception:  # noqa: BLE001, S110
            pass
        self._recording = False
        self._recorded = 0

    def flush(self):
        if not self._recording:
            return
        vk = self.vk
        vk.vkEndCommandBuffer(self.command_buffer)
        self._recording = False
        if not self._recorded:
            return
        if self._pending:
            self._wait()
        vk.vkResetFences(self.handle, 1, [self.fence])
        vk.vkQueueSubmit(
            self.queue,
            1,
            [
                vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[self.command_buffer],
                )
            ],
            self.fence,
        )
        self._pending = True
        self.submissions += 1
        self._wait()

    def wait_idle(self):
        self.flush()
        try:
            self.vk.vkQueueWaitIdle(self.queue)
            self._pending = False
        except Exception:  # noqa: BLE001, S110
            pass

    def dequantize(self, qtype, source, target, n_blocks, off_src=0, off_dst=0):
        """Unpack n_blocks of a ggml type from source (bytes) into float16."""
        name = f"dequant_{qtype.lower()}"
        if name not in SHADER_VARIANTS:
            raise VulkanError(f"There is no GPU dequantizer for {qtype}.")
        block, type_bytes = BLOCK_SIZES[qtype]
        done = 0
        while done < n_blocks:
            count = min(n_blocks - done, self.max_workgroups[0] * 64)
            push = struct.pack(
                "<III", count, off_src + done * type_bytes, off_dst + done * block
            )
            self.dispatch(name, [source, target], push, ((count + 63) // 64, 1, 1))
            done += count

    def matmul(
        self,
        a,
        b,
        c,
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        b_f16=True,
        b_kn=False,
        off_a=0,
        off_b=0,
        off_c=0,
        batch=1,
        batch_a=0,
        batch_b=0,
        batch_c=0,
        alpha=1.0,
        accumulate=False,
        bias=None,
        off_bias=0,
    ):
        """C[M, N] (+)= alpha * A[M, K] @ B (+ bias), batched over Z."""
        small = M < MATMUL_LARGE_TILE or N < MATMUL_LARGE_TILE or not self.large_tiles
        tile = MATMUL_SMALL_TILE if small else MATMUL_LARGE_TILE
        name = (
            f"matmul_{'f16' if b_f16 else 'f32'}_{'kn' if b_kn else 'nk'}_"
            f"{'small' if small else 'large'}"
        )
        flags = (MATMUL_ACCUMULATE if accumulate else 0) | (
            MATMUL_BIAS if bias is not None else 0
        )
        if bias is None:
            bias = a
        push = struct.pack(
            "<IIIIIIIIIIIIIIf",
            M,
            N,
            K,
            lda,
            ldb,
            ldc,
            off_a,
            off_b,
            off_c,
            batch_a,
            batch_b,
            batch_c,
            flags,
            off_bias,
            float(alpha),
        )
        groups = ((M + tile - 1) // tile, (N + tile - 1) // tile, batch)
        self.dispatch(name, [a, b, c, bias], push, groups)

    def norm(
        self,
        x,
        y,
        rows,
        dim,
        ldx,
        ldy,
        flags,
        eps,
        weight=None,
        modulation=None,
        off_x=0,
        off_y=0,
        off_w=0,
        off_s=0,
    ):
        weight = x if weight is None else weight
        modulation = x if modulation is None else modulation
        done = 0
        while done < rows:
            count = min(rows - done, self.max_workgroups[0])
            push = struct.pack(
                "<IIIIIIIIIf",
                count,
                dim,
                ldx,
                ldy,
                off_x + done * ldx,
                off_y + done * ldy,
                off_w,
                off_s,
                flags,
                float(eps),
            )
            self.dispatch("norm", [x, y, weight, modulation], push, (count, 1, 1))
            done += count

    def head_norm_rope(
        self,
        x,
        tokens,
        heads,
        head_dim,
        ld,
        flags,
        eps,
        weight=None,
        cos=None,
        sin=None,
        off=0,
        off_w=0,
        off_cos=0,
        off_sin=0,
        ld_rope=0,
    ):
        weight = x if weight is None else weight
        cos = x if cos is None else cos
        sin = x if sin is None else sin
        done = 0
        while done < tokens:
            count = min(tokens - done, self.max_workgroups[0])
            push = struct.pack(
                "<IIIIIIIIIIf",
                count,
                heads,
                head_dim,
                ld,
                off + done * ld,
                off_w,
                off_cos + done * ld_rope,
                off_sin + done * ld_rope,
                ld_rope,
                flags,
                float(eps),
            )
            self.dispatch(
                "head_norm_rope", [x, weight, cos, sin], push, (count, heads, 1)
            )
            done += count

    def softmax(self, x, rows, cols, ld, off=0, batch=1, batch_stride=0, scale=1.0):
        done = 0
        while done < rows:
            count = min(rows - done, self.max_workgroups[0])
            push = struct.pack(
                "<IIIIIf", count, cols, ld, off + done * ld, batch_stride, float(scale)
            )
            self.dispatch("softmax", [x], push, (count, batch, 1))
            done += count

    def silu_mul(self, a, b, n, off_a=0, off_b=0):
        push = struct.pack("<III", n, off_a, off_b)
        groups = min((n + 255) // 256, 4096)
        self.dispatch("silu_mul", [a, b], push, (max(groups, 1), 1, 1))

    def im2col(
        self, x, a, H, W, C, OH, OW, ksize, upsample, row0, rows, off_in=0, off_out=0
    ):
        push = struct.pack(
            "<IIIIIIIIIII",
            H,
            W,
            C,
            OH,
            OW,
            ksize,
            1 if upsample else 0,
            row0,
            rows,
            off_in,
            off_out,
        )
        total = rows * OW * ksize * ksize * C
        groups = min((total + 255) // 256, 4096)
        self.dispatch("im2col", [x, a], push, (max(groups, 1), 1, 1))

    def groupnorm(
        self, x, y, P, C, G, eps, gamma, beta, partials, stats, silu, off_x=0, off_y=0
    ):
        if C > GN_MAX_CHANNELS or C % G:
            raise VulkanError(
                f"GroupNorm over {C} channels in {G} groups is not supported."
            )
        n_blocks = (P + GN_BLOCK_PIXELS - 1) // GN_BLOCK_PIXELS
        if n_blocks > self.max_workgroups[0]:
            raise VulkanError(f"{P} pixels is too many for one GroupNorm pass.")
        count = P * (C // G)
        for pass_ in (0, 1):
            push = struct.pack(
                "<IIIIIIII", P, C, G, GN_BLOCK_PIXELS, pass_, off_x, 0, 0
            )
            self.dispatch("gn_stats", [x, stats, partials], push, (n_blocks, 1, 1))
            push = struct.pack(
                "<IIIIIIf", n_blocks, G, count, pass_, 0, pass_ * G, float(eps)
            )
            self.dispatch("gn_finalize", [partials, stats], push, (1, 1, 1))
        push = struct.pack(
            "<IIIIIIIII", P, C, G, 1 if silu else 0, off_x, off_y, 0, 0, 0
        )
        groups = min((P * C + 255) // 256, 4096)
        self.dispatch(
            "gn_apply", [x, y, stats, gamma, beta], push, (max(groups, 1), 1, 1)
        )

    def close(self):
        vk = self.vk
        self.wait_idle()
        for buffer in list(self.buffers.values()):
            buffer.free()
        self.staging = None
        for pool in self.descriptor_pools:
            vk.vkDestroyDescriptorPool(self.handle, pool, None)
        self.descriptor_pools = []
        self.descriptor_sets = {}
        for pipeline in self.pipelines.values():
            vk.vkDestroyPipeline(self.handle, pipeline.handle, None)
            vk.vkDestroyPipelineLayout(self.handle, pipeline.layout, None)
            vk.vkDestroyDescriptorSetLayout(self.handle, pipeline.set_layout, None)
        self.pipelines = {}


_GGML_NAMES = None


def ggml_type_name(value):
    global _GGML_NAMES
    if isinstance(value, str):
        return value.upper()
    if _GGML_NAMES is None:
        try:
            from gguf import GGMLQuantizationType

            _GGML_NAMES = {
                int(member.value): member.name for member in GGMLQuantizationType
            }
        except ImportError:
            _GGML_NAMES = {}
    name = getattr(value, "name", None)
    if name:
        return name
    return _GGML_NAMES.get(int(value), str(value))


def quantize_q8_0(matrix):
    matrix = numpy.ascontiguousarray(matrix, dtype=numpy.float32)
    rows, cols = matrix.shape
    if cols % 32:
        raise VulkanError(f"A row of {cols} does not split into Q8_0 blocks of 32.")
    blocks = matrix.reshape(rows * cols // 32, 32)
    amax = numpy.abs(blocks).max(axis=1)
    scale = amax / 127.0
    inv = numpy.where(scale > 0, 1.0 / numpy.where(scale > 0, scale, 1.0), 0.0)
    q = numpy.rint(blocks * inv[:, None]).clip(-128, 127).astype(numpy.int8)
    out = numpy.empty((blocks.shape[0], 34), dtype=numpy.uint8)
    out[:, :2] = scale.astype(numpy.float16).view(numpy.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(numpy.uint8)
    return out.reshape(rows, cols // 32 * 34)


def dequantize_rows(raw, qtype, shape):
    qtype = ggml_type_name(qtype)
    if qtype == "F32":
        return numpy.ascontiguousarray(raw).view(numpy.float32).reshape(shape)
    if qtype == "F16":
        return (
            numpy.ascontiguousarray(raw)
            .view(numpy.float16)
            .astype(numpy.float32)
            .reshape(shape)
        )
    if qtype == "BF16":
        bits = (
            numpy.ascontiguousarray(raw).view(numpy.uint16).astype(numpy.uint32) << 16
        )
        return bits.view(numpy.float32).reshape(shape)
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    member = GGMLQuantizationType[qtype]
    return dequantize(numpy.ascontiguousarray(raw).view(numpy.uint8), member).reshape(
        shape
    )


def bf16_to_f16(raw):
    bits = numpy.ascontiguousarray(raw).view(numpy.uint16).astype(numpy.uint32) << 16
    return bits.view(numpy.float32).astype(numpy.float16)


class Tensor:
    def __init__(self, name, qtype, shape, loader, nbytes):
        self.name = name
        self.qtype = ggml_type_name(qtype)
        self.shape = tuple(int(d) for d in shape)
        self._loader = loader
        self.nbytes = int(nbytes)

    def raw(self):
        return self._loader()

    def numel(self):
        count = 1
        for d in self.shape:
            count *= d
        return count

    def float32(self):
        return dequantize_rows(self.raw(), self.qtype, self.shape)

    def __repr__(self):
        return f"Tensor({self.name}, {self.qtype}, {self.shape})"


Z_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.", "net.")
Z_NAME_ALIASES = (
    ("x_embedder.", "all_x_embedder.2-1."),
    ("final_layer.", "all_final_layer.2-1."),
    (".attention.q_norm.", ".attention.norm_q."),
    (".attention.k_norm.", ".attention.norm_k."),
    (".attention.out.", ".attention.to_out.0."),
)


def normalize_key(name):
    for prefix in Z_STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for source, target in Z_NAME_ALIASES:
        if source.startswith("."):
            name = name.replace(source, target)
        elif name.startswith(source):
            name = target + name[len(source) :]
    return name


def read_gguf(path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    tensors = {}
    for entry in reader.tensors:
        name = normalize_key(entry.name)
        shape = tuple(reversed([int(d) for d in entry.shape.tolist()]))
        data = entry.data

        def load(data=data):
            return numpy.ascontiguousarray(data).reshape(-1).view(numpy.uint8)

        tensors[name] = Tensor(name, entry.tensor_type, shape, load, data.nbytes)
    return tensors


def read_safetensors(path):
    try:
        import torch
        from safetensors import safe_open
    except ImportError as e:
        raise VulkanError(
            f"Reading safetensors needs torch and safetensors ({e})."
        ) from e

    tensors = {}
    handle = safe_open(str(path), framework="pt")
    for key in handle.keys():
        name = normalize_key(key)

        def load(key=key, handle=handle):
            tensor = handle.get_tensor(key)
            if tensor.dtype == torch.bfloat16 or tensor.dtype == torch.float16:
                if tensor.dim() <= 1:
                    return (
                        tensor.to(torch.float32)
                        .contiguous()
                        .numpy()
                        .view(numpy.uint8)
                        .reshape(-1)
                    )
                return (
                    tensor.to(torch.float16)
                    .contiguous()
                    .numpy()
                    .view(numpy.uint8)
                    .reshape(-1)
                )
            return (
                tensor.to(torch.float32)
                .contiguous()
                .numpy()
                .view(numpy.uint8)
                .reshape(-1)
            )

        slice_ = handle.get_slice(key)
        shape = tuple(slice_.get_shape())
        dtype = str(slice_.get_dtype()).upper()
        if dtype in ("BF16", "F16") and len(shape) > 1:
            qtype, itemsize = "F16", 2
        else:
            qtype, itemsize = "F32", 4
        count = 1
        for d in shape:
            count *= d
        tensors[name] = Tensor(name, qtype, shape, load, count * itemsize)
    return tensors


def tensors_from_arrays(arrays, quantize=None):
    tensors = {}
    for key, value in arrays.items():
        array = numpy.ascontiguousarray(value, dtype=numpy.float32)
        name = normalize_key(key)
        if array.ndim == 2 and quantize == "Q8_0" and array.shape[1] % 32 == 0:
            raw = quantize_q8_0(array).reshape(-1)
            qtype = "Q8_0"
        elif array.ndim == 2:
            raw = array.astype(numpy.float16).reshape(-1).view(numpy.uint8)
            qtype = "F16"
        else:
            raw = array.reshape(-1).view(numpy.uint8)
            qtype = "F32"
        tensors[name] = Tensor(
            name, qtype, array.shape, lambda raw=raw: raw, raw.nbytes
        )
    return tensors


def read_checkpoint(path):
    path = Path(path)
    if path.suffix.lower() == ".gguf":
        return read_gguf(path)
    return read_safetensors(path)


def read_checkpoints(paths):
    tensors = {}
    for path in paths:
        tensors.update(read_checkpoint(path))
    return tensors


def _count(tensors, prefix):
    best = -1
    for name in tensors:
        if not name.startswith(prefix):
            continue
        piece = name[len(prefix) :].split(".", 1)[0]
        if piece.isdigit():
            best = max(best, int(piece))
    return best + 1


def detect_config(tensors, shape_of=None):
    if shape_of is None:

        def shape_of(name):
            tensor = tensors.get(name)
            return None if tensor is None else tensor.shape

    x_embed = shape_of("all_x_embedder.2-1.weight")
    if x_embed is None:
        raise VulkanError(
            "This file has no all_x_embedder.2-1.weight, so it is not a "
            "Z-Image diffusion transformer."
        )
    dim, patch_features = int(x_embed[0]), int(x_embed[1])
    patch_size, f_patch_size = 2, 1
    in_channels = patch_features // (patch_size * patch_size * f_patch_size)
    cap = shape_of("cap_embedder.1.weight")
    cap_feat_dim = int(cap[1]) if cap else 2560
    n_layers = _count(tensors, "layers.") or 30
    n_refiner_layers = (
        max(_count(tensors, "noise_refiner."), _count(tensors, "context_refiner.")) or 2
    )
    q_norm = shape_of("layers.0.attention.norm_q.weight")
    if q_norm:
        head_dim = int(q_norm[0])
        n_heads = dim // head_dim
    else:
        head_dim, n_heads = 128, dim // 128
    to_k = shape_of("layers.0.attention.to_k.weight")
    qkv = shape_of("layers.0.attention.qkv.weight")
    if to_k:
        n_kv_heads = int(to_k[0]) // head_dim
    elif qkv:
        n_kv_heads = max(1, (int(qkv[0]) - dim) // (2 * head_dim))
    else:
        n_kv_heads = n_heads
    w1 = shape_of("layers.0.feed_forward.w1.weight")
    ffn_hidden = int(w1[0]) if w1 else int(dim / 3 * 8)
    t_mid = shape_of("t_embedder.mlp.0.weight")
    t_embedder_mid = int(t_mid[0]) if t_mid else 1024
    return dict(
        in_channels=in_channels,
        dim=dim,
        n_layers=n_layers,
        n_refiner_layers=n_refiner_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        cap_feat_dim=cap_feat_dim,
        patch_size=patch_size,
        f_patch_size=f_patch_size,
        ffn_hidden=ffn_hidden,
        t_embedder_mid=t_embedder_mid,
        head_dim=head_dim,
    )


Z_SEQ_MULTIPLE = 32
Z_ADALN_EMBED_DIM = 256
Z_FREQUENCY_EMBEDDING_SIZE = 256
Z_MAX_PERIOD = 10000
Z_ROPE_THETA = 256.0
Z_ROPE_AXES_DIMS = (32, 48, 48)
Z_ROPE_AXES_LENS = (1536, 512, 512)
Z_NORM_EPS = 1e-5
Z_FINAL_NORM_EPS = 1e-6
Z_T_SCALE = 1000.0


class Linear:
    def __init__(self, name, qtype, out_features, in_features, view, bias=None):
        self.name = name
        self.qtype = qtype
        self.out_features = out_features
        self.in_features = in_features
        self.view = view
        self.bias = bias
        self.loras = {}

    @property
    def needs_dequant(self):
        return self.qtype != "F16"

    def scratch_elements(self):
        return self.out_features * self.in_features if self.needs_dequant else 0


class Lora:
    def __init__(self, down, up, rank, scale, factor=1.0, col_offset=0):
        self.down = down
        self.up = up
        self.rank = rank
        self.factor = float(factor)
        self.scale = float(scale) * self.factor
        self.col_offset = col_offset


def rope_tables(ids, theta=Z_ROPE_THETA, axes_dims=Z_ROPE_AXES_DIMS):
    """cos, sin [tokens, sum(axes_dims) // 2] for integer ids [tokens, 3],
    matching ZRopeEmbedder in animus.py."""
    cos, sin = [], []
    for axis, dim in enumerate(axes_dims):
        freqs = 1.0 / (theta ** (numpy.arange(0, dim, 2, dtype=numpy.float64) / dim))
        angles = (ids[:, axis].astype(numpy.float64)[:, None] * freqs[None, :]).astype(
            numpy.float32
        )
        cos.append(numpy.cos(angles))
        sin.append(numpy.sin(angles))
    return (
        numpy.ascontiguousarray(numpy.concatenate(cos, axis=1), dtype=numpy.float32),
        numpy.ascontiguousarray(numpy.concatenate(sin, axis=1), dtype=numpy.float32),
    )


def coordinate_grid(size, start):
    axes = [
        numpy.arange(x0, x0 + span, dtype=numpy.int64) for x0, span in zip(start, size)
    ]
    grid = numpy.stack(numpy.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def timestep_embedding(t, dim=Z_FREQUENCY_EMBEDDING_SIZE, max_period=Z_MAX_PERIOD):
    half = dim // 2
    freqs = numpy.exp(
        -math.log(max_period) * numpy.arange(0, half, dtype=numpy.float32) / half
    )
    args = numpy.float32(t) * freqs
    return numpy.concatenate([numpy.cos(args), numpy.sin(args)]).astype(numpy.float32)


def silu(x):
    return x / (1.0 + numpy.exp(-x))


def pad_rows(count, multiple=Z_SEQ_MULTIPLE):
    return (-count) % multiple


class VulkanZImage:
    def __init__(self, device, tensors, config=None, progress=None, stop_check=None):
        self.device = device
        self.progress = progress or (lambda message: None)
        self.stop_check = stop_check or (lambda: False)
        self.config = config or detect_config(tensors)
        cfg = self.config
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["in_channels"]
        self.dim = cfg["dim"]
        self.patch_size = cfg["patch_size"]
        self.f_patch_size = cfg["f_patch_size"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg["n_kv_heads"]
        self.head_dim = cfg["head_dim"]
        self.ffn_hidden = cfg["ffn_hidden"]
        self.cap_feat_dim = cfg["cap_feat_dim"]
        self.n_layers = cfg["n_layers"]
        self.n_refiner_layers = cfg["n_refiner_layers"]
        self.t_scale = Z_T_SCALE
        self.key = f"{self.patch_size}-{self.f_patch_size}"
        if self.head_dim != sum(Z_ROPE_AXES_DIMS):
            raise VulkanError(
                f"A head is {self.head_dim} wide but the rotary axes come to "
                f"{sum(Z_ROPE_AXES_DIMS)}. This is not a Z-Image DiT."
            )
        if self.n_heads % self.n_kv_heads:
            raise VulkanError(
                f"{self.n_heads} query heads do not divide into "
                f"{self.n_kv_heads} key heads."
            )
        self.kv_groups = self.n_heads // self.n_kv_heads
        self.patch_features = (
            self.f_patch_size * self.patch_size * self.patch_size * self.in_channels
        )

        self.weights = Arena(device)
        self.small = Arena(device, chunk_bytes=16 * 1024 * 1024)
        self.linears = {}
        self.lora_targets = {}
        self.activations = {}
        self._activation_tokens = None
        self.scratch = None
        self.scratch_elements = 0
        self.lora_tmp = None
        self.lora_tmp_rank = 0
        self.uploaded_bytes = 0
        self.qtypes = {}
        self.loaded = False
        try:
            self._load(tensors)
        except BaseException:
            self.release()
            raise
        self.loaded = True

    def _vector(self, tensors, name, expected=None, required=True):
        tensor = tensors.get(name)
        if tensor is None:
            if required:
                raise VulkanError(f"This checkpoint is missing {name}.")
            return None
        values = tensor.float32().reshape(-1)
        if expected is not None and values.size != expected:
            raise VulkanError(
                f"{name} has {values.size} values but the DiT wants {expected}."
            )
        return self.small.upload(values)

    def _cpu_matrix(self, tensors, name, expected):
        tensor = tensors.get(name)
        if tensor is None:
            raise VulkanError(f"This checkpoint is missing {name}.")
        values = tensor.float32().reshape(-1)
        wanted = expected[0] * expected[1]
        if values.size != wanted:
            raise VulkanError(
                f"{name} is {tensor.shape} in this checkpoint but the DiT wants {expected}."
            )
        return values.reshape(expected)

    def _linear(
        self, tensors, name, out_features, in_features, bias=False, required=True
    ):
        weight = tensors.get(f"{name}.weight")
        if weight is None:
            if required:
                raise VulkanError(f"This checkpoint is missing {name}.weight.")
            return None
        shape = (
            tuple(d for d in weight.shape if d != 1)
            if len(weight.shape) > 2
            else weight.shape
        )
        if tuple(shape) != (out_features, in_features):
            raise VulkanError(
                f"{name}.weight is {weight.shape} in this checkpoint but the "
                f"DiT wants {(out_features, in_features)}. This file is not "
                "the model it claims."
            )
        qtype = weight.qtype
        raw = weight.raw()
        if qtype == "F32":
            raw = raw.view(numpy.float32).astype(numpy.float16)
            qtype = "F16"
        elif qtype == "BF16":
            raw = bf16_to_f16(raw)
            qtype = "F16"
        elif qtype != "F16" and f"dequant_{qtype.lower()}" not in SHADER_VARIANTS:
            raise VulkanError(
                f"{name}.weight is stored as {qtype}, which has no GPU "
                "dequantizer here. Q2_K to Q6_K, Q4_0 to Q5_1, Q8_0 and "
                "float16 are supported."
            )
        block, type_bytes = BLOCK_SIZES[qtype]
        if in_features % block:
            raise VulkanError(
                f"{name}.weight has rows of {in_features}, which {qtype} "
                f"blocks of {block} do not divide."
            )
        expected_bytes = out_features * in_features // block * type_bytes
        if raw.nbytes != expected_bytes:
            raise VulkanError(
                f"{name}.weight holds {raw.nbytes} bytes but {qtype} at "
                f"{(out_features, in_features)} needs {expected_bytes}."
            )
        view = self.weights.upload(raw)
        self.uploaded_bytes += raw.nbytes
        self.qtypes[qtype] = self.qtypes.get(qtype, 0) + 1
        bias_view = None
        if bias:
            bias_view = self._vector(tensors, f"{name}.bias", out_features)
        linear = Linear(name, qtype, out_features, in_features, view, bias_view)
        self.linears[name] = linear
        self.lora_targets[name] = (linear, 0, out_features)
        self.scratch_elements = max(self.scratch_elements, linear.scratch_elements())
        return linear

    def _fused_qkv(self, tensors, prefix):
        D, hd = self.dim, self.head_dim
        kv = self.n_kv_heads * hd
        fused_name = f"{prefix}.attention.qkv"

        def fused(source):
            linear = self._linear(source, fused_name, D + 2 * kv, D)
            for part, offset, rows in (
                ("to_q", 0, D),
                ("to_k", D, kv),
                ("to_v", D + kv, kv),
            ):
                self.lora_targets[f"{prefix}.attention.{part}"] = (linear, offset, rows)
            return [linear]

        if f"{fused_name}.weight" in tensors:
            return fused(tensors)
        parts = []
        for part, rows in (("to_q", D), ("to_k", kv), ("to_v", kv)):
            weight = tensors.get(f"{prefix}.attention.{part}.weight")
            if weight is None:
                raise VulkanError(
                    f"This checkpoint is missing {prefix}.attention.{part}.weight."
                )
            parts.append((part, rows, weight))
        types = {weight.qtype for _p, _r, weight in parts}
        if len(types) == 1 and next(iter(types)) not in ("F32", "BF16"):
            raw = numpy.concatenate([weight.raw() for _p, _r, weight in parts])
            qtype = next(iter(types))
            fused_tensor = Tensor(
                f"{fused_name}.weight",
                qtype,
                (D + 2 * kv, D),
                lambda raw=raw: raw,
                raw.nbytes,
            )
            merged = dict(tensors)
            merged[f"{fused_name}.weight"] = fused_tensor
            return fused(merged)
        return [
            self._linear(tensors, f"{prefix}.attention.{part}", rows, D)
            for part, rows, _weight in parts
        ]

    def _block(self, tensors, prefix, modulated):
        D = self.dim
        self._check_stop()
        block = {
            "qkv": self._fused_qkv(tensors, prefix),
            "out": self._linear(tensors, f"{prefix}.attention.to_out.0", D, D),
            "norm_q": self._vector(
                tensors,
                f"{prefix}.attention.norm_q.weight",
                self.head_dim,
                required=False,
            ),
            "norm_k": self._vector(
                tensors,
                f"{prefix}.attention.norm_k.weight",
                self.head_dim,
                required=False,
            ),
            "w1": self._linear(
                tensors, f"{prefix}.feed_forward.w1", self.ffn_hidden, D
            ),
            "w2": self._linear(
                tensors, f"{prefix}.feed_forward.w2", D, self.ffn_hidden
            ),
            "w3": self._linear(
                tensors, f"{prefix}.feed_forward.w3", self.ffn_hidden, D
            ),
            "attention_norm1": self._vector(
                tensors, f"{prefix}.attention_norm1.weight", D
            ),
            "ffn_norm1": self._vector(tensors, f"{prefix}.ffn_norm1.weight", D),
            "attention_norm2": self._vector(
                tensors, f"{prefix}.attention_norm2.weight", D
            ),
            "ffn_norm2": self._vector(tensors, f"{prefix}.ffn_norm2.weight", D),
            "modulated": modulated,
            "adaln": None,
        }
        if modulated:
            block["adaln"] = self._linear(
                tensors,
                f"{prefix}.adaLN_modulation.0",
                4 * D,
                min(D, Z_ADALN_EMBED_DIM),
                bias=True,
            )
        return block

    def _check_stop(self):
        if self.stop_check():
            raise KeyboardInterrupt()

    def _load(self, tensors):
        D = self.dim
        embed = min(D, Z_ADALN_EMBED_DIM)
        mid = self.config["t_embedder_mid"]
        self.progress(f"Uploading the DiT to {self.device.name}...")
        self.device.reserve = self.activation_bytes(RESERVE_TOKENS)
        self.weights.plan(sum(t.nbytes for t in tensors.values() if len(t.shape) > 1))

        self.x_embedder = self._linear(
            tensors, f"all_x_embedder.{self.key}", D, self.patch_features, bias=True
        )
        self.final_linear = self._linear(
            tensors,
            f"all_final_layer.{self.key}.linear",
            self.patch_features,
            D,
            bias=True,
        )
        self.final_adaln = self._linear(
            tensors,
            f"all_final_layer.{self.key}.adaLN_modulation.1",
            D,
            embed,
            bias=True,
        )
        self.cap_norm = self._vector(
            tensors, "cap_embedder.0.weight", self.cap_feat_dim
        )
        self.cap_linear = self._linear(
            tensors, "cap_embedder.1", D, self.cap_feat_dim, bias=True
        )
        self.x_pad_token = (
            tensors["x_pad_token"].float32().reshape(-1)
            if "x_pad_token" in tensors
            else None
        )
        self.cap_pad_token = (
            tensors["cap_pad_token"].float32().reshape(-1)
            if "cap_pad_token" in tensors
            else None
        )
        for name, token in (
            ("x_pad_token", self.x_pad_token),
            ("cap_pad_token", self.cap_pad_token),
        ):
            if token is None:
                raise VulkanError(f"This checkpoint is missing {name}.")
            if token.size != D:
                raise VulkanError(
                    f"{name} has {token.size} values but the DiT wants {D}."
                )

        self.t_w1 = self._cpu_matrix(
            tensors, "t_embedder.mlp.0.weight", (mid, Z_FREQUENCY_EMBEDDING_SIZE)
        )
        self.t_b1 = (
            tensors["t_embedder.mlp.0.bias"].float32().reshape(-1)
            if "t_embedder.mlp.0.bias" in tensors
            else numpy.zeros(mid, numpy.float32)
        )
        self.t_w2 = self._cpu_matrix(tensors, "t_embedder.mlp.2.weight", (embed, mid))
        self.t_b2 = (
            tensors["t_embedder.mlp.2.bias"].float32().reshape(-1)
            if "t_embedder.mlp.2.bias" in tensors
            else numpy.zeros(embed, numpy.float32)
        )

        self.noise_refiner = []
        self.context_refiner = []
        self.layers = []
        total = 2 * self.n_refiner_layers + self.n_layers
        done = 0
        for index in range(self.n_refiner_layers):
            self.noise_refiner.append(
                self._block(tensors, f"noise_refiner.{index}", True)
            )
            done += 1
            self.context_refiner.append(
                self._block(tensors, f"context_refiner.{index}", False)
            )
            done += 1
            self.progress(
                f"  {done}/{total} blocks, {format_bytes(self.uploaded_bytes)}"
            )
        for index in range(self.n_layers):
            self.layers.append(self._block(tensors, f"layers.{index}", True))
            done += 1
            if done % 5 == 0 or done == total:
                self.progress(
                    f"  {done}/{total} blocks, {format_bytes(self.uploaded_bytes)}"
                )

        self.modulated_blocks = [
            b for b in self.noise_refiner + self.layers if b["modulated"]
        ]
        for index, block in enumerate(self.modulated_blocks):
            block["mod_index"] = index

        self.adaln_input = self.small.allocate(embed * 4)
        self.final_input = self.small.allocate(embed * 4)
        self.modulation = self.small.allocate(
            max(1, len(self.modulated_blocks)) * 4 * D * 4
        )
        self.final_scale = self.small.allocate(D * 4)
        self._ensure_scratch()

        types = ", ".join(
            f"{count} x {name}" for name, count in sorted(self.qtypes.items())
        )
        where = (
            f"{format_bytes(self.weights.spilled)} of it in host memory"
            if self.weights.spilled
            else "all of it in device memory"
        )
        self.progress(
            f"Uploaded {format_bytes(self.uploaded_bytes)} of weights ({types}), {where}."
        )
        free, total = self.device.budget()
        self.progress(
            f"{self.device.name} reports {format_bytes(free)} of "
            f"{format_bytes(total)} device memory free."
        )

    def _ensure_scratch(self):
        needed = self.scratch_elements * 2
        if needed and (self.scratch is None or self.scratch.size < needed):
            if self.scratch is not None:
                self.scratch.buffer.free()
            self.scratch = self.device.create_buffer(needed).view()

    def lora_shape(self, target):
        found = self.lora_targets.get(target)
        if found is None:
            return None
        linear, _offset, rows = found
        return rows, linear.in_features

    def add_lora(self, target, adapter, down, up, scale=1.0, factor=1.0):
        """Attach down [r, in] and up [out, r] to a linear, at scale * factor."""
        found = self.lora_targets.get(target)
        if found is None:
            raise VulkanError(f"The DiT has no linear called {target}.")
        linear, offset, rows = found
        down = numpy.ascontiguousarray(down, dtype=numpy.float32)
        up = numpy.ascontiguousarray(up, dtype=numpy.float32)
        rank = down.shape[0]
        if (
            down.shape[1] != linear.in_features
            or up.shape[0] != rows
            or up.shape[1] != rank
        ):
            raise VulkanError(
                f"A LoRA of {down.shape} x {up.shape} does not fit {target}, "
                f"which is {rows} x {linear.in_features}."
            )
        key = (adapter, target)
        old = linear.loras.pop(key, None)
        if old is not None:
            self.device.wait_idle()
            old.down.buffer.free()
            old.up.buffer.free()
        down_view = self.device.create_buffer(down.size * 2).view()
        self.device.upload(down_view, down.astype(numpy.float16))
        up_view = self.device.create_buffer(up.size * 2).view()
        self.device.upload(up_view, up.astype(numpy.float16))
        linear.loras[key] = Lora(down_view, up_view, rank, scale, factor, offset)
        self.lora_tmp_rank = max(self.lora_tmp_rank, rank)

    def set_lora_scale(self, adapter, scale):
        for linear in self.linears.values():
            for (name, _target), lora in linear.loras.items():
                if name == adapter:
                    lora.scale = float(scale) * lora.factor

    def clear_loras(self):
        self.device.wait_idle()
        for linear in self.linears.values():
            for lora in linear.loras.values():
                lora.down.buffer.free()
                lora.up.buffer.free()
            linear.loras = {}
        self.lora_tmp_rank = 0

    def activation_bytes(self, tokens, heads=1):
        D, hd, F = self.dim, self.head_dim, self.ffn_hidden
        qkv_cols = D + 2 * self.n_kv_heads * hd
        per_token = (
            3 * D + qkv_cols + 2 * F + hd + max(self.patch_features, self.cap_feat_dim)
        )
        return (
            tokens * per_token * 4
            + self.scratch_elements * 2
            + heads * tokens * tokens * 4
        )

    def _activation(self, name, nbytes):
        found = self.activations.pop(name, None)
        if found is not None and found.size >= nbytes:
            self.activations[name] = found
            return found
        self._activation_tokens = None
        if found is not None:
            found.buffer.free()
        if nbytes > self.device.max_storage_range:
            raise VulkanError(
                f"The {name} activations need {format_bytes(nbytes)} in one "
                f"buffer, more than {self.device.name} can bind "
                f"({format_bytes(self.device.max_storage_range)}). Use a "
                "smaller image or another device."
            )
        view = self.device.create_buffer(nbytes).view()
        if not view.buffer.device_local and not self.device.unified:
            self.progress(
                f"Warning: the {name} activations ({format_bytes(nbytes)}) did "
                f"not fit {self.device.name}'s memory and live in host memory, "
                "which is much slower. A smaller quantization or image helps."
            )
        self.activations[name] = view
        return view

    def release_activations(self):
        self.device.wait_idle()
        for view in self.activations.values():
            view.buffer.free()
        self.activations = {}
        self._activation_tokens = None
        if self.lora_tmp is not None:
            self.lora_tmp.buffer.free()
            self.lora_tmp = None

    def release(self):
        try:
            self.device.wait_idle()
        except Exception:  # noqa: BLE001, S110
            pass
        self.release_activations()
        self.clear_loras()
        if self.scratch is not None:
            self.scratch.buffer.free()
            self.scratch = None
        self.weights.free()
        self.small.free()
        self.linears = {}
        self.loaded = False

    def __del__(self):
        try:
            if getattr(self, "loaded", False):
                self.release()
        except Exception:  # noqa: BLE001, S110
            pass

    def _linear_forward(self, linear, a, M, lda, c, ldc, off_a=0, off_c=0):
        device = self.device
        N, K = linear.out_features, linear.in_features
        if linear.needs_dequant:
            block, _bytes = BLOCK_SIZES[linear.qtype]
            device.dequantize(linear.qtype, linear.view, self.scratch, N * K // block)
            b = self.scratch
        else:
            b = linear.view
        device.matmul(
            a,
            b,
            c,
            M,
            N,
            K,
            lda,
            K,
            ldc,
            b_f16=True,
            off_a=off_a,
            off_c=off_c,
            bias=linear.bias,
        )
        for lora in linear.loras.values():
            if not lora.scale:
                continue
            needed = max(M * self.lora_tmp_rank * 4, 4)
            if self.lora_tmp is None or self.lora_tmp.size < needed:
                old, self.lora_tmp = self.lora_tmp, None
                if old is not None:
                    old.buffer.free()
                self.lora_tmp = device.create_buffer(needed).view()
            r = lora.rank
            rows = lora.up.size // (2 * r)
            device.matmul(
                a, lora.down, self.lora_tmp, M, r, K, lda, K, r, b_f16=True, off_a=off_a
            )
            device.matmul(
                self.lora_tmp,
                lora.up,
                c,
                M,
                rows,
                r,
                r,
                r,
                ldc,
                b_f16=True,
                off_c=off_c + lora.col_offset,
                alpha=lora.scale,
                accumulate=True,
            )
        device.flush()

    def _attention(self, act, tokens, off_rows):
        """Attention over qkv rows [off_rows, off_rows + tokens), head groups at a time."""
        device = self.device
        D, hd, H, Hkv = self.dim, self.head_dim, self.n_heads, self.n_kv_heads
        qkv_cols = D + 2 * Hkv * hd
        qkv, attn, scores = act["qkv"], act["attn"], act["scores"]
        group = max(1, min(H, scores.size // max(1, tokens * tokens * 4)))
        scale = 1.0 / math.sqrt(hd)
        base_q = off_rows * qkv_cols
        base_o = off_rows * D
        head = 0
        while head < H:
            count = min(group, H - head)
            if self.kv_groups == 1:
                device.matmul(
                    qkv,
                    qkv,
                    scores,
                    tokens,
                    tokens,
                    hd,
                    qkv_cols,
                    qkv_cols,
                    tokens,
                    b_f16=False,
                    b_kn=False,
                    off_a=base_q + head * hd,
                    off_b=base_q + D + head * hd,
                    off_c=0,
                    batch=count,
                    batch_a=hd,
                    batch_b=hd,
                    batch_c=tokens * tokens,
                    alpha=scale,
                )
                device.softmax(
                    scores,
                    tokens,
                    tokens,
                    tokens,
                    batch=count,
                    batch_stride=tokens * tokens,
                )
                device.matmul(
                    scores,
                    qkv,
                    attn,
                    tokens,
                    hd,
                    tokens,
                    tokens,
                    qkv_cols,
                    D,
                    b_f16=False,
                    b_kn=True,
                    off_a=0,
                    off_b=base_q + D + Hkv * hd + head * hd,
                    off_c=base_o + head * hd,
                    batch=count,
                    batch_a=tokens * tokens,
                    batch_b=hd,
                    batch_c=hd,
                )
            else:
                for h in range(head, head + count):
                    kv = h // self.kv_groups
                    slot = (h - head) * tokens * tokens
                    device.matmul(
                        qkv,
                        qkv,
                        scores,
                        tokens,
                        tokens,
                        hd,
                        qkv_cols,
                        qkv_cols,
                        tokens,
                        b_f16=False,
                        off_a=base_q + h * hd,
                        off_b=base_q + D + kv * hd,
                        off_c=slot,
                        alpha=scale,
                    )
                device.softmax(
                    scores,
                    tokens,
                    tokens,
                    tokens,
                    batch=count,
                    batch_stride=tokens * tokens,
                )
                for h in range(head, head + count):
                    kv = h // self.kv_groups
                    slot = (h - head) * tokens * tokens
                    device.matmul(
                        scores,
                        qkv,
                        attn,
                        tokens,
                        hd,
                        tokens,
                        tokens,
                        qkv_cols,
                        D,
                        b_f16=False,
                        b_kn=True,
                        off_a=slot,
                        off_b=base_q + D + Hkv * hd + kv * hd,
                        off_c=base_o + h * hd,
                    )
            device.flush()
            head += count

    def _block_forward(self, block, act, tokens, off_rows, rope_off):
        device = self.device
        D, hd, H, Hkv = self.dim, self.head_dim, self.n_heads, self.n_kv_heads
        F = self.ffn_hidden
        qkv_cols = D + 2 * Hkv * hd
        x, xn, qkv, attn = act["x"], act["xn"], act["qkv"], act["attn"]
        h1, h3 = act["h1"], act["h3"]
        cos, sin = act["cos"], act["sin"]
        modulated = block["modulated"]
        off_x = off_rows * D
        mod = 0
        if modulated:
            mod = block["mod_index"] * 4 * D
            self._linear_forward(
                block["adaln"],
                self.adaln_input,
                1,
                block["adaln"].in_features,
                self.modulation,
                4 * D,
                off_c=mod,
            )

        flags = NORM_WEIGHT | (NORM_SCALE if modulated else 0)
        device.norm(
            x,
            xn,
            tokens,
            D,
            D,
            D,
            flags,
            Z_NORM_EPS,
            block["attention_norm1"],
            self.modulation,
            off_x=off_x,
            off_y=off_x,
            off_s=mod,
        )

        col = 0
        for linear in block["qkv"]:
            self._linear_forward(
                linear,
                xn,
                tokens,
                D,
                qkv,
                qkv_cols,
                off_a=off_x,
                off_c=off_rows * qkv_cols + col,
            )
            col += linear.out_features

        rope_flags = ROPE_ROTATE
        device.head_norm_rope(
            qkv,
            tokens,
            H,
            hd,
            qkv_cols,
            rope_flags | (ROPE_NORM if block["norm_q"] is not None else 0),
            Z_NORM_EPS,
            block["norm_q"],
            cos,
            sin,
            off=off_rows * qkv_cols,
            off_cos=rope_off,
            off_sin=rope_off,
            ld_rope=hd // 2,
        )
        device.head_norm_rope(
            qkv,
            tokens,
            Hkv,
            hd,
            qkv_cols,
            rope_flags | (ROPE_NORM if block["norm_k"] is not None else 0),
            Z_NORM_EPS,
            block["norm_k"],
            cos,
            sin,
            off=off_rows * qkv_cols + D,
            off_cos=rope_off,
            off_sin=rope_off,
            ld_rope=hd // 2,
        )

        self._attention(act, tokens, off_rows)

        self._linear_forward(
            block["out"], attn, tokens, D, xn, D, off_a=off_x, off_c=off_x
        )
        flags = NORM_WEIGHT | NORM_RESIDUAL | (NORM_GATE if modulated else 0)
        device.norm(
            xn,
            x,
            tokens,
            D,
            D,
            D,
            flags,
            Z_NORM_EPS,
            block["attention_norm2"],
            self.modulation,
            off_x=off_x,
            off_y=off_x,
            off_s=mod + D,
        )

        flags = NORM_WEIGHT | (NORM_SCALE if modulated else 0)
        device.norm(
            x,
            xn,
            tokens,
            D,
            D,
            D,
            flags,
            Z_NORM_EPS,
            block["ffn_norm1"],
            self.modulation,
            off_x=off_x,
            off_y=off_x,
            off_s=mod + 2 * D,
        )
        self._linear_forward(block["w1"], xn, tokens, D, h1, F, off_a=off_x)
        self._linear_forward(block["w3"], xn, tokens, D, h3, F, off_a=off_x)
        device.silu_mul(h1, h3, tokens * F)
        self._linear_forward(block["w2"], h1, tokens, F, xn, D, off_c=off_x)
        flags = NORM_WEIGHT | NORM_RESIDUAL | (NORM_GATE if modulated else 0)
        device.norm(
            xn,
            x,
            tokens,
            D,
            D,
            D,
            flags,
            Z_NORM_EPS,
            block["ffn_norm2"],
            self.modulation,
            off_x=off_x,
            off_y=off_x,
            off_s=mod + 3 * D,
        )

    def _prepare(self, n_img, n_cap):
        """Activation buffers for a sequence of n_img + n_cap tokens."""
        D, hd, H, Hkv, F = (
            self.dim,
            self.head_dim,
            self.n_heads,
            self.n_kv_heads,
            self.ffn_hidden,
        )
        n = n_img + n_cap
        if self._activation_tokens == (n_img, n_cap):
            return self.activations
        qkv_cols = D + 2 * Hkv * hd
        self._activation("x", n * D * 4)
        self._activation("xn", n * max(D, self.cap_feat_dim) * 4)
        self._activation("qkv", n * qkv_cols * 4)
        self._activation("attn", n * D * 4)
        self._activation("h1", n * F * 4)
        self._activation("h3", n * F * 4)
        self._activation("cos", n * (hd // 2) * 4)
        self._activation("sin", n * (hd // 2) * 4)
        self._activation(
            "x_in", max(n_img * self.patch_features, n_cap * self.cap_feat_dim) * 4
        )
        self._activation("out", n_img * self.patch_features * 4)
        per_head = n * n * 4
        limit = self.device.max_storage_range
        if per_head > limit:
            raise VulkanError(
                f"{n} tokens need {format_bytes(per_head)} of attention scores per "
                f"head, more than {self.device.name} can bind at once "
                f"({format_bytes(limit)})."
            )
        free, _total = self.device.budget()
        if self.device.unified:
            share, cap = 0.25, SCORES_MAX_UNIFIED_BYTES
        else:
            share, cap = 0.5, SCORES_MAX_BYTES
            free = max(0, free - self.device.margin())
        want = max(
            1, min(H, int(free * share) // per_head, limit // per_head, cap // per_head)
        )
        scores = self.activations.get("scores")
        if scores is None or scores.size < per_head * want:
            self._activation("scores", per_head * want)
        self._activation_tokens = (n_img, n_cap)
        scores = self.activations["scores"]
        total = sum(view.size for view in self.activations.values())
        free_after, _total = self.device.budget()
        self.progress(
            f"Activations for {n} tokens: {format_bytes(total)}, of which "
            f"{format_bytes(scores.size)} of attention scores for "
            f"{scores.size // per_head} heads per pass. {format_bytes(free_after)} "
            "of device memory still free."
        )
        return self.activations

    def patchify(self, image):
        pH = pW = self.patch_size
        pF = self.f_patch_size
        channels, frames, height, width = image.shape
        f_tokens, h_tokens, w_tokens = frames // pF, height // pH, width // pW
        image = image.reshape(channels, f_tokens, pF, h_tokens, pH, w_tokens, pW)
        image = image.transpose(1, 3, 5, 2, 4, 6, 0)
        return (
            numpy.ascontiguousarray(
                image.reshape(f_tokens * h_tokens * w_tokens, pF * pH * pW * channels)
            ),
            (f_tokens, h_tokens, w_tokens),
        )

    def unpatchify(self, tokens, grid):
        pH = pW = self.patch_size
        pF = self.f_patch_size
        f_tokens, h_tokens, w_tokens = grid
        out = tokens[: f_tokens * h_tokens * w_tokens].reshape(
            f_tokens, h_tokens, w_tokens, pF, pH, pW, self.out_channels
        )
        out = out.transpose(6, 0, 3, 1, 4, 2, 5)
        return numpy.ascontiguousarray(
            out.reshape(self.out_channels, f_tokens * pF, h_tokens * pH, w_tokens * pW)
        )

    def forward(self, x, timestep, cap_feats):
        """x: [C, F, H, W] float32; timestep: scalar; cap_feats: [L, D]."""
        try:
            return self._forward(x, timestep, cap_feats)
        except BaseException:
            self.device.abort()
            self.device.wait_idle()
            raise

    def _forward(self, x, timestep, cap_feats):
        device = self.device
        D, hd = self.dim, self.head_dim
        x = numpy.ascontiguousarray(x, dtype=numpy.float32)
        cap_feats = numpy.ascontiguousarray(cap_feats, dtype=numpy.float32)
        if cap_feats.ndim != 2 or cap_feats.shape[1] != self.cap_feat_dim:
            raise VulkanError(
                f"Caption features are {cap_feats.shape}; the DiT wants [tokens, {self.cap_feat_dim}]."
            )
        t = (
            float(numpy.asarray(timestep, dtype=numpy.float32).reshape(-1)[0])
            * self.t_scale
        )

        cap_len = cap_feats.shape[0]
        cap_pad = pad_rows(cap_len)
        n_cap = cap_len + cap_pad
        cap_ids = coordinate_grid((n_cap, 1, 1), (1, 0, 0))

        image, grid = self.patchify(x)
        image_len = image.shape[0]
        image_pad = pad_rows(image_len)
        n_img = image_len + image_pad
        image_ids = coordinate_grid(grid, (n_cap + 1, 0, 0))
        if image_pad:
            image_ids = numpy.concatenate(
                [image_ids, numpy.zeros((image_pad, 3), dtype=numpy.int64)]
            )
            image = numpy.concatenate(
                [image, numpy.repeat(image[-1:], image_pad, axis=0)]
            )
        if cap_pad:
            cap_feats = numpy.concatenate(
                [cap_feats, numpy.repeat(cap_feats[-1:], cap_pad, axis=0)]
            )

        ids = numpy.concatenate([image_ids, cap_ids])
        cos, sin = rope_tables(ids)

        emb = timestep_embedding(t)
        hidden = silu(self.t_w1 @ emb + self.t_b1)
        adaln_input = (self.t_w2 @ hidden + self.t_b2).astype(numpy.float32)

        act = self._prepare(n_img, n_cap)
        device.upload(act["cos"], cos)
        device.upload(act["sin"], sin)
        device.upload(self.adaln_input, adaln_input)
        device.upload(self.final_input, silu(adaln_input).astype(numpy.float32))
        started = time.monotonic()
        marks = []

        def mark(label):
            if PROFILE:
                device.flush()
                marks.append((label, time.monotonic()))

        # Image tokens.
        device.upload(act["x_in"], image)
        self._linear_forward(
            self.x_embedder, act["x_in"], n_img, self.patch_features, act["x"], D
        )
        if image_pad:
            device.upload(
                act["x"].sub(image_len * D * 4, image_pad * D * 4),
                numpy.tile(self.x_pad_token, image_pad),
            )
        for block in self.noise_refiner:
            self._check_stop()
            self._block_forward(block, act, n_img, 0, 0)
            device.flush()
        mark("noise refiner")

        # Caption tokens.
        device.upload(act["x_in"], cap_feats)
        device.norm(
            act["x_in"],
            act["xn"],
            n_cap,
            self.cap_feat_dim,
            self.cap_feat_dim,
            self.cap_feat_dim,
            NORM_WEIGHT,
            Z_NORM_EPS,
            self.cap_norm,
        )
        self._linear_forward(
            self.cap_linear,
            act["xn"],
            n_cap,
            self.cap_feat_dim,
            act["x"],
            D,
            off_c=n_img * D,
        )
        if cap_pad:
            device.upload(
                act["x"].sub((n_img + cap_len) * D * 4, cap_pad * D * 4),
                numpy.tile(self.cap_pad_token, cap_pad),
            )
        for block in self.context_refiner:
            self._check_stop()
            self._block_forward(block, act, n_cap, n_img, n_img * (hd // 2))
            device.flush()
        mark("context refiner")

        # The unified sequence.
        n = n_img + n_cap
        for block in self.layers:
            self._check_stop()
            self._block_forward(block, act, n, 0, 0)
            device.flush()
        mark("layers")

        # Final layer, image rows only.
        self._linear_forward(
            self.final_adaln,
            self.final_input,
            1,
            self.final_adaln.in_features,
            self.final_scale,
            D,
        )
        device.norm(
            act["x"],
            act["xn"],
            n_img,
            D,
            D,
            D,
            NORM_CENTER | NORM_SCALE,
            Z_FINAL_NORM_EPS,
            None,
            self.final_scale,
        )
        self._linear_forward(
            self.final_linear, act["xn"], n_img, D, act["out"], self.patch_features
        )
        out = device.download(act["out"], numpy.float32, n_img * self.patch_features)
        mark("final layer")
        if PROFILE:
            previous = started
            parts = []
            for label, at in marks:
                parts.append(f"{label} {at - previous:.2f}s")
                previous = at
            heads = act["scores"].size // (n * n * 4)
            free_now, _total = device.budget()
            print(
                f"  {n} tokens, {heads} heads per attention pass: "
                + ", ".join(parts)
                + f", {marks[-1][1] - started:.2f}s in all, "
                f"{format_bytes(free_now)} of device memory free"
            )
        return self.unpatchify(out.reshape(n_img, self.patch_features), grid)

    def __call__(self, x, timestep, cap_feats):
        torch = sys.modules.get("torch")
        is_torch = torch is not None and any(
            torch.is_tensor(value) for value in (x, timestep, cap_feats)
        )
        if is_torch:
            x = x.detach().to("cpu", torch.float32).numpy()
            timestep = timestep.detach().to("cpu", torch.float32).numpy()
            cap_feats = cap_feats.detach().to("cpu", torch.float32).numpy()
        out = self.forward(x, timestep, cap_feats)
        if is_torch:
            return torch.from_numpy(out)
        return out

    @property
    def dtype(self):
        torch = sys.modules.get("torch")
        return torch.float32 if torch is not None else numpy.float32


def load_z_image(paths, device=None, progress=None, stop_check=None):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    tensors = read_checkpoints(paths)
    dev = device if isinstance(device, Device) else get_device(device)
    return VulkanZImage(dev, tensors, progress=progress, stop_check=stop_check)


class Conv:
    def __init__(self, weight, bias, out_channels, in_channels, ksize):
        self.weight = weight
        self.bias = bias
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.ksize = ksize


class Norm:
    def __init__(self, gamma, beta, eps):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps


class VulkanVAE:
    def __init__(self, device, vae, progress=None, fallback=None):
        self.device = device
        self.progress = progress or (lambda message: None)
        self.fallback = fallback
        self.config = vae.config
        self.groups = int(self.config.norm_num_groups)
        torch = sys.modules["torch"]
        self.dtype = torch.float32
        self.use_tiling = bool(getattr(vae, "use_tiling", False))
        self.use_slicing = bool(getattr(vae, "use_slicing", False))
        self.tile_latent_min_size = int(getattr(vae, "tile_latent_min_size", 128))
        self.tile_overlap_factor = float(getattr(vae, "tile_overlap_factor", 0.25))
        self.scale = 2 ** (len(self.config.block_out_channels) - 1)
        self.weights = Arena(device, chunk_bytes=32 * 1024 * 1024)
        self.buffers = {}
        self.loaded = False
        try:
            self._load(vae)
        except BaseException:
            self.release()
            raise
        self.loaded = True

    def _conv(self, module):
        weight = module.weight.detach().to("cpu").float()
        out_channels, in_channels, kh, kw = weight.shape
        if kh != kw or kh not in (1, 3):
            raise VulkanError(f"A {kh}x{kw} convolution is not supported.")
        packed = weight.permute(0, 2, 3, 1).reshape(out_channels, kh * kw * in_channels)
        view = self.weights.upload(packed.numpy().astype(numpy.float16))
        bias = None
        if module.bias is not None:
            bias = self.weights.upload(module.bias.detach().to("cpu").float().numpy())
        return Conv(view, bias, out_channels, in_channels, kh)

    def _norm(self, module):
        return Norm(
            self.weights.upload(module.weight.detach().to("cpu").float().numpy()),
            self.weights.upload(module.bias.detach().to("cpu").float().numpy()),
            float(module.eps),
        )

    def _resnet(self, module):
        return {
            "norm1": self._norm(module.norm1),
            "conv1": self._conv(module.conv1),
            "norm2": self._norm(module.norm2),
            "conv2": self._conv(module.conv2),
            "shortcut": (
                self._conv(module.conv_shortcut)
                if module.conv_shortcut is not None
                else None
            ),
            "scale": float(getattr(module, "output_scale_factor", 1.0)),
        }

    def _attention(self, module):
        torch = sys.modules["torch"]
        qkv = (
            torch.cat(
                [module.to_q.weight, module.to_k.weight, module.to_v.weight], dim=0
            )
            .detach()
            .to("cpu")
            .float()
        )
        bias = (
            torch.cat([module.to_q.bias, module.to_k.bias, module.to_v.bias], dim=0)
            .detach()
            .to("cpu")
            .float()
        )
        channels = int(module.to_q.weight.shape[1])
        if int(getattr(module, "heads", 1)) != 1:
            raise VulkanError("Only single-head VAE attention is supported.")
        return {
            "norm": self._norm(module.group_norm),
            "qkv": Conv(
                self.weights.upload(qkv.numpy().astype(numpy.float16)),
                self.weights.upload(bias.numpy()),
                3 * channels,
                channels,
                1,
            ),
            "out": self._conv_from_linear(module.to_out[0]),
            "scale": float(getattr(module, "scale", channels**-0.5)),
            "rescale": float(getattr(module, "rescale_output_factor", 1.0)),
            "channels": channels,
        }

    def _conv_from_linear(self, module):
        weight = module.weight.detach().to("cpu").float()
        out_channels, in_channels = weight.shape
        view = self.weights.upload(weight.numpy().astype(numpy.float16))
        bias = (
            self.weights.upload(module.bias.detach().to("cpu").float().numpy())
            if module.bias is not None
            else None
        )
        return Conv(view, bias, out_channels, in_channels, 1)

    def _load(self, vae):
        decoder = vae.decoder
        self.progress(f"Uploading the VAE decoder to {self.device.name}...")
        self.post_quant = None
        if (
            getattr(self.config, "use_post_quant_conv", False)
            and getattr(vae, "post_quant_conv", None) is not None
        ):
            self.post_quant = self._conv(vae.post_quant_conv)
        self.conv_in = self._conv(decoder.conv_in)
        mid = decoder.mid_block
        self.mid = [("resnet", self._resnet(mid.resnets[0]))]
        for attention, resnet in zip(mid.attentions, mid.resnets[1:]):
            if attention is not None:
                self.mid.append(("attention", self._attention(attention)))
            self.mid.append(("resnet", self._resnet(resnet)))
        self.up = []
        for block in decoder.up_blocks:
            stage = [("resnet", self._resnet(resnet)) for resnet in block.resnets]
            for upsampler in block.upsamplers or []:
                stage.append(("upsample", self._conv(upsampler.conv)))
            self.up.append(stage)
        self.norm_out = self._norm(decoder.conv_norm_out)
        self.conv_out = self._conv(decoder.conv_out)
        self.max_channels = max(conv.out_channels for conv in self._convs())
        self.progress(f"Uploaded {format_bytes(self.weights.total)} of VAE weights.")

    def _convs(self):
        yield self.conv_in
        for kind, layer in self.mid + [item for stage in self.up for item in stage]:
            if kind == "resnet":
                yield layer["conv1"]
                yield layer["conv2"]
                if layer["shortcut"] is not None:
                    yield layer["shortcut"]
            elif kind == "upsample":
                yield layer
        yield self.conv_out

    def _buffer(self, name, nbytes):
        found = self.buffers.pop(name, None)
        if found is not None and found.size >= nbytes:
            self.buffers[name] = found
            return found
        if found is not None:
            found.buffer.free()
        if nbytes > self.device.max_storage_range:
            raise VulkanError(
                f"The VAE needs {format_bytes(nbytes)} in one buffer, more than "
                f"{self.device.name} can bind."
            )
        view = self.device.create_buffer(nbytes).view()
        if not view.buffer.device_local and not self.device.unified:
            raise VulkanError("The VAE's working set did not fit device memory.")
        self.buffers[name] = view
        return view

    def release_buffers(self):
        self.device.wait_idle()
        for view in self.buffers.values():
            view.buffer.free()
        self.buffers = {}

    def release(self):
        try:
            self.device.wait_idle()
        except Exception:  # noqa: BLE001, S110
            pass
        self.release_buffers()
        self.weights.free()
        self.loaded = False

    def __del__(self):
        try:
            if getattr(self, "loaded", False):
                self.release()
        except Exception:  # noqa: BLE001, S110
            pass

    def enable_tiling(self, use_tiling=True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.use_tiling = False

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def working_set(self, tile):
        sizes = {}

        def need(name, nbytes):
            sizes[name] = max(sizes.get(name, 0), nbytes)

        def swap(a, b):
            sizes[a], sizes[b] = sizes.get(b, 0), sizes.get(a, 0)

        P = tile * tile
        need("x", P * self.max_channels * 4)
        need("col", IM2COL_BYTES)
        if self.post_quant is not None:
            need("c1", P * self.post_quant.out_channels * 4)
            swap("x", "c1")
        need("c1", P * self.conv_in.out_channels * 4)
        swap("x", "c1")
        C = self.conv_in.out_channels
        for kind, layer in self.mid + [item for stage in self.up for item in stage]:
            if kind == "resnet":
                out = layer["conv1"].out_channels
                need("h", P * max(C, out) * 4)
                need("c1", P * out * 4)
                if layer["shortcut"] is not None:
                    swap("x", "c1")
                C = out
            elif kind == "attention":
                need("h", P * C * 4)
                need("qkv", P * 3 * C * 4)
                need("attn", P * C * 4)
                chunk = max(1, min(P, (64 * 1024 * 1024) // (P * 4)))
                need("scores", chunk * P * 4)
            else:
                need("up", 4 * P * layer.out_channels * 4)
                swap("x", "up")
                P *= 4
                C = layer.out_channels
        need("h", P * C * 4)
        need("out", P * self.conv_out.out_channels * 4)
        need("partials", (P + GN_BLOCK_PIXELS - 1) // GN_BLOCK_PIXELS * self.groups * 4)
        return sum(sizes.values())

    def _conv_forward(self, conv, x, H, W, y, upsample=False, accumulate=False):
        device = self.device
        OH, OW = (2 * H, 2 * W) if upsample else (H, W)
        rowlen = conv.ksize * conv.ksize * conv.in_channels
        col = self._buffer("col", IM2COL_BYTES)
        band = max(1, min(OH, col.size // (OW * rowlen * 4)))
        row0 = 0
        while row0 < OH:
            rows = min(band, OH - row0)
            device.im2col(
                x, col, H, W, conv.in_channels, OH, OW, conv.ksize, upsample, row0, rows
            )
            device.matmul(
                col,
                conv.weight,
                y,
                rows * OW,
                conv.out_channels,
                rowlen,
                rowlen,
                rowlen,
                conv.out_channels,
                b_f16=True,
                off_c=row0 * OW * conv.out_channels,
                bias=conv.bias,
                accumulate=accumulate,
            )
            device.flush()
            row0 += rows
        return OH, OW

    def _groupnorm_forward(self, norm, x, y, P, C, silu):
        n_blocks = (P + GN_BLOCK_PIXELS - 1) // GN_BLOCK_PIXELS
        partials = self._buffer("partials", n_blocks * self.groups * 4)
        stats = self._buffer("stats", 2 * self.groups * 4)
        self.device.groupnorm(
            x,
            y,
            P,
            C,
            self.groups,
            norm.eps,
            norm.gamma,
            norm.beta,
            partials,
            stats,
            silu,
        )

    def _resnet_forward(self, layer, x, H, W, C):
        P = H * W
        out_channels = layer["conv1"].out_channels
        h = self._buffer("h", P * max(C, out_channels) * 4)
        c1 = self._buffer("c1", P * out_channels * 4)
        self._groupnorm_forward(layer["norm1"], x, h, P, C, True)
        self._conv_forward(layer["conv1"], h, H, W, c1)
        self._groupnorm_forward(layer["norm2"], c1, h, P, out_channels, True)
        if layer["shortcut"] is not None:
            self._conv_forward(layer["shortcut"], x, H, W, c1)
            self._conv_forward(layer["conv2"], h, H, W, c1, accumulate=True)
            self.buffers["c1"], self.buffers["x"] = self.buffers.get("x"), c1
            return c1, out_channels
        self._conv_forward(layer["conv2"], h, H, W, x, accumulate=True)
        return x, C

    def _attention_forward(self, layer, x, H, W, C):
        device = self.device
        P = H * W
        h = self._buffer("h", P * C * 4)
        self._groupnorm_forward(layer["norm"], x, h, P, C, False)
        qkv = self._buffer("qkv", P * 3 * C * 4)
        device.matmul(
            h,
            layer["qkv"].weight,
            qkv,
            P,
            3 * C,
            C,
            C,
            C,
            3 * C,
            b_f16=True,
            bias=layer["qkv"].bias,
        )
        attn = self._buffer("attn", P * C * 4)
        chunk = max(1, min(P, (64 * 1024 * 1024) // (P * 4)))
        scores = self._buffer("scores", chunk * P * 4)
        r0 = 0
        while r0 < P:
            rows = min(chunk, P - r0)
            device.matmul(
                qkv,
                qkv,
                scores,
                rows,
                P,
                C,
                3 * C,
                3 * C,
                P,
                b_f16=False,
                off_a=r0 * 3 * C,
                off_b=C,
                alpha=layer["scale"],
            )
            device.softmax(scores, rows, P, P)
            device.matmul(
                scores,
                qkv,
                attn,
                rows,
                C,
                P,
                P,
                3 * C,
                C,
                b_f16=False,
                b_kn=True,
                off_b=2 * C,
                off_c=r0 * C,
            )
            device.flush()
            r0 += rows
        if layer["rescale"] != 1.0:
            raise VulkanError("A rescaled VAE attention block is not supported.")
        device.matmul(
            attn,
            layer["out"].weight,
            x,
            P,
            C,
            C,
            C,
            C,
            C,
            b_f16=True,
            bias=layer["out"].bias,
            accumulate=True,
        )
        device.flush()

    def _decode_tile(self, latent):
        """latent: [C, h, w] float32 numpy -> [3, 8h, 8w] float32 numpy."""
        device = self.device
        C, H, W = latent.shape
        P = H * W
        x = self._buffer("x", P * self.max_channels * 4)
        device.upload(x, numpy.ascontiguousarray(latent.transpose(1, 2, 0)))
        channels = C
        if self.post_quant is not None:
            y = self._buffer("c1", P * self.post_quant.out_channels * 4)
            self._conv_forward(self.post_quant, x, H, W, y)
            self.buffers["x"], self.buffers["c1"] = y, x
            x = y
        y = self._buffer("c1", P * self.conv_in.out_channels * 4)
        self._conv_forward(self.conv_in, x, H, W, y)
        self.buffers["x"], self.buffers["c1"] = y, x
        x, channels = y, self.conv_in.out_channels

        for kind, layer in self.mid:
            if kind == "resnet":
                x, channels = self._resnet_forward(layer, x, H, W, channels)
            else:
                self._attention_forward(layer, x, H, W, channels)

        for stage in self.up:
            for kind, layer in stage:
                if kind == "resnet":
                    x, channels = self._resnet_forward(layer, x, H, W, channels)
                else:
                    y = self._buffer("up", 4 * H * W * layer.out_channels * 4)
                    H, W = self._conv_forward(layer, x, H, W, y, upsample=True)
                    self.buffers["x"], self.buffers["up"] = y, self.buffers["x"]
                    x, channels = y, layer.out_channels

        P = H * W
        h = self._buffer("h", P * channels * 4)
        self._groupnorm_forward(self.norm_out, x, h, P, channels, True)
        out = self._buffer("out", P * self.conv_out.out_channels * 4)
        self._conv_forward(self.conv_out, h, H, W, out)
        image = device.download(out, numpy.float32, P * self.conv_out.out_channels)
        return numpy.ascontiguousarray(
            image.reshape(H, W, self.conv_out.out_channels).transpose(2, 0, 1)
        )

    def _tile_size(self, h, w):
        limit = self.tile_latent_min_size
        whole = max(h, w)
        wanted = whole if (not self.use_tiling or whole <= limit) else limit
        free, _total = self.device.budget()
        if not self.device.unified:
            free = max(0, free - self.device.margin())
        for tile in [wanted] + [t for t in VAE_TILE_CHOICES if t < wanted]:
            if self.working_set(min(tile, whole)) <= free:
                return tile
        raise VulkanError(
            f"No VAE tile fits the {format_bytes(free)} of device memory left. "
            "A smaller image or quantization would make room."
        )

    def decode_numpy(self, latent):
        """latent: [C, h, w] float32 numpy -> [3, H, W] float32 numpy."""
        C, h, w = latent.shape
        tile = self._tile_size(h, w)
        if tile >= max(h, w):
            return self._decode_tile(latent)
        overlap = int(tile * (1 - self.tile_overlap_factor))
        sample_tile = tile * self.scale
        blend = int(sample_tile * self.tile_overlap_factor)
        row_limit = sample_tile - blend
        rows = []
        for i in range(0, h, overlap):
            row = []
            for j in range(0, w, overlap):
                row.append(self._decode_tile(latent[:, i : i + tile, j : j + tile]))
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, piece in enumerate(row):
                if i > 0:
                    piece = _blend_v(rows[i - 1][j], piece, blend)
                if j > 0:
                    piece = _blend_h(row[j - 1], piece, blend)
                result_row.append(piece[:, :row_limit, :row_limit])
            result_rows.append(numpy.concatenate(result_row, axis=2))
        return numpy.ascontiguousarray(numpy.concatenate(result_rows, axis=1))

    def decode(self, latents, return_dict=False, generator=None):
        torch = sys.modules["torch"]
        source = latents.detach().to("cpu", torch.float32)
        started = time.monotonic()
        images = []
        try:
            for index in range(source.shape[0]):
                images.append(
                    torch.from_numpy(self.decode_numpy(source[index].numpy()))
                )
            image = torch.stack(images, dim=0)
        except VulkanError as e:
            if self.fallback is None:
                raise
            self.progress(f"Decoding on the CPU instead: {e}")
            self.release_buffers()
            return self.fallback.decode(latents, return_dict=return_dict)
        if PROFILE:
            print(
                f"  VAE decode of {tuple(source.shape[-2:])} in {time.monotonic() - started:.2f}s"
            )
        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=image)
        return (image,)


def _blend_v(a, b, extent):
    extent = min(a.shape[1], b.shape[1], extent)
    for y in range(extent):
        b[:, y, :] = a[:, -extent + y, :] * (1 - y / extent) + b[:, y, :] * (y / extent)
    return b


def _blend_h(a, b, extent):
    extent = min(a.shape[2], b.shape[2], extent)
    for x in range(extent):
        b[:, :, x] = a[:, :, -extent + x] * (1 - x / extent) + b[:, :, x] * (x / extent)
    return b


def _random_blocks(rng, qtype, n_blocks):
    block, size = BLOCK_SIZES[qtype]
    raw = rng.integers(0, 256, size=(n_blocks, size), dtype=numpy.uint8)
    scales = (rng.random(n_blocks, dtype=numpy.float32) * 0.5 + 0.01).astype(
        numpy.float16
    )
    mins = (rng.random(n_blocks, dtype=numpy.float32) * 0.2).astype(numpy.float16)
    positions = {
        "Q8_0": (0, None),
        "Q4_0": (0, None),
        "Q4_1": (0, 2),
        "Q5_0": (0, None),
        "Q5_1": (0, 2),
        "Q2_K": (80, 82),
        "Q3_K": (108, None),
        "Q4_K": (0, 2),
        "Q5_K": (0, 2),
        "Q6_K": (208, None),
    }
    d_at, m_at = positions[qtype]
    raw[:, d_at : d_at + 2] = scales.view(numpy.uint8).reshape(-1, 2)
    if m_at is not None:
        raw[:, m_at : m_at + 2] = mins.view(numpy.uint8).reshape(-1, 2)
    return raw.reshape(-1)


def self_test(check, device_ident=None):
    try:
        device = get_device(device_ident)
    except (VulkanUnavailable, VulkanError) as e:
        check("Vulkan: device", False, str(e))
        return None
    check("Vulkan: device", True, device.describe())
    rng = numpy.random.default_rng(0)

    def compare(label, got, want, tolerance):
        got = numpy.asarray(got, dtype=numpy.float64)
        want = numpy.asarray(want, dtype=numpy.float64)
        if got.shape != want.shape:
            check(label, False, f"shape {got.shape} vs {want.shape}")
            return
        scale = max(1.0, numpy.abs(want).max())
        error = numpy.abs(got - want).max() / scale
        check(
            label,
            bool(error < tolerance) and bool(numpy.isfinite(got).all()),
            f"max relative error {error:.2e}",
        )

    # Dequantizers, against gguf's numpy code.
    for qtype in (
        "Q8_0",
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
    ):
        n_blocks = 37
        block, _size = BLOCK_SIZES[qtype]
        raw = _random_blocks(rng, qtype, n_blocks)
        try:
            want = dequantize_rows(raw, qtype, (n_blocks * block,))
            source = device.create_buffer(raw.nbytes).view()
            device.upload(source, raw)
            target = device.create_buffer(n_blocks * block * 2).view()
            device.dequantize(qtype, source, target, n_blocks)
            got = device.download(target, numpy.float16).astype(numpy.float32)
            source.buffer.free()
            target.buffer.free()
            compare(f"Vulkan: dequantize {qtype}", got, want, 2e-3)
        except Exception as e:  # noqa: BLE001
            check(f"Vulkan: dequantize {qtype}", False, str(e))

    # Q8_0 packing round trip.
    try:
        matrix = rng.standard_normal((8, 64), dtype=numpy.float32)
        packed = quantize_q8_0(matrix)
        unpacked = dequantize_rows(packed.reshape(-1), "Q8_0", (8, 64))
        compare("Vulkan: Q8_0 packing", unpacked, matrix, 1e-2)
    except Exception as e:  # noqa: BLE001
        check("Vulkan: Q8_0 packing", False, str(e))

    # Matmul variants, including odd sizes, batches, bias and accumulate.
    for label, M, N, K, b_f16, b_kn in (
        ("f16 nk large", 200, 150, 70, True, False),
        ("f16 nk small", 33, 17, 40, True, False),
        ("f16 kn large", 140, 130, 50, True, True),
        ("f16 kn small", 20, 9, 45, True, True),
        ("f32 nk large", 130, 140, 32, False, False),
        ("f32 nk small", 31, 100, 64, False, False),
        ("f32 kn large", 130, 129, 150, False, True),
        ("f32 kn small", 20, 8, 45, False, True),
    ):
        try:
            a = rng.standard_normal((M, K), dtype=numpy.float32)
            b = rng.standard_normal((K, N) if b_kn else (N, K), dtype=numpy.float32)
            bias = rng.standard_normal(N, dtype=numpy.float32)
            b_stored = b.astype(numpy.float16) if b_f16 else b
            want = a @ (
                b_stored.astype(numpy.float32)
                if b_kn
                else b_stored.astype(numpy.float32).T
            )
            want = 0.5 * want + bias + 1.0
            va = device.create_buffer(a.nbytes).view()
            vb = device.create_buffer(b_stored.nbytes).view()
            vc = device.create_buffer(M * N * 4).view()
            vbias = device.create_buffer(bias.nbytes).view()
            device.upload(va, a)
            device.upload(vb, b_stored)
            device.upload(vbias, bias)
            device.upload(vc, numpy.ones((M, N), dtype=numpy.float32))
            device.matmul(
                va,
                vb,
                vc,
                M,
                N,
                K,
                K,
                N if b_kn else K,
                N,
                b_f16=b_f16,
                b_kn=b_kn,
                alpha=0.5,
                accumulate=True,
                bias=vbias,
            )
            got = device.download(vc, numpy.float32).reshape(M, N)
            for view in (va, vb, vc, vbias):
                view.buffer.free()
            compare(f"Vulkan: matmul {label}", got, want, 2e-3 if b_f16 else 1e-4)
        except Exception as e:  # noqa: BLE001
            check(f"Vulkan: matmul {label}", False, str(e))

    # Batched, strided matmul the way attention uses it.
    try:
        tokens, hd, heads = 40, 16, 3
        cols = heads * hd * 3
        qkv = rng.standard_normal((tokens, cols), dtype=numpy.float32)
        v = device.create_buffer(qkv.nbytes).view()
        device.upload(v, qkv)
        scores = device.create_buffer(heads * tokens * tokens * 4).view()
        device.matmul(
            v,
            v,
            scores,
            tokens,
            tokens,
            hd,
            cols,
            cols,
            tokens,
            b_f16=False,
            off_a=0,
            off_b=heads * hd,
            batch=heads,
            batch_a=hd,
            batch_b=hd,
            batch_c=tokens * tokens,
            alpha=0.25,
        )
        device.softmax(
            scores, tokens, tokens, tokens, batch=heads, batch_stride=tokens * tokens
        )
        out = device.create_buffer(tokens * heads * hd * 4).view()
        device.matmul(
            scores,
            v,
            out,
            tokens,
            hd,
            tokens,
            tokens,
            cols,
            heads * hd,
            b_f16=False,
            b_kn=True,
            off_b=2 * heads * hd,
            batch=heads,
            batch_a=tokens * tokens,
            batch_b=hd,
            batch_c=hd,
        )
        got = device.download(out, numpy.float32).reshape(tokens, heads * hd)
        want = numpy.zeros_like(got)
        for h in range(heads):
            q = qkv[:, h * hd : (h + 1) * hd]
            k = qkv[:, heads * hd + h * hd : heads * hd + (h + 1) * hd]
            vv = qkv[:, 2 * heads * hd + h * hd : 2 * heads * hd + (h + 1) * hd]
            s = q @ k.T * 0.25
            s = numpy.exp(s - s.max(axis=1, keepdims=True))
            s /= s.sum(axis=1, keepdims=True)
            want[:, h * hd : (h + 1) * hd] = s @ vv
        for view in (v, scores, out):
            view.buffer.free()
        compare("Vulkan: batched attention", got, want, 1e-4)
    except Exception as e:  # noqa: BLE001
        check("Vulkan: batched attention", False, str(e))

    # Norms.
    try:
        rows, dim = 21, 300
        x = rng.standard_normal((rows, dim), dtype=numpy.float32)
        w = rng.standard_normal(dim, dtype=numpy.float32)
        s = rng.standard_normal(dim, dtype=numpy.float32)
        y0 = rng.standard_normal((rows, dim), dtype=numpy.float32)
        vx = device.create_buffer(x.nbytes).view()
        vy = device.create_buffer(x.nbytes).view()
        vw = device.create_buffer(w.nbytes).view()
        vs = device.create_buffer(s.nbytes).view()
        device.upload(vx, x)
        device.upload(vw, w)
        device.upload(vs, s)
        rms = x / numpy.sqrt((x * x).mean(axis=1, keepdims=True) + 1e-5)

        device.norm(vx, vy, rows, dim, dim, dim, NORM_WEIGHT | NORM_SCALE, 1e-5, vw, vs)
        compare(
            "Vulkan: rmsnorm with scale",
            device.download(vy, numpy.float32).reshape(rows, dim),
            rms * w * (1 + s),
            1e-4,
        )

        device.upload(vy, y0)
        device.norm(
            vx,
            vy,
            rows,
            dim,
            dim,
            dim,
            NORM_WEIGHT | NORM_RESIDUAL | NORM_GATE,
            1e-5,
            vw,
            vs,
        )
        compare(
            "Vulkan: gated residual",
            device.download(vy, numpy.float32).reshape(rows, dim),
            y0 + numpy.tanh(s) * rms * w,
            1e-4,
        )

        ln = (x - x.mean(axis=1, keepdims=True)) / numpy.sqrt(
            x.var(axis=1, keepdims=True) + 1e-6
        )
        device.norm(
            vx, vy, rows, dim, dim, dim, NORM_CENTER | NORM_SCALE, 1e-6, None, vs
        )
        compare(
            "Vulkan: layernorm with scale",
            device.download(vy, numpy.float32).reshape(rows, dim),
            ln * (1 + s),
            1e-4,
        )
        for view in (vx, vy, vw, vs):
            view.buffer.free()
    except Exception as e:  # noqa: BLE001
        check("Vulkan: norms", False, str(e))

    # Head norm + rope, against the torch formulation.
    try:
        tokens, heads, hd = 9, 3, 128
        cols = heads * hd * 2
        x = rng.standard_normal((tokens, cols), dtype=numpy.float32)
        w = rng.standard_normal(hd, dtype=numpy.float32)
        ids = numpy.stack([rng.integers(0, 100, tokens) for _ in range(3)], axis=1)
        cos, sin = rope_tables(ids)
        vx = device.create_buffer(x.nbytes).view()
        vw = device.create_buffer(w.nbytes).view()
        vcos = device.create_buffer(cos.nbytes).view()
        vsin = device.create_buffer(sin.nbytes).view()
        device.upload(vx, x)
        device.upload(vw, w)
        device.upload(vcos, cos)
        device.upload(vsin, sin)
        device.head_norm_rope(
            vx,
            tokens,
            heads,
            hd,
            cols,
            ROPE_NORM | ROPE_ROTATE,
            1e-5,
            vw,
            vcos,
            vsin,
            off=heads * hd,
            ld_rope=hd // 2,
        )
        got = device.download(vx, numpy.float32).reshape(tokens, cols)
        want = x.copy()
        part = x[:, heads * hd :].reshape(tokens, heads, hd)
        normed = (
            part / numpy.sqrt((part * part).mean(axis=-1, keepdims=True) + 1e-5) * w
        )
        even, odd = normed[..., 0::2], normed[..., 1::2]
        c, s = cos[:, None, :], sin[:, None, :]
        rotated = numpy.stack(
            [even * c - odd * s, even * s + odd * c], axis=-1
        ).reshape(tokens, heads, hd)
        want[:, heads * hd :] = rotated.reshape(tokens, heads * hd)
        compare("Vulkan: head norm and rope", got, want, 1e-4)
        for view in (vx, vw, vcos, vsin):
            view.buffer.free()
    except Exception as e:  # noqa: BLE001
        check("Vulkan: head norm and rope", False, str(e))

    # SwiGLU.
    try:
        n = 1000
        a = rng.standard_normal(n, dtype=numpy.float32)
        b = rng.standard_normal(n, dtype=numpy.float32)
        va = device.create_buffer(a.nbytes).view()
        vb = device.create_buffer(b.nbytes).view()
        device.upload(va, a)
        device.upload(vb, b)
        device.silu_mul(va, vb, n)
        compare(
            "Vulkan: silu_mul", device.download(va, numpy.float32), silu(a) * b, 1e-5
        )
        va.buffer.free()
        vb.buffer.free()
    except Exception as e:  # noqa: BLE001
        check("Vulkan: silu_mul", False, str(e))

    return device


def main(arguments):
    if "--build-shaders" in arguments:
        build_shaders(force="--force" in arguments)
        return 0
    if "--devices" in arguments:
        found = vulkan_devices()
        if not found:
            print("No Vulkan devices.")
            return 1
        for ident, label, _kind in found:
            print(f"{ident}  {label}")
        preferred = preferred_vulkan_device(allow_cpu=True)
        print(f"Preferred: {preferred}")
        return 0
    if "--self-test" in arguments:
        failures = []

        def check(label, ok, detail=""):
            print(
                f"{'ok  ' if ok else 'FAIL'}  {label}"
                + (f" - {detail}" if detail else "")
            )
            if not ok:
                failures.append(label)

        ident = None
        for index, argument in enumerate(arguments):
            if argument == "--device" and index + 1 < len(arguments):
                ident = arguments[index + 1]
        started = time.monotonic()
        self_test(check, ident)
        print(
            f"{'FAIL' if failures else 'ok'}: {len(failures)} failure(s) in {time.monotonic() - started:.1f}s."
        )
        return 1 if failures else 0
    print(
        "Usage: animus_vulkan.py [--build-shaders [--force]] [--devices] [--self-test [--device vulkan:N]]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
