import ctypes
from functools import wraps
from importlib import import_module
import logging
from os.path import basename
from typing import Tuple
import torch

logger = logging.getLogger(__name__)
_HIP_RUNTIME = None

def _hip_runtime():
    global _HIP_RUNTIME
    if _HIP_RUNTIME is False:
        return None
    if _HIP_RUNTIME is not None:
        return _HIP_RUNTIME
    for library in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            lib = ctypes.CDLL(library)
        except OSError:
            continue
        lib.hipHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.hipHostRegister.restype = ctypes.c_int
        lib.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        lib.hipHostGetDevicePointer.restype = ctypes.c_int
        _HIP_RUNTIME = lib
        return lib
    _HIP_RUNTIME = False
    return None


def _engram_registered_device_ptr(host_ptr: int) -> int:
    hip = _hip_runtime()
    if hip is None:
        raise RuntimeError("Engram host table requires the HIP runtime")
    device_ptr = ctypes.c_void_p()
    rc = hip.hipHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
    if rc != 0 or not device_ptr.value:
        raise RuntimeError(f"Engram hipHostGetDevicePointer failed: rc={rc}, host={host_ptr:#x}")
    return device_ptr.value


def _install_hip_library_lookup(wrapper):
    original = getattr(wrapper, "find_loaded_library", None)
    if original is None or getattr(original, "_hcu_galaxyhip_lookup", False) is True:
        return

    @wraps(original)
    def lookup(lib_name):
        path = original(lib_name)
        if path is not None or lib_name != "libamdhip64":
            return path
        path = original("libgalaxyhip")
        if path is not None:
            name = basename(path)
            if name == "libgalaxyhip.so" or name.startswith("libgalaxyhip.so."):
                return path
        return None

    lookup._hcu_galaxyhip_lookup = True
    wrapper.find_loaded_library = lookup


def install_hip_library_lookup():
    if torch.version.hip:
        module = "sglang.srt.distributed.device_communicators.cuda_wrapper"
        try:
            wrapper = import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == module:
                return
            raise
        _install_hip_library_lookup(wrapper)


def install_engram_host_mapping() -> None:
    if not torch.version.hip:
        return
    try:
        engram = import_module("sglang.srt.layers.engram")
    except ModuleNotFoundError as exc:
        if exc.name == "sglang.srt.layers.engram":
            return
        raise
    # CUDA platform semantics must not suppress HIP host address translation.
    if getattr(engram, "_is_hip", True) is False and hasattr(engram, "_registered_device_ptr"):
        engram._registered_device_ptr = _engram_registered_device_ptr


def _device_ptr_for_host_tensor(tensor: torch.Tensor) -> int:
    """HCU kernels cannot dereference the raw host pointer; translate it."""
    host_ptr = int(tensor.data_ptr())
    hip = _hip_runtime()
    if hip is None:
        return host_ptr
    device_ptr = ctypes.c_void_p()
    rc = int(
        hip.hipHostGetDevicePointer(
            ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
        )
    )
    if rc == 0 and device_ptr.value:
        return int(device_ptr.value)
    logger.warning(
        "PLE hipHostGetDevicePointer failed rc=%s host=%#x; using host pointer",
        rc,
        host_ptr,
    )
    return host_ptr


class _CudaArrayInterface:
    """Non-owning CUDA array interface over a mapped host allocation."""

    def __init__(self, ptr: int, shape: Tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": tuple(shape),
            "typestr": typestr,
            "version": 3,
        }


def _wrap_mapped_host_as_cuda_tensor(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, _CudaArrayInterface]:
    """Expose mapped GTT as a CUDA tensor so ATen gather is HIP-graph-safe.

    Triton ``tl.load`` from a host-mapped pointer is not replayed correctly
    inside HIP full CUDA graphs (decode collapses to token 0). ``F.embedding``
    on this view is captured and replays the current ids.
    """
    if dtype == torch.bfloat16:
        typestr = "|u2"
    elif dtype == torch.int8:
        typestr = "|i1"
    elif dtype == torch.float8_e4m3fn:
        typestr = "|u1"
    else:
        raise TypeError(f"unsupported PLE mapped-view dtype: {dtype}")
    holder = _CudaArrayInterface(ptr, shape, typestr)
    tensor = torch.as_tensor(holder, device=device)
    if dtype == torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
    elif dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.float8_e4m3fn)
    from .registry import register, MECH_SEAM
    register("mapped_host_view", "dtk", __name__+"._wrap_mapped_host_as_cuda_tensor", MECH_SEAM, source=__name__)
    return tensor, holder
