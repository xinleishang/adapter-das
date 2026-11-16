import ctypes
import mmap
import os
import unittest
from unittest.mock import Mock, patch

import torch

from hcu_adapter import mapped_host_dtk as bridge


class TestEngramHostMapping(unittest.TestCase):
    def test_lookup_failure_never_falls_back_to_host_address(self):
        for rc in (None, 0, 1):
            runtime = None if rc is None else Mock()
            if runtime is not None:
                runtime.hipHostGetDevicePointer.return_value = rc
            with self.subTest(rc=rc), patch.object(bridge, "_hip_runtime", return_value=runtime):
                with self.assertRaises(RuntimeError):
                    bridge._engram_registered_device_ptr(0x1000)

    def test_native_hip_keeps_upstream_helper(self):
        engram = Mock(_is_hip=True)
        original = engram._registered_device_ptr
        with patch.object(bridge.torch.version, "hip", "6"), patch.object(
            bridge, "import_module", return_value=engram
        ):
            bridge.install_engram_host_mapping()
        self.assertIs(engram._registered_device_ptr, original)

    @unittest.skipUnless(os.environ.get("HCU_TEST_ENGRAM_MAPPING") == "1",
                         "requires an idle HCU and SGLang with adapter bootstrap")
    def test_bootstrapped_gather_uses_registered_address(self):
        from sglang.srt.layers import engram

        self.assertIs(engram._registered_device_ptr, bridge._engram_registered_device_ptr)
        torch.cuda.set_device(0)
        ids = torch.tensor([1000, 1063, 999, 1064], device="cuda")
        out = torch.empty((4, 128), device="cuda", dtype=torch.bfloat16)
        with mmap.mmap(-1, 65536, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as mm:
            data = torch.frombuffer(mm, dtype=torch.uint8)
            data.zero_()
            data[:8192].fill_(0x38)
            data[8192:8448].fill_(127)
            data[8444:8448].fill_(128)
            host = data.data_ptr()
            self.assertEqual(int(torch.cuda.cudart().cudaHostRegister(host, len(mm), 0)), 0)
            try:
                ptr = engram._registered_device_ptr(host)
                alias = ctypes.c_void_p()
                self.assertEqual(bridge._hip_runtime().hipHostGetDevicePointer(
                    ctypes.byref(alias), host, 0), 0)
                self.assertEqual(ptr, alias.value)
                engram.engram_gather(ptr, ptr + 8192, ids, out, 128, 32, 1000, 1064)
                expected = torch.tensor([1, 2, 0, 0], dtype=out.dtype)[:, None]
                self.assertTrue(torch.equal(out.cpu(), expected.expand(-1, 128)))
            finally:
                self.assertEqual(int(torch.cuda.cudart().cudaHostUnregister(host)), 0)
                del data


if __name__ == "__main__":
    unittest.main()
