import os
import unittest
from unittest.mock import patch

import torch
from sgl_kernel import kvcacheio as K


class MlaAotBridgeTests(unittest.TestCase):
    def test_page_first_keeps_native_arguments(self):
        src, dst = torch.empty(3, 2, 32), torch.empty(3, 32)
        idx = torch.tensor([1], dtype=torch.int64)
        with patch.object(torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf, "default") as op:
            K.transfer_kv_per_layer_mla_pf_lf(src, dst, idx, idx, 1, 32, 64, 4, 8)
            op.assert_called_once_with(src, dst, idx, idx, 1, 32, 64, 4, 8)

    def test_empty_transfers_do_not_launch(self):
        x = torch.empty(0)
        idx = torch.empty(0, dtype=torch.int64)
        cases = {
            "transfer_kv_per_layer_mla": (x, x, idx, idx, 32),
            "transfer_kv_per_layer_mla_pf_lf": (x, x, idx, idx, 0, 32, 64),
            "transfer_kv_all_layer_mla": (x, x, idx, idx, 32, 2),
            "transfer_kv_all_layer_mla_lf_pf": (x, x, idx, idx, 32, 64, 2),
        }
        for name, args in cases.items():
            with self.subTest(name=name), patch.object(getattr(torch.ops.sgl_kernel, name), "default") as op:
                getattr(K, name)(*args)
                op.assert_not_called()


@unittest.skipUnless(os.environ.get("HICACHE_TEST_GPU") == "1", "GPU opt-in required")
class MlaAotGpuTests(unittest.TestCase):
    def test_roundtrip_noncontiguous_pages_and_layers(self):
        for size in (32, 40, 656, 17408, 149760):
            for layers in (1, 3):
                for layout in ("page_first", "layer_first"):
                    with self.subTest(size=size, layers=layers, layout=layout):
                        self.roundtrip(size, layers, layout)

    def roundtrip(self, size, layers, layout):
        dev = torch.device("cuda:0")
        src = [torch.arange(7 * size, device=dev).add(13 * l).remainder(251).to(torch.uint8).reshape(7, size) for l in range(layers)]
        dst = [torch.full_like(x, 253) for x in src]
        host = torch.full((7, layers, size), 254, dtype=torch.uint8, pin_memory=True) if layout == "page_first" else [torch.full((7, size), 254, dtype=torch.uint8, pin_memory=True) for _ in src]
        si = torch.tensor([4, 0, 6], dtype=torch.int64, device=dev)
        hi = torch.tensor([1, 5, 2], dtype=torch.int64, device=dev)
        di = torch.tensor([6, 3, 0], dtype=torch.int64, device=dev)
        sp = torch.tensor([x.data_ptr() for x in src], dtype=torch.uint64, device=dev)
        hp = None if layout == "page_first" else torch.tensor([K.get_device_accessible_ptr(x, 0) for x in host], dtype=torch.uint64, device=dev)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if layout == "page_first":
                K.transfer_kv_all_layer_mla_lf_pf(sp, host, si, hi, size, layers * size, layers)
                for l in range(layers):
                    K.transfer_kv_per_layer_mla_pf_lf(host, dst[l], hi, di, l, size, layers * size)
            else:
                K.transfer_kv_all_layer_mla(sp, hp, si, hi, size, layers)
                for l in range(layers):
                    K.transfer_kv_per_layer_mla(host[l], dst[l], hi, di, size)
            done = torch.cuda.Event()
            done.record()
        done.synchronize()
        for l in range(layers):
            expected_host = torch.full((7, size), 254, dtype=torch.uint8)
            expected_host[[1, 5, 2]] = src[l].cpu()[[4, 0, 6]]
            actual_host = host[:, l] if layout == "page_first" else host[l]
            self.assertTrue(torch.equal(actual_host, expected_host))
            expected_dst = torch.full((7, size), 253, dtype=torch.uint8)
            expected_dst[[6, 3, 0]] = src[l].cpu()[[4, 0, 6]]
            self.assertTrue(torch.equal(dst[l].cpu(), expected_dst))


if __name__ == "__main__":
    unittest.main()
