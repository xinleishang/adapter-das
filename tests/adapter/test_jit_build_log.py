# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JIT build progress reporter pins (observability, no GPU / no sglang).

The gate's cold-JIT storms must never look like a wedge again (2026-10-20
user report: hour-long silent compiles). wrap() on a stub ninja module must:

* emit compiling/built lines around a successful build, elapsed > 0;
* emit the FAILED line and re-raise when the real builder raises;
* be idempotent (second wrap is a no-op -- sitecustomize re-runs in every
  spawned child);
* write the journal via O_APPEND so pytest capture cannot swallow it, and
  honor HCU_JIT_PROGRESS_LOG="" as "journal disabled" (stderr only);
* keep the real builder's return value verbatim.

Run: python tests/adapter/test_jit_build_log.py  (or pytest; collected by
the L1 kernels segment together with tests/adapter).
"""

import os
import sys
import types
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter import jit_build_log  # noqa: E402


class _Spec:
    def __init__(self, name):
        self.module_name = name


@pytest.fixture()
def journal(tmp_path, monkeypatch):
    path = str(tmp_path / "jit_progress.log")
    monkeypatch.setenv("HCU_JIT_PROGRESS_LOG", path)
    return path


def _read_journal(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def _stub_builder(ok=True, result="stub.so"):
    calls = []

    def build(*, spec, build_dir, build_file):
        calls.append(spec.module_name)
        if not ok:
            raise RuntimeError(f"ninja boom: {spec.module_name}")
        return f"{build_dir}/{result}"

    return build, calls


def test_success_emits_compiling_and_built(journal, capsys):
    mod = types.SimpleNamespace(build=_stub_builder()[0])
    jit_build_log.wrap(mod)

    so = mod.build(spec=_Spec("sgl_kernel_jit_metadata"),
                   build_dir="/tmp/b", build_file="build.ninja")
    assert so == "/tmp/b/stub.so"

    lines = _read_journal(journal)
    assert lines[0].startswith("HCU JIT: compiling sgl_kernel_jit_metadata pid=")
    assert "/tmp/b" in lines[0]
    assert lines[-1].startswith("HCU JIT: built sgl_kernel_jit_metadata in ")
    assert "-> /tmp/b/stub.so" in lines[-1]
    assert " in 0s " in lines[-1] or " in 1s " in lines[-1]  # elapsed present
    # stderr mirror also fired (dual sink)
    assert "HCU JIT: compiling" in capsys.readouterr().err


def test_failure_emits_failed_line_and_reraises(journal):
    real, calls = _stub_builder(ok=False)
    mod = types.SimpleNamespace(build=real)
    jit_build_log.wrap(mod)

    with pytest.raises(RuntimeError, match="ninja boom"):
        mod.build(spec=_Spec("sgl_kernel_jit_dpsk_v4_topk"),
                  build_dir="/tmp/b", build_file="build.ninja")
    assert calls == ["sgl_kernel_jit_dpsk_v4_topk"]

    lines = _read_journal(journal)
    assert lines[0].startswith("HCU JIT: compiling sgl_kernel_jit_dpsk_v4_topk")
    assert lines[-1].startswith("HCU JIT: FAILED sgl_kernel_jit_dpsk_v4_topk in ")
    # retry-by-caller re-enters a fresh pair of lines (the retry-storm killer)
    with pytest.raises(RuntimeError):
        mod.build(spec=_Spec("sgl_kernel_jit_dpsk_v4_topk"),
                  build_dir="/tmp/b2", build_file="build.ninja")
    assert len(_read_journal(journal)) == 4


def test_wrap_is_idempotent(journal):
    real, calls = _stub_builder()
    mod = types.SimpleNamespace(build=real)
    assert jit_build_log.wrap(mod) is mod
    wrapped = mod.build
    assert jit_build_log.wrap(mod) is mod
    assert mod.build is wrapped  # second wrap must not layer a second probe

    mod.build(spec=_Spec("m"), build_dir="/d", build_file="bf")
    assert len(_read_journal(journal)) == 2  # one compiling + one built


def test_journal_disabled_by_empty_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HCU_JIT_PROGRESS_LOG", "")
    mod = types.SimpleNamespace(build=_stub_builder()[0])
    jit_build_log.wrap(mod)

    mod.build(spec=_Spec("m"), build_dir="/d", build_file="bf")
    assert not list(tmp_path.iterdir())  # nothing written anywhere


def test_ninja_child_does_not_reenter_adapter_or_change_parent(monkeypatch, journal):
    monkeypatch.setenv("PYTHONPATH", "/adapter:/official")
    monkeypatch.setenv("HCU_ADAPTER_HOST", "sglang")
    monkeypatch.setenv("_HCU_ADAPTER_ACTIVE_HOST", "sglang")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    parent = dict(os.environ)
    run = Mock(return_value="built")
    subprocess_module = types.SimpleNamespace(run=run, PIPE=-1)
    mod = types.ModuleType("stub_ninja")
    mod.subprocess = subprocess_module
    exec("def build(*, spec, build_dir, build_file):\n    return subprocess.run(['ninja', '-f', build_file], cwd=build_dir, check=False)\n", mod.__dict__)
    cached_alias = mod.build
    jit_build_log.wrap(mod)
    facade = mod.subprocess
    jit_build_log.wrap(mod)
    assert mod.subprocess is facade
    assert mod.subprocess.PIPE == -1
    assert subprocess_module.run is run
    assert cached_alias(spec=_Spec("m"), build_dir="/tmp/b", build_file="build.ninja") == "built"
    env = run.call_args.kwargs["env"]
    assert "PYTHONPATH" not in env
    assert env["HCU_ADAPTER_HOST"] == env["_HCU_ADAPTER_ACTIVE_HOST"] == "none"
    assert env["HIP_VISIBLE_DEVICES"] == "3"
    assert dict(os.environ) == parent


def test_ninja_child_preserves_explicit_environment_and_failure():
    run = Mock(side_effect=RuntimeError("compile failed"))
    facade = jit_build_log._BuildSubprocess(types.SimpleNamespace(run=run))
    env = {"PYTHONPATH": "/adapter", "CC": "compiler", "HCU_ADAPTER_HOST": "sglang"}
    with pytest.raises(RuntimeError, match="compile failed"):
        facade.run(["/usr/bin/ninja"], env=env, timeout=30)
    assert run.call_args.kwargs["env"]["CC"] == "compiler"
    assert run.call_args.kwargs["env"]["HCU_ADAPTER_HOST"] == "none"
    assert env["PYTHONPATH"] == "/adapter"
    assert env["HCU_ADAPTER_HOST"] == "sglang"
    assert run.call_args.kwargs["timeout"] == 30


def test_non_ninja_child_keeps_original_environment():
    run = Mock()
    facade = jit_build_log._BuildSubprocess(types.SimpleNamespace(run=run))
    env = {"PYTHONPATH": "/adapter", "HCU_ADAPTER_HOST": "sglang"}
    facade.run(["compiler", "input.cpp"], env=env)
    assert run.call_args.kwargs["env"] is env


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
