from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from hcu_adapter.mapped_host_dtk import _install_hip_library_lookup


@pytest.mark.parametrize("name,primary,alias,expected", [
    ("libamdhip64", "/lib/libamdhip64.so.6", None, "/lib/libamdhip64.so.6"),
    ("libamdhip64", None, "/dtk/libgalaxyhip.so.5.2", "/dtk/libgalaxyhip.so.5.2"),
    ("libamdhip64", None, "/dtk/libgalaxyhip.so", "/dtk/libgalaxyhip.so"),
    ("libamdhip64", None, "/dtk/libgalaxyhip_stub.so", None),
    ("libamdhip64", None, None, None),
    ("libcudart", None, "/dtk/libgalaxyhip.so", None),
])
def test_lookup(name, primary, alias, expected):
    original = Mock(side_effect=[primary, alias])
    wrapper = SimpleNamespace(find_loaded_library=original)
    _install_hip_library_lookup(wrapper)
    lookup = wrapper.find_loaded_library
    _install_hip_library_lookup(wrapper)
    assert wrapper.find_loaded_library is lookup
    assert lookup(lib_name=name) == expected
    calls = [call(name)]
    if primary is None and name == "libamdhip64":
        calls.append(call("libgalaxyhip"))
    assert original.call_args_list == calls
