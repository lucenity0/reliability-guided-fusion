import os
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hmf.utils.seed import restrict_sdpa, seed_everything


@pytest.fixture(autouse=True)
def restore_rngs_and_environment(monkeypatch):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def test_python_numpy_determinism_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    def sample(seed):
        seed_everything(seed)
        return [random.random() for _ in range(8)], np.random.random(8)

    first, repeated, different = sample(42), sample(42), sample(43)
    assert first[0] == repeated[0]
    np.testing.assert_array_equal(first[1], repeated[1])
    assert first[0] != different[0]
    assert not np.array_equal(first[1], different[1])


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "1"])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed)


def test_invalid_deterministic_flag():
    with pytest.raises(ValueError, match="deterministic"):
        seed_everything(0, deterministic="false")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:1"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("initial", [(True, True, False), (False, False, True)])
def test_restrict_sdpa_flags(monkeypatch, device, deterministic, initial):
    torch = pytest.importorskip("torch")
    flags = dict(zip(("flash", "mem_efficient", "math"), initial, strict=True))
    setters = {}
    for name in flags:
        setters[name] = Mock(side_effect=lambda enabled, name=name: flags.update({name: enabled}))
        monkeypatch.setattr(torch.backends.cuda, f"enable_{name}_sdp", setters[name])
        monkeypatch.setattr(torch.backends.cuda, f"{name}_sdp_enabled",
                            lambda name=name: flags[name])

    result = restrict_sdpa(torch.device(device), deterministic=deterministic)

    if deterministic and device.startswith("cuda"):
        assert result == {"flash": False, "mem_efficient": False, "math": True}
        for name, enabled in result.items():
            setters[name].assert_called_once_with(enabled)
    else:
        assert tuple(result.values()) == initial
        for setter in setters.values():
            setter.assert_not_called()
    assert result == flags


def test_torch_flags_and_cublas_configuration(monkeypatch):
    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    torch = SimpleNamespace(
        manual_seed=Mock(),
        use_deterministic_algorithms=Mock(),
        backends=SimpleNamespace(cudnn=cudnn),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    seed_everything(7)
    torch.manual_seed.assert_called_once_with(7)
    torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=True)
    assert cudnn.deterministic is True
    assert cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_everything(8)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    seed_everything(9, deterministic=False)
    torch.use_deterministic_algorithms.assert_called_with(False, warn_only=True)
    assert cudnn.deterministic is False
    assert cudnn.benchmark is True


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_real_torch_repeatability(device):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_flags = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    accelerator = getattr(torch, device) if device != "cpu" else None
    accelerator_state = accelerator.get_rng_state() if accelerator else None
    try:
        seed_everything(123)
        first = torch.rand(32, device=device)
        seed_everything(123)
        assert torch.equal(first, torch.rand(32, device=device))
        seed_everything(124)
        assert not torch.equal(first, torch.rand(32, device=device))
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.set_rng_state(state)
        if accelerator:
            accelerator.set_rng_state(accelerator_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
