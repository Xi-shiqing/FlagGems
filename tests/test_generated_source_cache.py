"""Different generated implementations must not overwrite each other's files."""
import hashlib
import importlib

import pytest
import torch

import flag_gems


@pytest.mark.parametrize("kind", ["pad", "index", "index_linearized"])
def test_generated_versions_keep_separate_source(monkeypatch, tmp_path, kind):
    module = importlib.import_module("flag_gems.ops." + ("pad" if kind == "pad" else "index"))
    monkeypatch.setattr(module, "code_cache_dir", lambda: tmp_path)
    generator_name = "generate_linearized_code" if kind == "index_linearized" else "generate_code"
    original_generator = getattr(module, generator_name)
    x = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7).to(flag_gems.device)
    index = torch.tensor([0, 2], device=x.device)
    if kind == "index_linearized" and not module._linearized_adjacent_index_configs(x, [index]):
        pytest.skip("adjacent-index configuration is unavailable on this backend")
    expected = torch.nn.functional.pad(x.cpu(), (1, 2)) if kind == "pad" else x.cpu()[[0, 2]]
    with torch.no_grad():
        for version in ["a", "b"]:
            def generate(*args, _version=version, **kwargs):
                code = original_generator(*args, **kwargs)
                code.writeline("# cached implementation " + _version)
                return code
            monkeypatch.setattr(module, generator_name, generate)
            if kind == "pad":
                output = module.PadFunction()(x, (1, 2), "constant", 0.0)
            elif kind == "index":
                output = torch.empty((2, 5, 7), device=x.device)
                module.IndexFunction()(x, [index], output)
            else:
                output = torch.empty((2, 5, 7), device=x.device)
                module.LinearizedAdjacentIndexFunction()(x, 0, [index], output)
            assert torch.equal(output.cpu(), expected)
    sources = list(tmp_path.glob("*.py"))
    assert len(sources) == 2
    for source in sources:
        assert source.stem.endswith(hashlib.sha256(source.read_bytes()).hexdigest()[:16])
