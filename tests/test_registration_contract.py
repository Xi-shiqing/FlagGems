"""Check actual dispatcher declarations and repeated context registration."""
import logging

import torch

import flag_gems


def test_extra_dispatch_key_specifications():
    for entry in flag_gems._FULL_CONFIG:
        if len(entry) < 4:
            continue
        for specification in entry[3]:
            if isinstance(specification, str):
                continue
            assert isinstance(specification, tuple), entry[0]
            assert len(specification) == 2, entry[0]
            assert isinstance(specification[0], str), entry[0]
            assert callable(specification[1]), entry[0]


def test_unbind_copy_repeated_registration(caplog):
    original = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7)
    device_input = original.to(flag_gems.device)
    expected = list(original.unbind(1))
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            with flag_gems.use_gems(include=["unbind_copy"]):
                actual = torch.unbind_copy(device_input, dim=1)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value.cpu(), reference, rtol=0, atol=0)
                assert value.untyped_storage().data_ptr() != device_input.untyped_storage().data_ptr()
    assert not any("Skipped registering operator" in record.message for record in caplog.records)
