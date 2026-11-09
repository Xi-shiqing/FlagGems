// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/EmptyTensor.h>
#include <torch/csrc/utils/pybind.h>

// Uses the normal ATen CUDA allocator and its stream/lifetime handling. The
// caller must overwrite every logical element before consuming the tensor.
at::Tensor empty_full_write(const std::vector<int64_t>& shape,
                            c10::ScalarType dtype,
                            c10::Device device) {
  TORCH_CHECK(device.is_cuda(), "full-write allocation requires a CUDA device");
  return at::Tensor(at::detail::empty_cuda(
      shape, dtype, device, c10::MemoryFormat::Contiguous));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("empty_full_write", &empty_full_write,
             "Allocate output for a kernel that writes every logical element");
}
