from __future__ import annotations

from enum import Enum

from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import is_hip


def _is_sm8x() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8


class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"
    TRITON = "triton"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    def is_triton(self) -> bool:
        return self == DSAPagedMQALogitsBackend.TRITON

    @staticmethod
    def resolve(value: str) -> DSAPagedMQALogitsBackend:
        if is_hip():
            if value not in ("auto", "aiter"):
                raise ValueError(
                    f"dsa_paged_mqa_logits_backend={value!r} is not supported on "
                    "ROCm; only 'aiter' is implemented."
                )
            return DSAPagedMQALogitsBackend.AITER

        if value == "triton" or (value == "auto" and _is_sm8x()):
            # F28's Triton kernels (f28_mqa.py): DeepGEMM has no SM8x kernels.
            return DSAPagedMQALogitsBackend.TRITON
        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "cutedsl":
            if not get_platform().is_sm100:
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")
