from __future__ import annotations

import os
from typing import Iterable, Optional

import dpctl


def _try_queue_selectors(selectors: Iterable[str]) -> dpctl.SyclQueue:
    """
    Try multiple SYCL device selectors and return the first queue that works.

    Args:
        selectors: Ordered list of SYCL filter strings.

    Returns:
        A dpctl.SyclQueue created from the first valid selector.

    Raises:
        RuntimeError: If none of the selectors can create a queue.
    """
    errors: list[str] = []

    for selector in selectors:
        try:
            return dpctl.SyclQueue(selector)
        except Exception as exc:
            errors.append(f"{selector}: {exc}")

    raise RuntimeError(
        "No SYCL device available for the requested selection. "
        + " | ".join(errors)
    )


def get_sycl_queue(device: str | None = "auto") -> dpctl.SyclQueue:
    """
    Create a dpctl.SyclQueue for the requested device.

    Supported values:
        - "auto": try CUDA GPU, Level Zero GPU, OpenCL GPU, generic GPU, then CPU
        - "gpu": try available GPU backends
        - "cpu": try available CPU backends
        - "cuda": try CUDA GPU selectors
        - explicit SYCL selectors such as "cuda:gpu", "level_zero:gpu",
          "opencl:gpu", "opencl:cpu"

    The environment variable SIGMO_SYCL_DEVICE can be used to override the
    selection, for example:

        SIGMO_SYCL_DEVICE=cuda:gpu
    """
    env_selector = os.environ.get("SIGMO_SYCL_DEVICE")
    if env_selector:
        return _try_queue_selectors([env_selector])

    selected = "auto" if device is None else str(device).lower()

    if selected == "auto":
        return _try_queue_selectors(
            [
                "cuda:gpu",
                "cuda:gpu:0",
                "level_zero:gpu",
                "opencl:gpu",
                "gpu",
                "opencl:cpu",
                "cpu",
            ]
        )

    if selected == "gpu":
        return _try_queue_selectors(
            [
                "cuda:gpu",
                "cuda:gpu:0",
                "level_zero:gpu",
                "opencl:gpu",
                "gpu",
            ]
        )

    if selected == "cuda":
        return _try_queue_selectors(
            [
                "cuda:gpu",
                "cuda:gpu:0",
            ]
        )

    if selected == "cpu":
        return _try_queue_selectors(
            [
                "opencl:cpu",
                "cpu",
            ]
        )

    return _try_queue_selectors([selected])


def get_default_queue() -> dpctl.SyclQueue:
    """
    Return the default SIGMo queue.

    This is a backward-compatible alias for get_sycl_queue("auto").
    """
    return get_sycl_queue("auto")


def describe_queue(queue: Optional[dpctl.SyclQueue]) -> str:
    """
    Return a human-readable device name for a SYCL queue.
    """
    if queue is None:
        return "unknown"

    try:
        return str(queue.sycl_device.name)
    except Exception:
        return "unknown"