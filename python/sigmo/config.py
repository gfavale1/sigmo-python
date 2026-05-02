from __future__ import annotations

from typing import Optional
import dpctl


def get_sycl_queue(device: str = "auto", *, verbose: bool = False) -> dpctl.SyclQueue:
    """
    Crea una coda SYCL in modo user-friendly.

    device:
        - "auto": prova GPU, poi CPU
        - "gpu": richiede una GPU
        - "cpu": richiede una CPU
        - qualsiasi filtro accettato da dpctl.SyclQueue, es. "level_zero:gpu"
    """
    device = (device or "auto").lower()

    attempts = []
    if device == "auto":
        attempts = ["gpu", "cpu"]
    elif device in {"gpu", "cpu"}:
        attempts = [device]
    else:
        attempts = [device]

    errors = []
    for selector in attempts:
        try:
            if selector == "gpu":
                sycl_device = dpctl.select_gpu_device()
                queue = dpctl.SyclQueue(sycl_device)
            elif selector == "cpu":
                sycl_device = dpctl.select_cpu_device()
                queue = dpctl.SyclQueue(sycl_device)
            else:
                queue = dpctl.SyclQueue(selector)

            if verbose:
                print(f"[SIGMo] Selected SYCL device: {queue.sycl_device.name}")
            return queue
        except Exception as exc:  # dpctl puo' sollevare eccezioni diverse in base al backend
            errors.append(f"{selector}: {exc}")

    raise RuntimeError(
        "Nessun dispositivo SYCL disponibile per la selezione richiesta. "
        + " | ".join(errors)
    )


def get_default_queue() -> dpctl.SyclQueue:
    """Alias retrocompatibile: GPU se disponibile, altrimenti CPU."""
    return get_sycl_queue("auto")


def describe_queue(queue: Optional[dpctl.SyclQueue]) -> str:
    if queue is None:
        return "unknown"
    try:
        return str(queue.sycl_device.name)
    except Exception:
        return "unknown"
