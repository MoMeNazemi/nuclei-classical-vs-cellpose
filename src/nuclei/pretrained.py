"""Pretrained Cellpose models.

Cellpose works in two stages:
1. A neural network predicts, for every pixel, a "flow" (which way to the
   centre of its cell) and a cell probability. This is the slow part (GPU).
2. Pixels are followed along the flows; pixels that end up at the same centre
   form one object. Two thresholds act here: cellprob_threshold (which pixels
   count as cell) and flow_threshold (how well an object's flows must agree
   with its shape for it to be kept).

Because the thresholds only affect stage 2, we run the network once per image
and then build masks for every threshold setting from the same network output.

Cellpose is imported inside the functions, so the rest of the project still
works in an environment without it.
"""

from __future__ import annotations

import numpy as np

CELLPOSE_VERSION = "4.2.1.1"  # the version this code was written and tested against
DEFAULTS = {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "min_size": 15}
NITER = 200  # dynamics steps; what model.eval uses at diameter=None


class CellposeRunner:
    def __init__(self, model_name: str, gpu: bool = True):
        from cellpose import models

        self.model_name = model_name
        self.model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
        self.device = self.model.device

    def network(self, img: np.ndarray):
        """Stage 1: returns (flows, cell probability) for one 2D image.

        diameter=None runs the image at its own size. That is right here:
        the models were trained around a 30 px mean cell diameter, and the
        median BBBC039 nucleus is 28 px across.
        """
        _, flows, _ = self.model.eval(img, diameter=None, normalize=True, compute_masks=False)
        return flows[1], flows[2]

    def masks(self, dP: np.ndarray, cellprob: np.ndarray, **settings) -> np.ndarray:
        """Stage 2: labelled nuclei from the network output."""
        return masks_from_output(dP, cellprob, device=self.device, **settings)

    def segment(self, img: np.ndarray, **settings) -> np.ndarray:
        """Both stages in one call, through Cellpose's own entry point."""
        s = {**DEFAULTS, **settings}
        masks, _, _ = self.model.eval(img, diameter=None, normalize=True, **s)
        return masks

    def release(self) -> None:
        """Free GPU memory before loading another model."""
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def masks_from_output(dP: np.ndarray, cellprob: np.ndarray,
                      flow_threshold: float = DEFAULTS["flow_threshold"],
                      cellprob_threshold: float = DEFAULTS["cellprob_threshold"],
                      min_size: int = DEFAULTS["min_size"], device=None) -> np.ndarray:
    """Labelled nuclei from flows and cell probability.

    Calls the same function, with the same settings, that model.eval calls
    for a single 2D image at diameter=None.
    """
    import torch
    from cellpose import dynamics

    return dynamics.resize_and_compute_masks(
        dP, cellprob, niter=NITER, cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold, resize=None, min_size=min_size,
        max_size_fraction=0.4, device=device if device is not None else torch.device("cpu"))


def environment_info() -> dict:
    """Versions and hardware, saved with the results so they can be reproduced."""
    from importlib.metadata import version

    import torch

    info = {
        "cellpose": version("cellpose"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
