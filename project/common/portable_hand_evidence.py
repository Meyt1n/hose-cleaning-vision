"""The validated R19 -> R29 fallback policy with relocatable weights."""

import copy
import hashlib
from pathlib import Path

import torch

from common.glove_evidence import GloveHandEvidence


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PortableHandEvidence(GloveHandEvidence):
    def __init__(self, manifest, fallback_file, fallback_sha256, **kwargs):
        super().__init__(manifest, **kwargs)
        if sha256_file(fallback_file) != fallback_sha256:
            raise ValueError("R29 fallback SHA256 mismatch")
        secondary = copy.deepcopy(self.detector)
        delta = torch.load(fallback_file, map_location="cpu", weights_only=True)["model"]
        state = secondary.state_dict()
        if set(delta) != {key for key in state if not key.startswith("backbone.body.")}:
            raise ValueError("R29 mutable key contract mismatch")
        state.update(delta)
        secondary.load_state_dict(state, strict=True)
        secondary.backbone.body = self.detector.backbone.body
        self.secondary = secondary.requires_grad_(False).eval()
        self.primary_calls = self.fallback_attempts = self.fallback_successes = 0

    @torch.inference_mode()
    def predict(self, *args, **kwargs):
        self.primary_calls += 1
        result = super().predict(*args, **kwargs)
        result["R29_fallback_used"] = False
        if result["present"]:
            return result
        primary = self.detector
        self.detector = self.secondary
        self.fallback_attempts += 1
        try:
            result = super().predict(*args, **kwargs)
        finally:
            self.detector = primary
        result["R29_fallback_used"] = bool(result["present"])
        if result["present"]:
            self.fallback_successes += 1
            result["source"] = "R29_missing_hand_fallback_R18_pose"
            for hand in result["hands"]:
                hand["source"] = "R29_detector_R18_Hand5"
        return result
