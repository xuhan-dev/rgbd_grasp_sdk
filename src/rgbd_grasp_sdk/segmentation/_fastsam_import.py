from __future__ import annotations

from typing import Any


class FastSamBackend:
    def __init__(self, options: dict[str, Any]):
        from fastsam import FastSAM, FastSAMPrompt

        self.options = options
        self.model = FastSAM(options["model_path"])
        self.prompt_cls = FastSAMPrompt

    def segment_text(self, image, text):
        result = self.model(
            image,
            device=self.options.get("device", "cuda"),
            retina_masks=bool(self.options.get("retina", True)),
            imgsz=int(self.options.get("image_size", 1280)),
            conf=float(self.options.get("confidence", 0.2)),
            iou=float(self.options.get("iou", 0.6)),
            save=False,
        )
        prompt = self.prompt_cls(image, result, device=self.options.get("device", "cuda"))
        ann = prompt.text_prompt(text=text)
        preview = prompt.plot(annotations=ann, output_path=None)
        if ann is None or len(ann) == 0:
            return None, preview
        return ann[0].astype(bool), preview
