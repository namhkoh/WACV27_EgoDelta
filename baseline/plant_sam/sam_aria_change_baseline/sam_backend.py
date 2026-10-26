from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a requested SAM backend is not importable or usable."""


@dataclass
class SamMask:
    mask_id: str
    prompt: str
    score: float
    bbox_xyxy: list[float]
    mask: np.ndarray
    area_px: int
    source: str = "text"
    raw_class_name: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "prompt": self.prompt,
            "score": float(self.score),
            "bbox_xyxy": [float(x) for x in self.bbox_xyxy],
            "area_px": int(self.area_px),
            "source": self.source,
            "raw_class_name": self.raw_class_name,
        }


class SamBackend(ABC):
    name: str = "base"
    supports_exemplars: bool = False
    supports_video: bool = False

    @abstractmethod
    def predict_image(
        self,
        image: np.ndarray,
        text_prompts: list[str],
        exemplar_boxes: list[list[float]] | None = None,
        exemplar_images: list[np.ndarray] | None = None,
        conf_thres: float = 0.25,
    ) -> list[SamMask]:
        raise NotImplementedError

    def predict_video(
        self,
        frame_paths: list[str],
        text_prompts: list[str],
        exemplar_boxes: list[list[float]] | None = None,
        exemplar_images: list[np.ndarray] | None = None,
        conf_thres: float = 0.25,
    ) -> dict[int, list[SamMask]]:
        raise NotImplementedError(f"{self.name} does not expose video mode in this adapter.")

    def predict_images(
        self,
        images: list[np.ndarray],
        text_prompts: list[str],
        exemplar_boxes: list[list[float]] | None = None,
        exemplar_images: list[np.ndarray] | None = None,
        conf_thres: float = 0.25,
    ) -> list[list[SamMask]]:
        return [
            self.predict_image(
                image=image,
                text_prompts=text_prompts,
                exemplar_boxes=exemplar_boxes,
                exemplar_images=exemplar_images,
                conf_thres=conf_thres,
            )
            for image in images
        ]


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        tensor = value.detach()
        dtype_name = str(getattr(tensor, "dtype", ""))
        if "bfloat16" in dtype_name or "float16" in dtype_name:
            tensor = tensor.float()
        return tensor.cpu().numpy()
    if hasattr(value, "cpu"):
        tensor = value
        dtype_name = str(getattr(tensor, "dtype", ""))
        if "bfloat16" in dtype_name or "float16" in dtype_name:
            tensor = tensor.float()
        return tensor.cpu().numpy()
    return np.asarray(value)


def _as_numpy_mask(mask: Any, image_shape: tuple[int, int]) -> np.ndarray:
    mask = _to_numpy(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected HxW mask, got shape {mask.shape}.")
    mask = mask.astype(bool)
    if mask.shape != image_shape:
        raise ValueError(
            f"Backend returned mask shape {mask.shape}, expected image shape {image_shape}."
        )
    return mask


def _normalize_result_like(
    result: Any,
    prompt: str,
    image_shape: tuple[int, int],
    conf_thres: float,
    mask_id_prefix: str,
) -> list[SamMask]:
    if isinstance(result, dict):
        masks_obj = result.get("masks")
        if masks_obj is None:
            return []
        raw_masks = _to_numpy(masks_obj)
        if raw_masks.ndim == 2:
            raw_masks = raw_masks[None, :, :]
        scores = _to_numpy(result.get("scores", np.ones((raw_masks.shape[0],), dtype=float)))
        xyxy = _to_numpy(result["boxes"]) if result.get("boxes") is not None else None
        classes = _to_numpy(result["classes"]) if result.get("classes") is not None else None
        names = result.get("names", {}) or {}
    else:
        masks_obj = getattr(result, "masks", None)
        boxes_obj = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}

        if masks_obj is None:
            return []

        raw_masks = getattr(masks_obj, "data", masks_obj)
        raw_masks = _to_numpy(raw_masks)
        if raw_masks.ndim == 2:
            raw_masks = raw_masks[None, :, :]

        xyxy = None
        classes = None
        if boxes_obj is not None:
            conf = getattr(boxes_obj, "conf", None)
            if conf is not None:
                scores = _to_numpy(conf)
            else:
                scores = np.ones((raw_masks.shape[0],), dtype=float)
            box_xyxy = getattr(boxes_obj, "xyxy", None)
            if box_xyxy is not None:
                xyxy = _to_numpy(box_xyxy)
            cls = getattr(boxes_obj, "cls", None)
            if cls is not None:
                classes = _to_numpy(cls)
        else:
            scores = np.ones((raw_masks.shape[0],), dtype=float)

    scores = np.asarray(scores).reshape((-1,))
    if xyxy is not None:
        xyxy = np.asarray(xyxy).reshape((-1, 4))
    if classes is not None:
        classes = np.asarray(classes).reshape((-1,))

    if raw_masks.ndim == 4 and raw_masks.shape[1] == 1:
        raw_masks = raw_masks[:, 0]

    normalized: list[SamMask] = []
    for idx, raw_mask in enumerate(raw_masks):
        score = float(scores[idx]) if idx < len(scores) else 1.0
        if score < conf_thres:
            continue
        mask = _as_numpy_mask(raw_mask, image_shape)
        bbox = (
            [float(x) for x in xyxy[idx].tolist()]
            if xyxy is not None and idx < len(xyxy)
            else _bbox_from_mask(mask)
        )
        raw_class_name = None
        if classes is not None and idx < len(classes):
            cls_id = int(classes[idx])
            raw_class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        normalized.append(
            SamMask(
                mask_id=f"{mask_id_prefix}_{idx:03d}",
                prompt=prompt,
                score=score,
                bbox_xyxy=bbox,
                mask=mask,
                area_px=int(mask.sum()),
                source="text",
                raw_class_name=raw_class_name,
            )
        )
    return normalized


def _normalize_legacy_result_like(
    result: Any,
    prompt: str,
    image_shape: tuple[int, int],
    conf_thres: float,
    mask_id_prefix: str,
) -> list[SamMask]:
    """Best-effort normalizer for common prediction result objects."""
    masks_obj = getattr(result, "masks", None)
    boxes_obj = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}

    if masks_obj is None:
        if isinstance(result, dict):
            masks_obj = result.get("masks")
            boxes_obj = result.get("boxes")
            names = result.get("names", names) or {}
        else:
            return []

    raw_masks = getattr(masks_obj, "data", masks_obj)
    if hasattr(raw_masks, "detach"):
        raw_masks = raw_masks.detach().cpu().numpy()
    elif hasattr(raw_masks, "cpu"):
        raw_masks = raw_masks.cpu().numpy()
    raw_masks = np.asarray(raw_masks)
    if raw_masks.ndim == 2:
        raw_masks = raw_masks[None, :, :]

    scores: np.ndarray
    xyxy: np.ndarray | None = None
    classes: np.ndarray | None = None
    if boxes_obj is not None:
        conf = getattr(boxes_obj, "conf", None)
        if conf is not None:
            scores = np.asarray(conf.detach().cpu().numpy() if hasattr(conf, "detach") else conf)
        else:
            scores = np.ones((raw_masks.shape[0],), dtype=float)
        box_xyxy = getattr(boxes_obj, "xyxy", None)
        if box_xyxy is not None:
            xyxy = np.asarray(
                box_xyxy.detach().cpu().numpy() if hasattr(box_xyxy, "detach") else box_xyxy
            )
        cls = getattr(boxes_obj, "cls", None)
        if cls is not None:
            classes = np.asarray(cls.detach().cpu().numpy() if hasattr(cls, "detach") else cls)
    elif isinstance(result, dict):
        scores = np.asarray(result.get("scores", np.ones((raw_masks.shape[0],), dtype=float)))
        if result.get("boxes") is not None:
            xyxy = np.asarray(result["boxes"])
        if result.get("classes") is not None:
            classes = np.asarray(result["classes"])
    else:
        scores = np.ones((raw_masks.shape[0],), dtype=float)

    normalized: list[SamMask] = []
    for idx, raw_mask in enumerate(raw_masks):
        score = float(scores[idx]) if idx < len(scores) else 1.0
        if score < conf_thres:
            continue
        mask = _as_numpy_mask(raw_mask, image_shape)
        bbox = (
            [float(x) for x in xyxy[idx].tolist()]
            if xyxy is not None and idx < len(xyxy)
            else _bbox_from_mask(mask)
        )
        raw_class_name = None
        if classes is not None and idx < len(classes):
            cls_id = int(classes[idx])
            raw_class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        normalized.append(
            SamMask(
                mask_id=f"{mask_id_prefix}_{idx:03d}",
                prompt=prompt,
                score=score,
                bbox_xyxy=bbox,
                mask=mask,
                area_px=int(mask.sum()),
                source="text",
                raw_class_name=raw_class_name,
            )
        )
    return normalized


class NativeSam3Backend(SamBackend):
    name = "native_sam3"
    supports_exemplars = False

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        try:
            import sam3  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name == "sam3":
                raise BackendUnavailableError(
                    "SAM backend native_sam3 is not installed. Run "
                    "scripts/setup_sam3_native.sh or use --sam-backend ultralytics_sam3."
                ) from exc
            raise BackendUnavailableError(
                "SAM backend native_sam3 is installed, but importing it failed because a "
                f"dependency is missing: {exc.name}. Install the missing dependency in the "
                "sam3_aria environment, then rerun the pipeline."
            ) from exc
        except Exception as exc:
            raise BackendUnavailableError(
                "SAM backend native_sam3 is installed, but importing it failed. Run "
                "`python -c \"import sam3\"` inside the same environment to see the full "
                f"traceback. Error: {exc}"
            ) from exc

        try:
            import torch

            self._torch = torch
            env_device = os.environ.get("SAM3_DEVICE") or os.environ.get("SAM_DEVICE")
            requested_device = device or env_device
            if requested_device and str(requested_device).startswith("cuda:"):
                torch.cuda.set_device(requested_device)
                self.device = "cuda"
            else:
                self.device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self._torch = None
            env_device = os.environ.get("SAM3_DEVICE") or os.environ.get("SAM_DEVICE")
            self.device = device or env_device or "cpu"
        self.model_path = model_path
        self.amp_dtype_name = os.environ.get("SAM3_NATIVE_AMP_DTYPE", "bfloat16").lower()
        self.model_dtype_name = os.environ.get("SAM3_NATIVE_MODEL_DTYPE", "").lower()
        self.processor = self._build_predictor()

    def _autocast_context(self):
        if self._torch is None or not str(self.device).startswith("cuda"):
            return nullcontext()
        if self.amp_dtype_name in ("none", "off", "false", "0", "float32", "fp32"):
            return self._torch.autocast(device_type="cuda", enabled=False)
        if self.amp_dtype_name in ("float16", "fp16", "half"):
            dtype = self._torch.float16
        else:
            dtype = self._torch.bfloat16
        return self._torch.autocast(device_type="cuda", dtype=dtype)

    def _build_predictor(self) -> Any:
        errors: list[str] = []
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model(checkpoint_path=self.model_path, device=self.device)
            if self._torch is not None and self.model_dtype_name:
                if self.model_dtype_name in ("bfloat16", "bf16"):
                    model = model.to(dtype=self._torch.bfloat16)
                elif self.model_dtype_name in ("float16", "fp16", "half"):
                    model = model.to(dtype=self._torch.float16)
                elif self.model_dtype_name in ("float32", "fp32"):
                    model = model.to(dtype=self._torch.float32)
            return Sam3Processor(model, device=self.device)
        except Exception as exc:  # pragma: no cover - depends on external package API.
            errors.append(str(exc))

        raise BackendUnavailableError(
            "native_sam3 is importable, but this adapter could not construct an image "
            "processor. Check Hugging Face checkpoint access and the installed "
            "facebookresearch/sam3 API. Last error: " + " | ".join(errors)
        )

    def predict_image(
        self,
        image: np.ndarray,
        text_prompts: list[str],
        exemplar_boxes: list[list[float]] | None = None,
        exemplar_images: list[np.ndarray] | None = None,
        conf_thres: float = 0.25,
    ) -> list[SamMask]:
        if exemplar_boxes or exemplar_images:
            print(
                "Warning: native_sam3 adapter v1 does not expose exemplar prompts; "
                "continuing with text-only prompts."
            )
        from PIL import Image

        h, w = image.shape[:2]
        results: list[SamMask] = []
        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        self.processor.set_confidence_threshold(conf_thres)
        with self._autocast_context():
            state = self.processor.set_image(pil_image)
            for prompt in text_prompts:
                self.processor.reset_all_prompts(state)
                raw = self.processor.set_text_prompt(prompt=prompt, state=state)
                results.extend(
                    _normalize_result_like(raw, prompt, (h, w), conf_thres, f"{prompt}_native")
                )
        for idx, mask in enumerate(results):
            mask.mask_id = f"sam3_native_{idx:05d}"
        return results

    def _predict_batch_prompt(
        self,
        state: dict[str, Any],
        prompt: str,
        image_shapes: list[tuple[int, int]],
        conf_thres: float,
    ) -> list[list[SamMask]]:
        if self._torch is None:
            raise RuntimeError("Torch is required for native SAM3 batched prediction.")
        from sam3.model import box_ops
        from sam3.model.data_misc import FindStage, interpolate

        torch = self._torch
        batch_size = len(image_shapes)
        text_outputs = self.processor.model.backbone.forward_text([prompt], device=self.device)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.processor.model._get_dummy_prompt(num_prompts=batch_size)

        find_stage = FindStage(
            img_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
            text_ids=torch.zeros(batch_size, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        outputs = self.processor.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        batch_results: list[list[SamMask]] = []
        for image_idx, (height, width) in enumerate(image_shapes):
            keep = out_probs[image_idx] > conf_thres
            scores = out_probs[image_idx][keep]
            masks = out_masks[image_idx][keep]
            boxes = out_bbox[image_idx][keep]
            if int(keep.sum()) == 0:
                batch_results.append([])
                continue
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            scale_fct = torch.tensor([width, height, width, height], device=self.device)
            boxes = boxes * scale_fct[None, :]
            masks = interpolate(
                masks.unsqueeze(1),
                (height, width),
                mode="bilinear",
                align_corners=False,
            ).sigmoid() > 0.5
            raw = {
                "masks": masks[:, 0],
                "boxes": boxes,
                "scores": scores,
            }
            batch_results.append(
                _normalize_result_like(
                    raw,
                    prompt,
                    (height, width),
                    conf_thres,
                    f"{prompt}_native_b{image_idx}",
                )
            )
        return batch_results

    def predict_images(
        self,
        images: list[np.ndarray],
        text_prompts: list[str],
        exemplar_boxes: list[list[float]] | None = None,
        exemplar_images: list[np.ndarray] | None = None,
        conf_thres: float = 0.25,
    ) -> list[list[SamMask]]:
        if len(images) <= 1:
            return super().predict_images(
                images=images,
                text_prompts=text_prompts,
                exemplar_boxes=exemplar_boxes,
                exemplar_images=exemplar_images,
                conf_thres=conf_thres,
            )
        if exemplar_boxes or exemplar_images:
            print(
                "Warning: native_sam3 adapter v1 does not expose exemplar prompts; "
                "continuing with text-only prompts."
            )
        from PIL import Image

        pil_images = [Image.fromarray(np.asarray(image, dtype=np.uint8)) for image in images]
        image_shapes = [tuple(image.shape[:2]) for image in images]
        self.processor.set_confidence_threshold(conf_thres)
        batch_results: list[list[SamMask]] = [[] for _ in images]
        try:
            with self._autocast_context():
                state = self.processor.set_image_batch(pil_images)
                for prompt in text_prompts:
                    self.processor.reset_all_prompts(state)
                    prompt_results = self._predict_batch_prompt(
                        state=state,
                        prompt=prompt,
                        image_shapes=image_shapes,
                        conf_thres=conf_thres,
                    )
                    for image_idx, masks in enumerate(prompt_results):
                        batch_results[image_idx].extend(masks)
        except Exception as exc:
            print(f"Warning: native_sam3 batched image prediction failed ({exc}); falling back to per-frame inference.")
            return super().predict_images(
                images=images,
                text_prompts=text_prompts,
                exemplar_boxes=exemplar_boxes,
                exemplar_images=exemplar_images,
                conf_thres=conf_thres,
            )
        for image_idx, masks in enumerate(batch_results):
            for mask_idx, mask in enumerate(masks):
                mask.mask_id = f"sam3_native_b{image_idx}_{mask_idx:05d}"
        return batch_results


class UltralyticsSam3Backend(SamBackend):
    name = "ultralytics_sam3"
    supports_exemplars = False

    def __init__(self, model: str = "sam3.pt", device: str | None = None) -> None:
        try:
            from ultralytics import SAM
        except Exception as exc:
            raise BackendUnavailableError(
                "SAM backend ultralytics_sam3 is not installed. Run "
                "scripts/setup_sam3_ultralytics.sh or use --sam-backend native_sam3."
            ) from exc
        self.device = device or os.environ.get("SAM3_DEVICE") or os.environ.get("SAM_DEVICE")
        try:
            self.model = SAM(model)
        except Exception as exc:
            raise BackendUnavailableError(
                "ultralytics is installed, but SAM 3 model construction failed. "
                "Ensure your ultralytics version exposes SAM 3 prediction."
            ) from exc

    def predict_image(
        self,
        image: np.ndarray,
        text_prompts: list[str],
        exemplar_boxes: list[list[float]] | None = None,
        exemplar_images: list[np.ndarray] | None = None,
        conf_thres: float = 0.25,
    ) -> list[SamMask]:
        if exemplar_boxes or exemplar_images:
            print(
                "Warning: ultralytics_sam3 adapter v1 does not expose exemplar prompts; "
                "continuing with text-only prompts."
            )
        h, w = image.shape[:2]
        results: list[SamMask] = []
        for prompt in text_prompts:
            raw_results = None
            candidate_kwargs = (
                {"texts": [prompt]},
                {"text": prompt},
                {"text_prompt": prompt},
                {"prompts": [prompt]},
            )
            last_error: Exception | None = None
            for extra in candidate_kwargs:
                kwargs = {"source": image, "conf": conf_thres, "verbose": False, **extra}
                if self.device is not None:
                    kwargs["device"] = self.device
                try:
                    raw_results = self.model.predict(**kwargs)
                    break
                except TypeError as exc:
                    last_error = exc
                    continue
            if raw_results is None:
                raise BackendUnavailableError(
                    "ultralytics is installed, but this version did not accept SAM 3 text "
                    f"prompt prediction arguments. Last error: {last_error}"
                )
            if not isinstance(raw_results, (list, tuple)):
                raw_results = [raw_results]
            for item in raw_results:
                results.extend(
                    _normalize_result_like(item, prompt, (h, w), conf_thres, f"{prompt}_ultra")
                )
        for idx, mask in enumerate(results):
            mask.mask_id = f"sam3_ultra_{idx:05d}"
        return results


def create_sam_backend(name: str, **kwargs: Any) -> SamBackend:
    if name == "native_sam3":
        return NativeSam3Backend(**kwargs)
    if name == "ultralytics_sam3":
        return UltralyticsSam3Backend(**kwargs)
    raise ValueError(f"Unknown SAM backend {name!r}.")
