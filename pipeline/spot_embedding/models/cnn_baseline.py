"""CNN holistic baseline (4.3) — frozen ImageNet ResNet18 features on the spot-pattern image.

The honest control for the per-spot models: does a plain deep-feature embedding of the whole
spot pattern already match the fancy aggregators? With only ~444 images and 87 multi-photo
individuals, training a CNN from scratch would overfit and make a meaningless floor, so this
uses a **frozen** ImageNet-pretrained ResNet18 as a fixed feature extractor — the standard
"off-the-shelf deep features" baseline.

Input is background-free by construction: all of a photo's per-spot masks OR-ed onto black,
resized to 224² and replicated to 3 channels, so the network sees only the spot constellation
(no body, no background). Inputs are cached to prepared/<name>/cnn_inputs.pkl; per-id
embeddings are memoised so the harness doesn't recompute across folds.

Requires torch + torchvision (imported lazily). Deterministic: frozen weights + fixed inputs.
"""
from __future__ import annotations

import pickle

import numpy as np

from .._common import contours_db_path, dataset_name, prepared_dir, resolve_dataset
from .base import EmbeddingMatcher

_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_spot_union_images(dataset: str, size: int = _SIZE) -> dict[str, np.ndarray]:
    """id -> uint8 (size, size): union of the photo's per-spot masks, resized. Cached to pkl."""
    import cv2
    import duckdb

    dataset_dir = resolve_dataset(dataset)
    name = dataset_name(dataset_dir)
    cache = prepared_dir(name) / "cnn_inputs.pkl"
    if cache.is_file():
        blob = pickle.loads(cache.read_bytes())
        if blob.get("size") == size:
            return blob["images"]

    con = duckdb.connect(database=str(contours_db_path(dataset_dir)), read_only=True)
    try:
        dims = {sid: (w, h) for sid, w, h in
                con.execute("SELECT salamander_id, width, height FROM images").fetchall()}
        rows = con.execute("SELECT salamander_id, mask_png FROM spots").fetchall()
    finally:
        con.close()

    masks_by_id: dict[str, list[bytes]] = {}
    for sid, png in rows:
        masks_by_id.setdefault(sid, []).append(png)

    images: dict[str, np.ndarray] = {}
    for sid, (w, h) in dims.items():
        union = np.zeros((h, w), np.uint8)
        for png in masks_by_id.get(sid, []):
            m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == union.shape:
                union = np.maximum(union, m)
        images[sid] = cv2.resize(union, (size, size), interpolation=cv2.INTER_AREA)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"size": size, "images": images}))
    return images


class CNNBaselineMatcher(EmbeddingMatcher):
    name = "cnn"

    def __init__(self, dataset: str, size: int = _SIZE, batch_size: int = 32):
        self.dataset = dataset
        self.size = size
        self.batch_size = batch_size
        self._images = _build_spot_union_images(dataset, size)
        self._emb_cache: dict[str, np.ndarray] = {}
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import torch
            import torchvision

            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            net = torchvision.models.resnet18(weights=weights)
            net.fc = torch.nn.Identity()   # keep the 512-d GAP feature
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._model = net
        return self._model

    def _to_tensor(self, ids: list[str]):
        import torch

        arrs = []
        for sid in ids:
            g = self._images.get(sid)
            if g is None:
                g = np.zeros((self.size, self.size), np.uint8)
            x = np.repeat((g.astype(np.float32) / 255.0)[..., None], 3, axis=2)  # HxWx3
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            arrs.append(x.transpose(2, 0, 1))                                    # 3xHxW
        return torch.from_numpy(np.stack(arrs)).float()

    def embed(self, spotsets: list) -> np.ndarray:
        import torch

        ids = [ss.salamander_id for ss in spotsets]
        todo = [sid for sid in ids if sid not in self._emb_cache]
        if todo:
            model = self._ensure_model()
            with torch.no_grad():
                for i in range(0, len(todo), self.batch_size):
                    chunk = todo[i : i + self.batch_size]
                    feats = model(self._to_tensor(chunk)).cpu().numpy()
                    for sid, f in zip(chunk, feats):
                        self._emb_cache[sid] = f.astype(np.float64)
        return np.vstack([self._emb_cache[sid] for sid in ids])
