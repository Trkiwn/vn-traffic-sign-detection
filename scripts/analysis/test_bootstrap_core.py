"""Checks that the bootstrap's fast COCO computation matches pycocotools.

Needs numpy and pycocotools (both are project dependencies); skipped otherwise.
"""
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "analysis" / "clean_test_and_bootstrap.py").is_file())

try:
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False


def load_module():
    spec = importlib.util.spec_from_file_location(
        "clean_test_and_bootstrap", ROOT / "scripts" / "analysis" / "clean_test_and_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_data(n_images=40, n_categories=6, seed=0):
    rng = np.random.default_rng(seed)
    images = [{"id": i + 1, "file_name": f"{100 + i}.jpg", "width": 640, "height": 480} for i in range(n_images)]
    annotations, detections = [], []
    for image in images:
        for _ in range(int(rng.integers(0, 4))):
            w, h = rng.uniform(15, 100, 2)
            x, y = rng.uniform(0, 540), rng.uniform(0, 380)
            category = int(rng.integers(1, n_categories + 1))
            annotations.append({"id": len(annotations) + 1, "image_id": image["id"], "category_id": category,
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            if rng.random() < 0.85:
                jitter = rng.normal(0, 3, 4)
                detections.append({"image_id": image["id"], "category_id": category if rng.random() < 0.9 else 1,
                                   "bbox": [x + jitter[0], y + jitter[1], max(1, w + jitter[2]), max(1, h + jitter[3])],
                                   "score": float(rng.uniform(0.2, 1))})
        for _ in range(3):
            detections.append({"image_id": image["id"], "category_id": int(rng.integers(1, n_categories + 1)),
                               "bbox": [float(rng.uniform(0, 500)), float(rng.uniform(0, 350)),
                                        float(rng.uniform(10, 80)), float(rng.uniform(10, 80))],
                               "score": float(rng.uniform(0.001, 0.3))})
    categories = [{"id": c, "name": str(c)} for c in range(1, n_categories + 1)]
    return {"images": images, "annotations": annotations, "categories": categories}, detections


@unittest.skipUnless(AVAILABLE, "numpy and pycocotools are required")
class TestFastCOCO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()
        data, cls.detections = synthetic_data()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.gt = COCO()
            cls.gt.dataset = data
            cls.gt.createIndex()
            evaluator = COCOeval(cls.gt, cls.gt.loadRes(cls.detections), "bbox")
            evaluator.evaluate()
        cls.fast = cls.mod.FastCOCO(evaluator)
        cls.image_ids = sorted(cls.gt.getImgIds())

    def reference(self, positions):
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = COCOeval(self.gt, self.gt.loadRes(self.detections), "bbox")
            evaluator.params.imgIds = [self.image_ids[i] for i in positions]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        s = evaluator.stats
        return dict(zip(self.mod.METRICS, (s[0], s[1], s[2], s[8])))

    def test_matches_pycocotools_on_full_set_and_subsets(self):
        rng = np.random.default_rng(1)
        subsets = [list(range(len(self.image_ids)))] + [
            sorted(rng.choice(len(self.image_ids), size=int(rng.integers(10, 40)), replace=False)) for _ in range(4)]
        for positions in subsets:
            counts = np.zeros(len(self.image_ids), dtype=np.int64)
            counts[positions] = 1
            fast, reference = self.fast.metrics(counts), self.reference(positions)
            for metric in self.mod.METRICS:
                self.assertAlmostEqual(fast[metric], reference[metric], places=9, msg=metric)

    def test_repeated_images_are_supported(self):
        counts = np.full(len(self.image_ids), 2, dtype=np.int64)
        doubled = self.fast.metrics(counts)
        single = self.fast.metrics(np.ones(len(self.image_ids), dtype=np.int64))
        self.assertTrue(all(0 <= doubled[m] <= 1 for m in self.mod.METRICS))
        self.assertAlmostEqual(doubled["mar_100"], single["mar_100"], places=6)

    def test_blocks_and_interval(self):
        blocks = self.mod.make_blocks(list(range(25)), 10)
        self.assertEqual([len(b) for b in blocks], [10, 10, 5])
        values = np.linspace(0.40, 0.50, 101)
        lo, hi = self.mod.interval(values, point=0.45)
        self.assertAlmostEqual((lo + hi) / 2, 0.45, places=6)  # re-centred on the point estimate
        self.assertLess(lo, 0.45)
        self.assertGreater(hi, 0.45)


if __name__ == "__main__":
    unittest.main()
