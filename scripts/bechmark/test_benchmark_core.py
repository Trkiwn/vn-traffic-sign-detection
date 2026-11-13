"""GPU-free checks of the benchmark harness: statistics, timing loop, table and driver."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark" / "benchmark_inference.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_inference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeImage:
    size = (960, 540)


class FakeAdapter:
    forward_includes_postprocess = False
    params = 1_000_000
    weights_bytes = 4_000_000

    def __init__(self):
        self.calls = []

    def preprocess(self, img):
        self.calls.append("pre")
        return "input"

    def forward(self, inp):
        self.calls.append("fwd")
        return "raw"

    def postprocess(self, raw, size):
        self.calls.append("post")
        assert size == (960, 540)
        return ()

    def gflops(self):
        return 12.5


class FakeRuntime:
    is_cuda = False
    device = "fake"

    def sync(self):
        pass

    def inference_context(self):
        import contextlib
        return contextlib.nullcontext()

    def reset_peak(self):
        pass

    def peak_mem_mb(self):
        return 321.0

    def release(self):
        pass

    def device_index(self):
        return 0


class TestBenchmarkCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_percentiles(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(self.mod.percentile(values, 50), 3)
        self.assertAlmostEqual(self.mod.percentile(values, 95), 4.8)
        stats = self.mod.summarise(values)
        self.assertEqual((stats["n"], stats["min"], stats["max"]), (5, 1, 5))
        self.assertAlmostEqual(stats["mean"], 3.0)

    def test_loop_counts_and_order(self):
        adapter = FakeAdapter()
        syncs = []
        out = self.mod.run_timed_loop(adapter, [FakeImage()], warmup=2, runs=5, sync=lambda: syncs.append(1))
        for stage in ("e2e", "pre", "fwd", "post"):
            self.assertEqual(len(out[stage]), 5)
            self.assertTrue(all(v >= 0 for v in out[stage]))
        # Each iteration: one end-to-end pass and one staged pass.
        self.assertEqual(adapter.calls, ["pre", "fwd", "post", "pre", "fwd", "post"] * 7)
        self.assertEqual(len(syncs), 7 * (2 + 4))

    def test_main_with_fake_adapters(self):
        mod = self.mod
        saved = (mod.make_runtime, mod.ADAPTER_FACTORY, mod.load_images, mod.collect_environment)
        try:
            mod.make_runtime = lambda device: FakeRuntime()
            mod.ADAPTER_FACTORY = lambda key, runtime, fp16, args: FakeAdapter()
            mod.load_images = lambda directory, count, seed: ([FakeImage(), FakeImage()], ["a.jpg", "b.jpg"])
            mod.collect_environment = lambda runtime: {"device": "fake"}
            with tempfile.TemporaryDirectory() as tmp:
                mod.main(["--models", "rtdetr", "faster_rcnn", "--precision", "fp32", "--allow-cpu",
                          "--warmup", "1", "--runs", "4", "--repeats", "2", "--output-dir", tmp])
                payload = json.loads((Path(tmp) / "benchmark_results.json").read_text(encoding="utf-8"))
                table = (Path(tmp) / "benchmark_table.md").read_text(encoding="utf-8")
        finally:
            mod.make_runtime, mod.ADAPTER_FACTORY, mod.load_images, mod.collect_environment = saved
        rows = payload["rows"]
        self.assertEqual([r["key"] for r in rows], ["rtdetr", "faster_rcnn"])
        self.assertEqual(rows[0]["e2e_ms"]["n"], 8)  # 4 runs x 2 repeats pooled
        self.assertEqual(rows[0]["gflops"], 12.5)
        self.assertEqual(rows[0]["peak_gpu_mem_mb"], 321.0)
        self.assertAlmostEqual(rows[0]["params_m"], 1.0)
        self.assertIn("RT-DETR R50vd", table)
        self.assertIn("Faster R-CNN", table)


if __name__ == "__main__":
    unittest.main()
