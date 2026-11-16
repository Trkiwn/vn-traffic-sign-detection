from pathlib import Path
import unittest
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
YOLO=ROOT / 'scripts' / 'YOLO11s'

class TestYOLO11sLayout(unittest.TestCase):
    def test_three_model_like_directories(self):
        for name in ('models','evaluate','visualisation'):
            self.assertTrue((YOLO/name).is_dir(), name)
    def test_all_documented_entrypoints_exist(self):
        for path in (
            YOLO/'models'/'VGU_YOLO11s_MinValLoss_GitHub_Ready.ipynb',
            YOLO/'evaluate'/'replay_coco.py',
            YOLO/'visualisation'/'plot_val_loss.py',
            YOLO/'visualisation'/'visualise_yolo11s.py',
        ): self.assertTrue(path.is_file(), str(path))
    def test_evaluation_root_correct_after_move(self):
        p=YOLO/'evaluate'/'replay_coco.py'
        spec=importlib.util.spec_from_file_location('yolo11s_replay_coco', p)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.ROOT, ROOT)
        self.assertTrue(mod.ARTIFACTS.joinpath('test_metrics.json').is_file())
    def test_readme_matches_actual_paths(self):
        s=(ROOT/'README.md').read_text(encoding='utf8')
        for file in (
            'scripts/YOLO11s/models/VGU_YOLO11s_MinValLoss_GitHub_Ready.ipynb',
            'scripts/YOLO11s/evaluate/replay_coco.py',
            'scripts/YOLO11s/visualisation/plot_val_loss.py',
            'scripts/YOLO11s/visualisation/visualise_yolo11s.py',
        ): self.assertIn(file,s)
        self.assertNotIn('scripts/YOLO11s/replay_coco.py',s)
        self.assertNotIn('scripts/YOLO11s/plot_val_loss.py',s)
