import json
import math
import unittest
from pathlib import Path
from collections import Counter

BASE=Path(__file__).resolve().parents[1]
MAIN=BASE/'results'/'YOLO11s'/'min_val_loss'
OLD=BASE/'results'/'YOLO11s'/'max_val_ap_pilot'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

class TestYOLO11sProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loss=read(MAIN/'val_loss_history.json')
        cls.select=read(MAIN/'selected_min_val_loss.json')
        cls.manifest=read(MAIN/'training_complete.json')
        cls.test=read(MAIN/'test_metrics.json')
        cls.preds=read(MAIN/'test_predictions.json')
        cls.proto=read(MAIN/'test_protocol.json')
    def test_50_epochs(self):
        self.assertEqual([x['epoch'] for x in self.loss], list(range(1,51)))
        self.assertEqual(set(self.manifest),{str(x) for x in range(1,51)})
    def test_validation_inputs(self):
        self.assertTrue(all(x['validation_images']==255 and x['batches']==32 for x in self.loss))
        self.assertTrue(all(math.isfinite(x['total_val_loss']) for x in self.loss))
        for x in self.loss:
            self.assertAlmostEqual(x['total_val_loss'],x['box_loss']+x['cls_loss']+x['dfl_loss'],places=9)
    def test_checkpoint_integrity(self):
        self.assertTrue(all(x['checkpoint_sha256']==self.manifest[str(x['epoch'])] for x in self.loss))
        self.assertEqual(self.manifest['37'],self.select['checkpoint_sha256'])
        self.assertEqual(self.select['checkpoint_sha256'],self.test['checkpoint_sha256'])
        self.assertEqual(self.proto['checkpoint_sha256'],self.select['checkpoint_sha256'])
    def test_correct_minimum(self):
        selected=min(self.loss,key=lambda x:(x['total_val_loss'],x['epoch']))
        self.assertEqual(selected['epoch'],37)
        self.assertEqual(self.select['epoch'],selected['epoch'])
        self.assertAlmostEqual(selected['total_val_loss'],self.select['total_val_loss'])
    def test_test_metrics(self):
        self.assertEqual(self.test['selected_epoch'],37)
        self.assertEqual(self.proto['selection_epoch'],37)
        self.assertEqual(self.test['dataset_sha256'],self.select['dataset_sha256'])
        self.assertEqual(self.test['dataset_sha256'],self.proto['dataset_sha256'])
        self.assertEqual(self.test['evaluated_images'],639)
        self.assertEqual(self.test['predictions'],2461)
        self.assertEqual(len(self.preds),2461)
        self.assertEqual(len(self.test['per_class']),52)
        self.assertAlmostEqual(self.test['map'],0.7031666801599523)
    def test_predictions(self):
        n=Counter()
        for x in self.preds:
            self.assertEqual(set(x),{'image_id','category_id','bbox','score'})
            self.assertIsInstance(x['image_id'],int)
            self.assertTrue(1<=x['category_id']<=52)
            self.assertTrue(0.001<=x['score']<=1)
            self.assertTrue(all(math.isfinite(y) for y in x['bbox']))
            self.assertTrue(x['bbox'][2]>0 and x['bbox'][3]>0)
            n[x['image_id']]+=1
        self.assertLessEqual(max(n.values()),100)
        self.assertEqual(len(n),638)  # One test image had no predictions above score floor.
    def test_preserves_pilot(self):
        prior=read(OLD/'test_metrics.json')
        self.assertEqual(prior['selected_epoch'],36)
        self.assertAlmostEqual(prior['map'],0.7033492208554228)
        self.assertNotEqual(prior['checkpoint_sha256'],self.test['checkpoint_sha256'])

if __name__=='__main__': unittest.main()
