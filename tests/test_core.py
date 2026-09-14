"""Small CPU tests covering scientific and train/test boundary invariants."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import pearsonr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from he2st.data import SlideDataset, panel_log1p, training_neighbor_targets
from he2st.engine import scores, validate_config
from he2st.metrics import per_gene_pearson
from he2st.models import ContextFusion
from he2st.preprocessing import crop_view
from he2st.cli import main as cli_main
from he2st.ensemble import grouped_average


class MetricsTests(unittest.TestCase):
    def test_ensemble_group_weights_in_expression_space(self):
        predictions={'old_a':np.array([[1.,3.]],np.float32),'old_b':np.array([[3.,1.]],np.float32),'new':np.array([[6.,10.]],np.float32)}
        groups=[{'weight':.75,'members':['old_a','old_b']},{'weight':.25,'members':['new']}]
        np.testing.assert_array_equal(grouped_average(predictions,groups),[[3.,4.]])

    def test_invalid_ensemble_weights_rejected(self):
        for weights in [[.7,.7],[-.2,1.2],[float('nan'),1.]]:
            with self.assertRaises(ValueError):
                grouped_average({'a':np.zeros((2,3))},[{'weight':w,'members':['a']} for w in weights])

    def test_pearson_axis_and_constant_columns(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(80, 4)); y = rng.normal(size=x.shape)
        y[:, 0] = x[:, 0]*3+12
        y[:, 1] = -x[:, 1]*5
        y[:, 3] = 7
        r = per_gene_pearson(x, y)
        np.testing.assert_allclose(r[:3], [pearsonr(x[:, i], y[:, i]).statistic for i in range(3)], atol=1e-12)
        self.assertTrue(np.isnan(r[3]))

    def test_normalization_and_invalid_input(self):
        counts = np.array([[1, 3], [0, 0]])
        np.testing.assert_allclose(panel_log1p(counts), np.log1p([[2500, 7500], [0, 0]]), atol=1e-6)
        with self.assertRaises(ValueError):
            panel_log1p(np.array([[-1, 2]]))

    def test_undefined_correlation_not_faked_as_zero(self):
        value = scores(np.ones((4, 50)), np.ones((4, 50)), np.ones(50, bool))
        self.assertIsNone(value['HEG200'])
        self.assertEqual(value['defined_genes'], 0)
        json.dumps(value, allow_nan=False)


class BoundaryTests(unittest.TestCase):
    def test_crop_coordinates_and_white_padding(self):
        image = np.zeros((300, 400, 3), np.uint8)
        image[145:155, 245:255, 0] = 123
        patch = crop_view(image, 250, 150, 224)
        self.assertEqual(int(patch[112,112,0]), 123)
        edge = crop_view(image, 0, 0, 224)
        np.testing.assert_array_equal(edge[:112], 255)
        self.assertEqual(crop_view(image, 200, 150, 896).shape, (224,224,3))
        with self.assertRaises(ValueError):
            crop_view(image, -1000, -1000, 224)

    def test_prediction_dataset_needs_no_rna(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'arrays').mkdir(); (root/'images').mkdir()
            genes = np.array([f'gene{i}' for i in range(200)])
            # Intentionally omit raw_counts/log_normalized entirely.
            np.savez(root/'arrays/test.npz', genes=genes, barcodes=['a','b'], coordinates_xy=[[0,0],[1,1]])
            np.save(root/'images/test_224_patches.npy', np.zeros((2,224,224,3), np.uint8))
            data = SlideDataset(root, ['test'], 'arrays', 'images', [224], genes, labels=False)
            self.assertIsNone(data.y)
            self.assertEqual(tuple(data[0].shape), (1,3,224,224))
            with self.assertRaises(KeyError):
                SlideDataset(root, ['test'], 'arrays', 'images', [224], genes, labels=True)

    def test_overlap_rejected(self):
        config = {'train':['A1','C1'], 'validation':['C1'], 'test':['D1']}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_spatial_smoothing_excludes_self_even_with_ties(self):
        class Dummy:
            y = np.array([[1.,0.],[3.,2.],[5.,4.]], dtype=np.float32)
            targets = [y]
            xy = [np.array([[0,0],[0,0],[1,0]])]
        expected = (Dummy.y.sum(0)-Dummy.y)/2
        np.testing.assert_allclose(training_neighbor_targets(Dummy(), 1), expected)


class ModelTests(unittest.TestCase):
    def test_backbones_and_view_contract(self):
        torch.set_num_threads(2)
        for name in ['resnet18','efficientnet_b0']:
            model = ContextFusion(backbone=name, scales=3, genes=200, pretrained=False).eval()
            with torch.inference_mode():
                predicted = model(torch.zeros(1,3,3,64,64))
            self.assertEqual(tuple(predicted.shape), (1,200))
            self.assertTrue(torch.isfinite(predicted).all())
            with self.assertRaises(ValueError):
                model(torch.zeros(1,1,3,64,64))


class WorkflowTests(unittest.TestCase):
    def test_fit_predict_evaluate_without_test_rna_and_reject_changed_mask(self):
        from he2st.engine import fit, predict
        import pandas as pd
        import contextlib
        import io
        class TinyModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.encoder = torch.nn.Linear(3, 4)
                self.head = torch.nn.Linear(4, 200)
            def forward(self, images):
                return self.head(self.encoder(images.mean((1,3,4))))
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp, patch('he2st.engine.ContextFusion', TinyModel), contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            for folder in ['arrays','images','benchmarks/liver']:
                (root/folder).mkdir(parents=True)
            genes=np.array([f'g{i}' for i in range(200)]); mask=np.arange(200)<50
            pd.DataFrame({'gene':genes,'is_primary_HEG50':mask}).to_csv(root/'benchmarks/liver/genes.csv',index=False)
            rng=np.random.default_rng(10)
            truth={}
            for slide in ['train','val','test']:
                truth=dict(genes=genes,barcodes=np.array(['a','b','c','d']),coordinates_xy=np.arange(8).reshape(4,2),raw_counts=rng.integers(1,80,(4,200)))
                metadata={key:value for key,value in truth.items() if slide!='test' or key!='raw_counts'}
                np.savez(root/f'arrays/{slide}.npz',**metadata)
                images=np.zeros((4,224,224,3),np.uint8)
                for i in range(4): images[i]=[30+i*30,60+i*10,90+i*15]
                np.save(root/f'images/{slide}_224_patches.npy',images)
            config=json.loads((Path(__file__).resolve().parents[1]/'configs/resnet18_ema.json').read_text())
            config.update(train=['train'],validation=['val'],test=['test'],arrays='arrays',images='images',scales=[224],epochs=1,workers=0,batch_size=4,pretrained=False)
            out=root/'run'
            fit(root,config,out,'cpu')
            # This checkpoint reload succeeds with a test NPZ that has no RNA.
            predict(root,out/'best.pt',['test'],out/'pred.npz','cpu')
            np.savez(root/'test_truth.npz',**truth)
            args=['--root',str(root),'evaluate','--prediction',str(out/'pred.npz'),'--truth',str(root/'test_truth.npz'),'--output',str(out/'metrics.json')]
            cli_main(args)
            self.assertEqual(json.loads((out/'metrics.json').read_text())['defined_genes'],200)
            with np.load(out/'pred.npz') as d: changed=dict(d)
            changed['heg50_mask']=np.roll(mask,1) # still 50 genes, but wrong fixed panel subset
            np.savez(out/'pred.npz',**changed)
            with self.assertRaises(AssertionError): cli_main(args)


if __name__ == '__main__':
    unittest.main()
