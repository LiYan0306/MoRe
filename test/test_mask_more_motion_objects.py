#!/usr/bin/env python3
"""Synthetic regression tests for mask_more_motion_objects.py. Run with unittest."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mask_more_motion_objects import (
    geometry_for_image, list_pairs, mask_in_original_coords, process,
)


class MaskMoReMotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / 'inputs'
        self.masks = self.root / 'masks'
        self.output = self.root / 'results'
        self.images.mkdir()
        self.masks.mkdir()

    def image(self, name, rgb, shape=None):
        arr = np.array(rgb, dtype=np.uint8)
        if shape is not None:
            h, w = shape
            arr = np.broadcast_to(arr, (h, w, 3)).copy()
        Image.fromarray(arr, 'RGB').save(self.images / name)
        return self.images / name

    def mask(self, idx, arr, name=None):
        Image.fromarray(np.array(arr, dtype=np.uint8), 'L').save(
            self.masks / (name or f'mask_{idx:05d}.png'))

    def output_image(self, pattern='*.png'):
        files = list((self.output / 'images').glob(pattern))
        self.assertEqual(len(files), 1)
        return np.array(Image.open(files[0]))

    def output_binary(self, pattern='*.png'):
        files = list((self.output / 'dynamic_masks').glob(pattern))
        self.assertEqual(len(files), 1)
        return np.array(Image.open(files[0]))

    def test_threshold_strict_and_original_pixel_exact(self):
        # Original image: all nonblack, so masked pixels are easy to identify.
        rgb = np.array([[[10,20,30],[40,50,60],[70,80,90],[100,110,120]]], dtype=np.uint8)
        self.image('00000.png', rgb)
        self.mask(0, [[0,127,128,255]])
        n, removed = process(self.images, self.masks, self.output, .5, 'exact')
        self.assertEqual((n,removed), (1,2))
        got = self.output_image()
        np.testing.assert_array_equal(got[0,:2],rgb[0,:2])
        np.testing.assert_array_equal(got[0,2:],np.zeros((2,3),dtype=np.uint8))
        np.testing.assert_array_equal(self.output_binary(),np.array([[0,0,255,255]],dtype=np.uint8))
        with (self.output / 'pairing.csv').open(encoding='utf-8-sig') as fd:
            rows=list(csv.DictReader(fd))
        self.assertEqual(rows[0]['source_image'],'00000.png')
        self.assertEqual(rows[0]['motion_mask'],'mask_00000.png')
        self.assertEqual(rows[0]['removed_pixels'],'2')

    def test_threshold_endpoints(self):
        rgb = np.ones((1,3,3),dtype=np.uint8)*99
        self.image('a.png',rgb)
        self.mask(0,[[0,1,255]])
        process(self.images,self.masks,self.output,0.0,'exact')
        np.testing.assert_array_equal(self.output_binary(),np.array([[0,255,255]],dtype=np.uint8))
        process(self.images,self.masks,self.output,1.0,'exact',overwrite=True)
        np.testing.assert_array_equal(self.output_image(),rgb)
        np.testing.assert_array_equal(self.output_binary(),np.zeros((1,3),dtype=np.uint8))

    def test_lexical_pairing_and_names(self):
        for name in ['17585898398570002.jpg','17585898398570001.jpg']:
            self.image(name,[120,80,60],shape=(5,6))
        self.mask(1,np.zeros((5,6),dtype=np.uint8))
        self.mask(0,np.ones((5,6),dtype=np.uint8)*255)
        pairs = list_pairs(self.images,self.masks)
        self.assertEqual([x.name for x,_ in pairs],['17585898398570001.jpg','17585898398570002.jpg'])
        n, removed = process(self.images,self.masks,self.output,.5,'exact')
        self.assertEqual((n,removed),(2,30))
        self.assertTrue((self.output/'images'/'00000_17585898398570001_motion_removed.png').is_file())
        self.assertTrue((self.output/'images'/'00001_17585898398570002_motion_removed.png').is_file())
        first=np.asarray(Image.open(self.output/'images'/'00000_17585898398570001_motion_removed.png'))
        second=np.asarray(Image.open(self.output/'images'/'00001_17585898398570002_motion_removed.png'))
        self.assertTrue((first==0).all())
        decoded_original = np.asarray(Image.open(self.images/'17585898398570002.jpg').convert('RGB'))
        np.testing.assert_array_equal(second, decoded_original)

    def test_crop_portrait_unobserved_unchanged(self):
        original=np.ones((420,140,3),dtype=np.uint8)*100
        self.image('portrait.png', original)
        self.mask(0,np.ones((518,518),dtype=np.uint8)*255)
        process(self.images,self.masks,self.output,.5,'more-crop')
        result=self.output_image()
        binary=self.output_binary()
        # Cropped top/bottom were never observed and must never be deleted.
        self.assertEqual(result[0,70].tolist(),[100,100,100])
        self.assertEqual(result[419,70].tolist(),[100,100,100])
        self.assertEqual(result[210,70].tolist(),[0,0,0])
        self.assertEqual(binary[0,70],0)
        self.assertEqual(binary[210,70],255)

    def test_batch_padding_removed_before_resizing(self):
        # landscape -> processed height 294, portrait -> 518, batch H = 518
        self.image('a.png',[31,41,51],shape=(720,1280))
        self.image('b.png',[20,30,40],shape=(720,400))
        m0=np.zeros((518,518),dtype=np.uint8)
        m0[:100,:]=255  # batch padding; not part of image
        m0[200:300,100:200]=255  # valid image content
        self.mask(0,m0)
        self.mask(1,np.zeros((518,518),dtype=np.uint8))
        process(self.images,self.masks,self.output,.5,'more-crop')
        binary=np.array(Image.open(self.output/'dynamic_masks'/'00000_a_dynamic_mask.png'))
        self.assertEqual(binary[0,0],0)
        self.assertTrue((binary>0).any())
        self.assertTrue((binary==0).any())

    def test_refuse_missing_indices_and_duplicates(self):
        self.image('a.png',[1,2,3],shape=(5,5))
        self.mask(1,np.zeros((5,5),dtype=np.uint8))
        with self.assertRaisesRegex(ValueError,'缺失编号'):
            process(self.images,self.masks,self.output,.5,'exact')
        self.mask(0,np.zeros((5,5),dtype=np.uint8))
        self.mask(0,np.zeros((5,5),dtype=np.uint8),name='mask_0.png')
        with self.assertRaisesRegex(ValueError,'重复'):
            process(self.images,self.masks,self.output,.5,'exact')
        self.assertFalse(self.output.exists())

    def test_refuse_overwrite_and_nested_destination(self):
        self.image('a.png',[1,2,3],shape=(5,5))
        self.mask(0,np.zeros((5,5),dtype=np.uint8))
        process(self.images,self.masks,self.output,.5,'exact')
        with self.assertRaisesRegex(ValueError,'已存在'):
            process(self.images,self.masks,self.output,.5,'exact')
        with self.assertRaisesRegex(ValueError,'输出目录不得'):
            process(self.images,self.masks,self.images/'outputs',.5,'exact')
        with self.assertRaisesRegex(ValueError,'输出目录不得'):
            process(self.images,self.masks,self.root,.5,'exact')

    def test_reject_bad_types_ranges_and_mask_shapes(self):
        self.image('a.png',[11,12,13],shape=(5,5))
        self.mask(0,np.zeros((5,6),dtype=np.uint8))
        for t in [-0.1, 1.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                process(self.images,self.masks,self.output,t,'exact')
        with self.assertRaisesRegex(ValueError,'尺寸'):
            process(self.images,self.masks,self.output,.5,'exact')
        with self.assertRaisesRegex(ValueError,'预期'):
            process(self.images,self.masks,self.output,.5,'more-crop')
        self.assertFalse(self.output.exists())

    def test_no_save_binary_and_no_dynamic(self):
        original = np.array([[[11,12,13],[14,15,16]]],dtype=np.uint8)
        self.image('a.png',original)
        self.mask(0,[[0,255]])
        process(self.images,self.masks,self.output,1.,'exact',save_binary=False)
        self.assertFalse((self.output/'dynamic_masks').exists())
        np.testing.assert_array_equal(self.output_image(),original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
