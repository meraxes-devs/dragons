#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dragons
----------------------------------

Tests for `dragons` module.
"""

import unittest
import numpy as np

import dragons

class TestMunge(unittest.TestCase):

    def setUp(self):
        self.masses = np.array([1.1,2.2,3.3,3.4,3.5,4.6,5.7,6.8,7.9,8.0])
        self.side_length = 100.
        self.volume = (self.side_length/0.705)**3.  # Mpc^3
        self.dim = 32
        self.grid = np.ones([self.dim,]*3)

    def test_mass_function(self):
        mf, edges = dragons.munge.mass_function(self.masses, self.volume,
                                                bins='knuth',
                                                return_edges=True)
        self.assertAlmostEqual(mf.max(), 6.85)
        self.assertEqual(mf.size, 6)
        self.assertEqual(edges.size, 4)

    def test_smooth_grid(self):
        self.grid[:] = 1.0
        smoothed_grid = dragons.munge.smooth_grid(self.grid, self.side_length,
                                                  30., filt="tophat")
        self.assertTrue(np.isclose(smoothed_grid, 1.0).all())

    def tearDown(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMunge)
    unittest.main()
