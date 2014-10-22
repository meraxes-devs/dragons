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
        self.volume = (100./0.705)**3.  # Mpc^3

    def test_mass_function(self):
        mf, edges = dragons.munge.mass_function(self.masses, self.volume,
                                               bins='knuth',
                                               return_edges=True)  
        self.assertAlmostEqual(mf.max(), 6.85)
        self.assertEqual(mf.size, 6)
        self.assertEqual(edges.size, 4)

    def tearDown(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMunge)
    unittest.main()
