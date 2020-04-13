# -*- coding: utf-8 -*-
import unittest
import numpy as np
import sys
import datetime
import pickle
sys.path.append('../')
import stage


class  TestStage(unittest.TestCase):
    """Test class for stage.py
    """

    def test_make_mouse_info(self):
        mouse_info = stage.read_mouse_info("../data")

        exp = np.array(["ID46770",
                        "ID45764",
                        "ID47313",
                        "ID37963",
                        "ID33572",
                        "ID47481",
                        "ID47479",
                        "ID45791"]
                       )

        res = mouse_info['Device label']

        np.testing.assert_array_equal(res, exp)
