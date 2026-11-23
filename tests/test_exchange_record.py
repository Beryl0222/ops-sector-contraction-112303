"""检查交换数据中用于回查的事实。"""
import unittest

import numpy as np

from sector_contraction_input import load_record


class ExchangeRecordTests(unittest.TestCase):
    def test_exchange_contract(self):
        record = load_record()
        self.assertTrue(record["nodes"][0]["own"] == record["nodes"][2]["own"] and record["nodes"][2]["axis_permutation"] == [1,0])
        self.assertTrue(record["blocks"]["B"]["111"] is None and record["blocks"]["D"]["0"] == "zero")
        self.assertTrue(np.shape(record["replacements"][0]["values"]) == tuple(record["replacements"][0]["shape"]))
        self.assertTrue(set(record["query"]["open_axes"]) == {"i","o"})
