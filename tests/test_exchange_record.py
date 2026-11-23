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
        self.assertTrue(all(np.shape(data) == tuple(record["axes"][axis][int(bit)] for axis, bit in zip(record["owner_axes"]["B"], sector)) for sector, data in record["blocks"]["B"].items() if data is not None))
        self.assertTrue(all(tuple(len(ix) for ix in indices) == tuple(record["axes"][axis][int(bit)] for axis, bit in zip(record["nodes"][2]["axes"], sector)) and all(0 <= i < np.shape(record["blocks"]["S"][sector[::-1]])[axis] for position, axis in enumerate(record["nodes"][2]["axis_permutation"]) for i in indices[position]) for sector, indices in record["nodes"][2]["indices"].items()))
