"""部署规格下的预算行为：分段收缩、路径无关性、受限报告与反向。"""
import unittest

import numpy as np

from sectornet import Engine, Query
from sectornet.testing import chain_record

from reference import dense_component_values, finite_difference
from test_engine import result_has_slice


def big_intermediate_record(extent, scratch_bytes):
    """三节点记录：中间张量 [x,z] 是 extent×extent 的临时量，输出只有一个分量。"""
    rng = np.random.default_rng(7)

    def block(shape):
        return rng.integers(-3, 4, size=shape).tolist()

    return {
        "axes": {"x": [extent, 1], "y": [extent, 1], "z": [extent, 1], "o": [1, 1]},
        "owner_axes": {"W1": ["x", "y"], "W2": ["y", "z"], "W3": ["x", "z", "o"]},
        "nodes": [
            {"id": "N1", "axes": ["x", "y"], "parity": 0, "own": "W1"},
            {"id": "N2", "axes": ["y", "z"], "parity": 0, "own": "W2"},
            {"id": "N3", "axes": ["x", "z", "o"], "parity": 0, "own": "W3"},
        ],
        "blocks": {
            "W1": {"00": block((extent, extent)), "11": block((1, 1))},
            "W2": {"00": block((extent, extent)), "11": block((1, 1))},
            "W3": {
                "000": block((extent, extent, 1)),
                "011": block((extent, 1, 1)),
                "101": block((1, extent, 1)),
                "110": block((1, 1, 1)),
            },
        },
        "replacements": [],
        "query": {"open_axes": ["o"], "sectors": ["0"], "coordinates": [[0]]},
        "scale": {
            "nodes": 3,
            "sectors": 2,
            "sector_extent": extent,
            "scratch_bytes": scratch_bytes,
        },
    }


class ChainDeploymentTests(unittest.TestCase):
    """部署规格：18 节点、扇区长度 32、24MiB 临时预算。"""

    @classmethod
    def setUpClass(cls):
        cls.record = chain_record(nodes=18, sectors=2, sector_extent=32,
                                  scratch_bytes=25165824)
        cls.query = Query.from_record(cls.record["query"])
        cls.result = Engine.from_record(cls.record).run(cls.query)

    def test_values_match_dense_reference(self):
        engine = Engine.from_record(self.record)
        ref = dense_component_values(engine, self.query)
        for comp in self.result.components:
            self.assertEqual(comp.status, "available")
            # 值超出 2^53，不同求和顺序只有浮点舍入级差异。
            expected = ref[(comp.sector_key, comp.coordinates)]
            self.assertLessEqual(
                abs(comp.value - expected), 1e-9 * max(1.0, abs(expected)),
                "分量 %s 与稠密参照不符" % comp.sector_key,
            )

    def test_temporary_peak_within_deployment_budget(self):
        self.assertLessEqual(self.result.peak_bytes, 25165824)

    def test_backward_returns_all_used_blocks(self):
        grads = Engine.from_record(self.record).backward(self.result)
        self.assertFalse(grads.restricted)
        # 每个基础块的两个合法扇区都参与了两个输出分量。
        self.assertEqual(len(grads.grads), 36)
        self.assertLessEqual(grads.peak_bytes, 25165824)

    def test_backward_matches_finite_differences(self):
        record = chain_record(nodes=6, sectors=2, sector_extent=4, seed=3)
        engine = Engine.from_record(record, scratch_bytes=1 << 30)
        query = Query.from_record(record["query"])
        grads = engine.backward(engine.run(query))
        for block, sector in (("W0", "00"), ("W1", "01"), ("W2", "11"), ("W5", "00")):
            numeric = finite_difference(engine, record, query, block, sector)
            grad = grads.for_block(block, sector)
            self.assertIsNotNone(grad, "%s/%s" % (block, sector))
            for index, value in numeric.items():
                self.assertAlmostEqual(grad[index], value, places=4)

    def test_tiny_budget_restricts_with_explanation(self):
        engine = Engine.from_record(self.record, scratch_bytes=8)
        result = engine.run(self.query)
        for comp in result.components:
            self.assertEqual(comp.status, "restricted")
            self.assertIn("预算", comp.reason)


class SegmentedContractionTests(unittest.TestCase):
    """输出很小、中间扇区很大：不能因整表构造不起而放弃可分段完成的输出。"""

    EXTENT = 1800  # 中间临时量 1800×1800×8B ≈ 25.9MB > 24MiB 的一半以上

    @classmethod
    def setUpClass(cls):
        cls.record = big_intermediate_record(cls.EXTENT, 25165824)
        cls.query = Query.from_record(cls.record["query"])
        cls.huge = Engine.from_record(cls.record, scratch_bytes=1 << 40)
        cls.tight = Engine.from_record(cls.record, scratch_bytes=25165824)
        cls.result_huge = cls.huge.run(cls.query)
        cls.result_tight = cls.tight.run(cls.query)

    def test_segmented_output_equals_unsegmented(self):
        value_huge = self.result_huge.components[0].value
        value_tight = self.result_tight.components[0].value
        self.assertEqual(self.result_tight.components[0].status, "available")
        self.assertEqual(value_huge, value_tight)

    def test_segmentation_actually_happened_within_budget(self):
        self.assertTrue(result_has_slice(self.result_tight))
        self.assertLessEqual(self.result_tight.peak_bytes, 25165824)

    def test_backward_within_budget_matches_unsegmented(self):
        grads_huge = self.huge.backward(self.result_huge)
        grads_tight = self.tight.backward(self.result_tight)
        self.assertFalse(grads_tight.restricted)
        self.assertLessEqual(grads_tight.peak_bytes, 25165824)
        for slot, grad in grads_huge.grads.items():
            np.testing.assert_array_equal(grad, grads_tight.for_block(*slot))

    def test_gradients_match_finite_differences(self):
        record = big_intermediate_record(6, 1 << 30)
        engine = Engine.from_record(record)
        query = Query.from_record(record["query"])
        grads = engine.backward(engine.run(query))
        for block, sector in (("W1", "00"), ("W2", "00"), ("W3", "000")):
            numeric = finite_difference(engine, record, query, block, sector)
            grad = grads.for_block(block, sector)
            for index, value in numeric.items():
                self.assertAlmostEqual(grad[index], value, places=4)


if __name__ == "__main__":
    unittest.main()
