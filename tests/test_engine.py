"""样本记录与特制记录上的端到端行为：数值、视图梯度、三态块语义、证明。"""
import unittest

import numpy as np

from sector_contraction_input import load_record
from sectornet import Engine, Query, RecordError

from reference import dense_component_values, finite_difference


def make_engine(record, budget=None):
    if budget is None:
        return Engine.from_record(record)
    return Engine.from_record(record, scratch_bytes=budget)


def plan_has_slice(plan):
    if isinstance(plan, dict):
        return plan.get("op") == "slice" or any(plan_has_slice(v) for v in plan.values())
    if isinstance(plan, list):
        return any(plan_has_slice(v) for v in plan)
    return False


def result_has_slice(result):
    for group in result.proof()["groups"]:
        for assignment in group["assignments"]:
            if "plan" in assignment and plan_has_slice(assignment["plan"]):
                return True
    return False


class SampleForwardTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.engine = Engine.from_record(self.record)
        self.query = Query.from_record(self.record["query"])
        self.result = self.engine.run(self.query)

    def test_component_value_matches_dense_reference(self):
        ref = dense_component_values(self.engine, self.query)
        comp = self.result.components[0]
        self.assertEqual(comp.status, "available")
        self.assertEqual(comp.value, -10.0)
        self.assertAlmostEqual(comp.value, ref[("00", (0, 0))])

    def test_missing_block_marks_undetermined_not_zero(self):
        comp = self.result.components[1]
        self.assertEqual(comp.status, "undetermined")
        self.assertIn(("B", "111"), comp.missing)
        self.assertIsNone(comp.value)
        self.assertIn("B/111", comp.reason)

    def test_loss_uses_only_available_sectors(self):
        self.assertEqual(self.result.loss, -10.0)
        self.assertEqual(self.result.available(), [(0, -10.0)])
        self.assertEqual(len(self.result.unavailable()), 1)

    def test_value_independent_of_path_and_chunking(self):
        for budget in (10**9, 256, 64, 48, 40, 32):
            engine = make_engine(self.record, budget)
            result = engine.run(self.query)
            self.assertEqual(result.components[0].value, -10.0, "预算 %d" % budget)
            self.assertEqual(result.components[1].status, "undetermined")
        tight = make_engine(self.record, 48).run(self.query)
        self.assertTrue(result_has_slice(tight), "小预算下应发生分段")

    def test_restricted_budget_explains_request(self):
        engine = make_engine(self.record, 8)
        result = engine.run(self.query)
        comp = result.components[0]
        self.assertEqual(comp.status, "restricted")
        self.assertIn("预算", comp.reason)
        self.assertIn("字节", comp.reason)


class SampleBackwardTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.engine = Engine.from_record(self.record)
        self.query = Query.from_record(self.record["query"])
        self.result = self.engine.run(self.query)
        self.grads = self.engine.backward(self.result)

    def test_hand_computed_gradients(self):
        np.testing.assert_array_equal(
            self.grads.for_block("S", "00"), [[-2.0, -9.0], [0.0, 0.0]]
        )
        np.testing.assert_array_equal(
            self.grads.for_block("B", "010"),
            [[[4.0, 4.0], [-2.0, -2.0]], [[8.0, 8.0], [-4.0, -4.0]]],
        )
        np.testing.assert_array_equal(self.grads.for_block("D", "1"), [6.0, 22.0])

    def test_gradients_match_finite_differences(self):
        for block, sector in (("S", "00"), ("B", "010"), ("D", "1")):
            numeric = finite_difference(self.engine, self.record, self.query, block, sector)
            grad = self.grads.for_block(block, sector)
            for index, value in numeric.items():
                self.assertAlmostEqual(grad[index], value, places=5)

    def test_repeated_view_position_accumulates_to_base(self):
        # 节点 C 在 j=1 处重复取位：S/00[0,1] 收到 A 路 -4 与 C 路两次共 -5。
        self.assertEqual(self.grads.for_block("S", "00")[0, 1], -9.0)

    def test_structural_zero_sector_has_no_trainable_gradient(self):
        self.assertIsNone(self.grads.for_block("D", "0"))
        status, reason = self.grads.status("D", "0")
        self.assertEqual(status, "structural-zero")
        self.assertIn("固定约束", reason)

    def test_missing_sector_has_no_gradient(self):
        self.assertIsNone(self.grads.for_block("B", "111"))
        self.assertEqual(self.grads.status("B", "111")[0], "missing")

    def test_sector_serving_only_undetermined_outputs(self):
        status, reason = self.grads.status("S", "11")
        self.assertEqual(status, "none")
        self.assertIn("无法确定", reason)

    def test_eligibility_report(self):
        report = self.grads.eligibility()
        outputs = {o["index"]: o for o in report["outputs"]}
        self.assertTrue(outputs[0]["trainable"])
        self.assertFalse(outputs[1]["trainable"])
        params = {(p["block"], p["sector"]): p for p in report["parameters"]}
        self.assertTrue(params[("S", "00")]["received_gradient"])
        self.assertFalse(params[("S", "11")]["received_gradient"])
        self.assertEqual(params[("B", "001")]["status"], "none")
        self.assertIn("未参与", params[("B", "001")]["reason"])

    def test_eligibility_follows_requested_axes(self):
        # 改变请求的轴范围后，同一参数失去资格的原因随之变化。
        query = Query(("i", "o"), ("00",), ((0, 0),))
        grads = self.engine.backward(self.engine.run(query))
        self.assertIn("未参与", grads.status("S", "11")[1])
        self.assertIn("无法确定", self.grads.status("S", "11")[1])

    def test_loss_weights_scale_upstream(self):
        query = Query(("i", "o"), ("00", "11"), ((0, 0), (0, 0)), weights=(2.0, 3.0))
        result = self.engine.run(query)
        self.assertEqual(result.loss, -20.0)
        grads = self.engine.backward(result)
        np.testing.assert_array_equal(grads.for_block("D", "1"), [12.0, 44.0])


class ReplacementAndProofTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.engine = Engine.from_record(self.record)
        self.query = Query.from_record(self.record["query"])
        self.result = self.engine.run(self.query)
        replacement = self.record["replacements"][0]
        self.engine2 = self.engine.with_replacement(
            replacement["source"], replacement["sector"], replacement["values"]
        )

    def test_replacement_makes_missing_sector_available(self):
        result2 = self.engine2.run(self.query)
        self.assertEqual(result2.components[0].value, -10.0)
        self.assertEqual(result2.components[1].value, -12.0)
        self.assertEqual(result2.loss, -22.0)

    def test_replanning_uses_new_block_identity(self):
        self.assertNotEqual(
            self.engine.store.identity("B", "111"),
            self.engine2.store.identity("B", "111"),
        )
        proof_old = self.result.proof()
        proof_new = self.engine2.run(self.query).proof()
        self.assertNotEqual(
            proof_old["identities"]["B/111"], proof_new["identities"]["B/111"]
        )
        self.assertEqual(
            proof_old["structure_hash"], proof_new["structure_hash"],
            "替换参数源不改变节点图结构",
        )

    def test_proof_reproduced_with_old_inputs(self):
        report = self.engine.verify(self.result.proof())
        self.assertTrue(report.structure_ok)
        self.assertTrue(report.ok, report.summary())

    def test_proof_reports_changed_inputs(self):
        report = self.engine2.verify(self.result.proof())
        changed = {(b["block"], b["sector"]) for b in report.blocks if not b["match"]}
        self.assertIn(("B", "111"), changed)
        comp0 = [c for c in report.components if c["index"] == 0][0]
        self.assertTrue(comp0["match"], "未受替换影响的分量仍应复得")

    def test_proof_roundtrip_through_json(self):
        import json

        proof = json.loads(json.dumps(self.result.proof()))
        self.assertTrue(self.engine.verify(proof).ok)


class QueryValidationTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.from_record(load_record())

    def test_coordinate_must_fit_sector(self):
        bad = Query(("i", "o"), ("11",), ((5, 0),))
        with self.assertRaises(RecordError):
            self.engine.run(bad)

    def test_open_axes_must_match_dangling(self):
        bad = Query(("i",), ("0",), ((0,),))
        with self.assertRaises(RecordError):
            self.engine.run(bad)

    def test_sector_key_must_match_open_axes(self):
        bad = Query(("i", "o"), ("0",), ((0, 0),))
        with self.assertRaises(RecordError):
            self.engine.run(bad)


# 结构零阻断对缺块的依赖：R/00 是声明的结构零，Q/00 缺失。
RECORD_Z = {
    "axes": {"p": [1, 1], "m": [1, 1], "n": [1, 1], "o": [1, 1]},
    "owner_axes": {"P": ["p", "m"], "Q": ["m", "n"], "R": ["n", "o"]},
    "nodes": [
        {"id": "P", "axes": ["p", "m"], "parity": 0, "own": "P"},
        {"id": "Q", "axes": ["m", "n"], "parity": 0, "own": "Q"},
        {"id": "R", "axes": ["n", "o"], "parity": 0, "own": "R"},
    ],
    "blocks": {
        "P": {"00": [[7]], "11": [[2]]},
        "Q": {"00": None, "11": [[5]]},
        "R": {"00": "zero", "11": [[3]]},
    },
    "replacements": [],
    "query": {
        "open_axes": ["p", "o"],
        "sectors": ["00", "11"],
        "coordinates": [[0, 0], [0, 0]],
    },
    "scale": {"nodes": 3, "sectors": 2, "sector_extent": 1, "scratch_bytes": 1 << 20},
}

# 普通零因子使缺块不参与取值，但其梯度仍取决于缺块。
RECORD_Y = {
    "axes": {"p": [1, 1], "m": [1, 1], "o": [1, 1]},
    "owner_axes": {"P": ["p", "m"], "Q": ["m", "o"]},
    "nodes": [
        {"id": "P", "axes": ["p", "m"], "parity": 0, "own": "P"},
        {"id": "Q", "axes": ["m", "o"], "parity": 0, "own": "Q"},
    ],
    "blocks": {
        "P": {"00": [[0]], "11": [[5]]},
        "Q": {"00": None, "11": [[2]]},
    },
    "replacements": [],
    "query": {
        "open_axes": ["p", "o"],
        "sectors": ["00", "11"],
        "coordinates": [[0, 0], [0, 0]],
    },
    "scale": {"nodes": 2, "sectors": 2, "sector_extent": 1, "scratch_bytes": 1 << 20},
}

# 普通参数恰好为零仍可接收梯度。
RECORD_W = {
    "axes": {"p": [1, 1], "m": [1, 1], "o": [1, 1]},
    "owner_axes": {"P": ["p", "m"], "Q": ["m", "o"]},
    "nodes": [
        {"id": "P", "axes": ["p", "m"], "parity": 0, "own": "P"},
        {"id": "Q", "axes": ["m", "o"], "parity": 0, "own": "Q"},
    ],
    "blocks": {
        "P": {"00": [[0]], "11": [[1]]},
        "Q": {"00": [[4]], "11": [[1]]},
    },
    "replacements": [],
    "query": {"open_axes": ["p", "o"], "sectors": ["00"], "coordinates": [[0, 0]]},
    "scale": {"nodes": 2, "sectors": 2, "sector_extent": 1, "scratch_bytes": 1 << 20},
}


class StructuralZeroBlocksMissingTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.from_record(RECORD_Z)
        self.query = Query.from_record(RECORD_Z["query"])
        self.result = self.engine.run(self.query)
        self.grads = self.engine.backward(self.result)

    def test_structural_zero_blocks_dependence_on_missing(self):
        comp = self.result.components[0]
        self.assertEqual(comp.status, "available")
        self.assertEqual(comp.value, 0.0)

    def test_other_sector_computes_normally(self):
        self.assertEqual(self.result.components[1].value, 30.0)
        self.assertEqual(self.result.loss, 30.0)

    def test_pruned_parameter_reports_structural_zero(self):
        status, reason = self.grads.status("P", "00")
        self.assertEqual(status, "zero")
        self.assertIn("结构零", reason)
        # 梯度恒为零，作为确定零梯度交付。
        np.testing.assert_array_equal(self.grads.for_block("P", "00"), [[0.0]])

    def test_gradients_flow_on_used_sector(self):
        np.testing.assert_array_equal(self.grads.for_block("P", "11"), [[15.0]])
        np.testing.assert_array_equal(self.grads.for_block("Q", "11"), [[6.0]])
        np.testing.assert_array_equal(self.grads.for_block("R", "11"), [[10.0]])


class OrdinaryZeroVsMissingTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.from_record(RECORD_Y)
        self.query = Query.from_record(RECORD_Y["query"])
        self.result = self.engine.run(self.query)
        self.grads = self.engine.backward(self.result)

    def test_zero_factor_determines_value_despite_missing(self):
        comp = self.result.components[0]
        self.assertEqual(comp.status, "available")
        self.assertEqual(comp.value, 0.0)

    def test_zero_factor_gradient_still_undetermined(self):
        status, reason = self.grads.status("P", "00")
        self.assertEqual(status, "undetermined")
        self.assertIn("缺", reason)
        self.assertIsNone(self.grads.for_block("P", "00"))

    def test_missing_sector_reported(self):
        self.assertEqual(self.grads.status("Q", "00")[0], "missing")

    def test_available_sector_trains_normally(self):
        np.testing.assert_array_equal(self.grads.for_block("P", "11"), [[2.0]])
        np.testing.assert_array_equal(self.grads.for_block("Q", "11"), [[5.0]])


class OrdinaryZeroReceivesGradientTests(unittest.TestCase):
    def test_ordinary_zero_parameter_receives_gradient(self):
        engine = Engine.from_record(RECORD_W)
        result = engine.run(Query.from_record(RECORD_W["query"]))
        self.assertEqual(result.components[0].value, 0.0)
        grads = engine.backward(result)
        status, _ = grads.status("P", "00")
        self.assertEqual(status, "full")
        np.testing.assert_array_equal(grads.for_block("P", "00"), [[4.0]])


class BoundaryBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.engine = Engine.from_record(self.record)

    def test_parity_inconsistent_sector_is_determined_zero(self):
        # i=0, o=1 沿链要求 k=0，与节点 D 的 parity=1 矛盾：空和即确定零。
        query = Query(("i", "o"), ("01",), ((0, 0),))
        result = self.engine.run(query)
        comp = result.components[0]
        self.assertEqual(comp.status, "available")
        self.assertEqual(comp.value, 0.0)
        self.assertIn("确定零", comp.reason)
        self.assertEqual(result.loss, 0.0)

    def test_backward_over_restricted_request_reports(self):
        engine = Engine.from_record(self.record, scratch_bytes=8)
        result = engine.run(self.engine_query())
        grads = engine.backward(result)
        report = grads.eligibility()
        outputs = {o["index"]: o for o in report["outputs"]}
        self.assertFalse(outputs[0]["trainable"])
        self.assertIn("预算", outputs[0]["reason"])
        params = {(p["block"], p["sector"]): p for p in report["parameters"]}
        self.assertEqual(params[("S", "00")]["status"], "restricted")

    def engine_query(self):
        return Query.from_record(self.record["query"])

    def test_replacement_shape_must_match_sector(self):
        with self.assertRaises(RecordError):
            self.engine.with_replacement("B", "111", [[1.0]])

    def test_incomplete_indices_rejected(self):
        import copy

        broken = copy.deepcopy(self.record)
        del broken["nodes"][2]["indices"]["11"]
        with self.assertRaises(RecordError):
            Engine.from_record(broken)

    def test_undeclared_legal_sector_rejected(self):
        import copy

        broken = copy.deepcopy(self.record)
        del broken["blocks"]["B"]["010"]
        with self.assertRaises(RecordError):
            Engine.from_record(broken)

    def test_verify_rejects_changed_structure(self):
        other = Engine.from_record(RECORD_W)
        report = other.verify(self.engine.run(self.engine_query()).proof())
        self.assertFalse(report.structure_ok)
        self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
