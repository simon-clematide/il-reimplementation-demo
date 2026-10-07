"""
Tests for demo_core loader and prediction presentation layers.
Uses the scratch test bundle trained in Phase 0.
"""

import os
import unittest
from demo_core.loader import load_bundle
from demo_core.predict import predict_single, explain_action
from trans import actions


class TestDemoCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.bundle_dir = os.path.join(root, "scratch", "out", "eng1000_mps")
        if not os.path.exists(cls.bundle_dir):
            cls.bundle_dir = os.path.join(root, "scratch", "out", "eng1000")

    def test_load_bundle_and_predict(self):
        if not os.path.exists(self.bundle_dir):
            self.skipTest(f"Bundle not found at {self.bundle_dir}")

        model, vocab, metadata = load_bundle(self.bundle_dir, device="cpu")
        self.assertIsNotNone(model)
        self.assertIsNotNone(vocab)
        self.assertIn("args", metadata)

        # Test greedy prediction
        res = predict_single(model, vocab, "walk", "V;PST", beam_width=1)
        self.assertEqual(res.lemma, "walk")
        self.assertTrue(len(res.predicted_form) > 0)
        self.assertIsInstance(res.decoder_score, float)
        self.assertTrue(len(res.action_trace) > 0)

        # Verify action trace fields
        first_act = res.action_trace[0]
        self.assertEqual(first_act.step, 1)
        self.assertIn(first_act.operation, ["COPY", "DELETE", "INSERT", "SUBSTITUTE", "END", "BEGIN"])

        # Test beam search prediction
        res_beam = predict_single(model, vocab, "walk", "V;PST", beam_width=4)
        self.assertEqual(res_beam.lemma, "walk")
        self.assertTrue(len(res_beam.predicted_form) > 0)

    def test_explain_action(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bundle_dir = os.path.join(root, "scratch", "out", "eng1000_mps")
        _, vocab, _ = load_bundle(bundle_dir, device="cpu")

        source = ["s", "i", "n", "g"]
        rec = explain_action(actions.ConditionalCopy(), vocab, source, 0, 1)
        self.assertEqual(rec.operation, "COPY")
        self.assertEqual(rec.source_char, "s")
        self.assertEqual(rec.emitted_char, "s")

        rec_sub = explain_action(actions.ConditionalSub(new="a"), vocab, source, 1, 2)
        self.assertEqual(rec_sub.operation, "SUBSTITUTE")
        self.assertEqual(rec_sub.source_char, "i")
        self.assertEqual(rec_sub.emitted_char, "a")


if __name__ == "__main__":
    unittest.main()
