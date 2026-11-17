"""Focused tests for the handoff's historical mixed-schema ledger parser."""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("handoff_export", Path(__file__).with_name("export.py"))
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class LedgerTests(unittest.TestCase):
    def test_both_real_schemas(self):
        text = (
            "id,status,server,gpus,config,best_miou,best_acc_05,best_acc_025,best_spmiou,best_epoch,notes\n"
            'AdaRES-B1,done_complete,server-2,"0,1",b1.yaml,0.502,0.553,0.610,0.467,18,note\n'
            "2026-09-22 09:18:09,AdaRES-B5,b5.yaml,server-2,1,done_complete,0.5030,20/20,\n"
        )
        rows, errors = export.parse_ledger(text)
        self.assertEqual(errors, [])
        self.assertEqual([row["id"] for row in rows], ["AdaRES-B1", "AdaRES-B5"])
        self.assertEqual(rows[0]["gpus"], "0,1")
        self.assertEqual(rows[1]["state"], "done_complete")
        self.assertEqual(rows[1]["best_miou"], "0.5030")

    def test_bad_row_is_reported(self):
        rows, errors = export.parse_ledger("id,status\nnot-an-id,done\nAdaRES-Y1,waiting,extra\n")
        self.assertEqual(rows, [])
        self.assertEqual(len(errors), 2)

    def test_extension_id_and_sort(self):
        rows, errors = export.parse_ledger("id,status\nAdaRES-B2-E30,done_complete\n")
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["id"], "AdaRES-B2-E30")
        self.assertLess(export.id_sort("AdaRES-Y9"), export.id_sort("AdaRES-Y100"))

    def test_secret_detection_does_not_echo_value(self):
        value = b"ghp_" + b"a" * 36
        with self.assertRaisesRegex(RuntimeError, "Potential secret"):
            export.check_secrets(value, "example.txt")


if __name__ == "__main__":
    unittest.main()
