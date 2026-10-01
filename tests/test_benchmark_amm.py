import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import benchmark_amm as benchmark


class BenchmarkTest(unittest.TestCase):
    def test_failures_never_create_or_overwrite_receipt(self):
        # Calls: reference, two warmups, two trials (forward then reverse).
        for call_index in range(7):
            failures = [ValueError("Replay failed"), [], [float("nan")], [float("inf")], [-float("inf")]]
            if call_index:
                failures += [[1.0], [-0.0]]
            for failure in failures:
                for existing in (False, True):
                    with self.subTest(call=call_index, failure=failure, existing=existing):
                        with tempfile.TemporaryDirectory() as directory:
                            output = Path(directory) / "receipt.json"
                            original = b'{"ordered_losses_equal": true, "ordered_losses": [0.0]}\n'
                            if existing:
                                output.write_bytes(original)
                            results = [[0.0] for _ in range(7)]
                            results[call_index] = failure
                            with patch.object(benchmark, "run", side_effect=results) as run:
                                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                                    with self.assertRaises(ValueError):
                                        benchmark.main(
                                            [
                                                "--baseline",
                                                "HEAD",
                                                "--candidate",
                                                "HEAD",
                                                "--windows",
                                                "1",
                                                "--warmup",
                                                "1",
                                                "--repeats",
                                                "2",
                                                "--output",
                                                str(output),
                                            ]
                                        )
                            self.assertEqual(run.call_count, call_index + 1)
                            self.assertEqual(stdout.getvalue(), "")
                            if existing:
                                self.assertEqual(output.read_bytes(), original)
                            else:
                                self.assertFalse(output.exists())

    def test_warmups_and_trials_execute_for_every_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(benchmark, "run", return_value=[0.0]) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main(
                        [
                            "--baseline",
                            "HEAD",
                            "HEAD",
                            "--candidate",
                            "HEAD",
                            "--windows",
                            "1",
                            "--warmup",
                            "2",
                            "--repeats",
                            "2",
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(run.call_count, 1 + 3 * 2 + 3 * 2)
            functions = [call.args[0] for call in run.call_args_list]
            a, b, c = functions[1:4]
            self.assertEqual(functions, [a, a, b, c, a, b, c, a, b, c, c, b, a])
            receipt = json.loads(output.read_text())
            self.assertTrue(receipt["ordered_losses_equal"])
            self.assertEqual(receipt["ordered_losses"], [0.0])
            self.assertEqual([len(times) for times in receipt["seconds"]], [2, 2, 2])

    def test_invalid_counts_are_rejected(self):
        for option, value in (("--windows", "0"), ("--windows", "-1"), ("--repeats", "0"), ("--warmup", "-1")):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(benchmark, "run") as run, self.assertRaises(SystemExit):
                    benchmark.main(["--baseline", "HEAD", option, value])
                run.assert_not_called()

    def test_empty_workloads_are_rejected(self):
        for tasks in ([], [[100, 0.003, []]]):
            with tempfile.TemporaryDirectory() as directory:
                workload = Path(directory) / "workload.json"
                workload.write_text(json.dumps({"tasks": tasks, "provenance": {}}))
                with contextlib.redirect_stderr(io.StringIO()):
                    with patch.object(benchmark, "run") as run, self.assertRaises(SystemExit):
                        benchmark.main(["--baseline", "HEAD", "--workload", str(workload)])
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
