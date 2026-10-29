"""The MCP front end: tools validate through the Session, results carry their caveats.

Skipped when the optional `mcp` package (>= 2.0) is not installed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from mcp import Client
    from mcp.server.mcpserver import MCPServer  # noqa: F401  (mcp >= 2)
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

from openkapitza.agent.mcp_server import INSTRUCTIONS, OpenKapitzaTools

ROOT = Path(__file__).resolve().parents[1]
JOINT = {"material_a": "Si", "material_b": "Al", "temperature_K": "300 K",
         "pressure_MPa": "5 MPa", "sigma_a_um": "0.05 um", "sigma_b_um": "200 nm",
         "slope_a": 0.02, "slope_b": 0.05, "medium": "vacuum", "clamping": "uniform"}


class TestTools(unittest.TestCase):
    """The tool logic without MCP plumbing."""

    def setUp(self):
        self.t = OpenKapitzaTools(n_samples=100)

    def test_set_inputs_converts_and_rejects_with_reasons(self):
        d = self.t.set_inputs({"material_a": "silicon", "pressure_MPa": "50 psi",
                               "sigma_a_um": "-1 um", "nonsense": 3})
        self.assertEqual(d["accepted"]["material_a"], "Si")
        self.assertAlmostEqual(d["accepted"]["pressure_MPa"], 0.344738, places=5)
        self.assertIn("greater than 0", d["rejected"]["sigma_a_um"])
        self.assertIn("unknown field", d["rejected"]["nonsense"])
        self.assertFalse(d["ready_to_run"])
        self.assertTrue(d["missing_required"])

    def test_run_reports_caveats_and_defaults(self):
        self.t.set_inputs(JOINT)
        d = self.t.run_joint()
        self.assertIn("Joint conductance", d["explanation"])
        codes = {m["code"] for m in d["must_report"]}
        self.assertIn("KAPITZA_PLACEHOLDER", codes)       # no MD yet: must be said
        self.assertTrue(all(m["severity"] in ("CAUTION", "WARNING", "CRITICAL")
                            for m in d["must_report"]))
        self.assertIn("phase_a", d["defaults_used"])
        self.assertGreater(d["result"]["h_joint_W_m2K"], 0)

    def test_run_refuses_when_required_inputs_missing(self):
        self.t.set_inputs({"material_a": "Si"})
        with self.assertRaises(ValueError):
            self.t.run_joint()

    def test_changing_inputs_invalidates_the_result(self):
        self.t.set_inputs(JOINT)
        self.t.run_joint()
        self.t.set_inputs({"pressure_MPa": "1 MPa"})
        self.assertIsNone(self.t.s.result)

    def test_what_if_sweep_lookup(self):
        self.t.set_inputs(JOINT)
        self.t.run_joint()
        self.assertIn("With pressure_MPa = 1", self.t.what_if("pressure_MPa", "1")["answer"])
        sw = self.t.sweep("pressure_MPa", "1:10:3")
        self.assertEqual(len(sw["points"]), 3)
        hits = self.t.lookup("why can't I add 1/h_c and 1/h_K in series", k=2)["hits"]
        self.assertTrue(hits and hits[0]["citation"].startswith("docs/"))

    def test_md_ingest_through_tools(self):
        self.t.set_inputs(JOINT)
        self.t.run_joint()
        msg = self.t.ingest_md(str(ROOT / "tests/data/md_runs/sial_plumbing"))["answer"]
        self.assertIn("Re-ran the joint with the MD h_K", msg)
        d = self.t.run_joint()
        self.assertIsNotNone(d["md_used"])

    def test_describe_fields(self):
        f = {x["name"]: x for x in self.t.describe_fields()["fields"]}
        self.assertIn("material_a", f)
        self.assertIn("Si", f["material_a"]["choices"])
        self.assertNotIn("z_trunc", f)                    # advanced hidden by default
        self.assertIn("z_trunc", {x["name"] for x in
                                  self.t.describe_fields(True)["fields"]})

    def test_session_tools_and_reset(self):
        self.t.set_inputs(JOINT)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "s.json")
            self.t.save_session(p)
            self.t.reset_session()
            self.assertEqual(self.t.show_inputs()["inputs"], {})
            self.assertEqual(self.t.load_session(p)["inputs"]["material_b"], "Al")

    def test_instructions_carry_the_rules(self):
        for phrase in ("Never invent", "must_report", "user's yes", "do not apply them"):
            self.assertIn(phrase, INSTRUCTIONS)


@unittest.skipUnless(HAVE_MCP, "mcp >= 2 not installed (pip install 'openkapitza[mcp]')")
class TestServer(unittest.TestCase):
    def test_in_process_client(self):
        from openkapitza.agent.mcp_server import build_server

        async def go():
            async with Client(build_server(n_samples=100)) as c:
                self.assertIn("Never invent", c.instructions)
                names = {t.name for t in (await c.list_tools()).tools}
                for n in ("describe_fields", "set_inputs", "run_joint", "lookup",
                          "submit_jobs", "job_status", "ingest_md"):
                    self.assertIn(n, names)
                tools = {t.name: t for t in (await c.list_tools()).tools}
                self.assertTrue(tools["lookup"].annotations.read_only_hint)
                self.assertTrue(tools["submit_jobs"].annotations.open_world_hint)
                r = await c.call_tool("set_inputs", {"values": JOINT})
                self.assertFalse(r.is_error)
                r = await c.call_tool("run_joint", {})
                d = json.loads(r.content[0].text)
                self.assertTrue(d["must_report"])
                r = await c.call_tool("ingest_md", {"path": "/no/such/folder"})
                self.assertTrue(r.is_error)                     # a tool error, not a crash
                self.assertIn("does not exist", r.content[0].text)
                self.assertEqual([p.name for p in (await c.list_prompts()).prompts],
                                 ["analyse_joint"])
        asyncio.run(go())

    def test_stdio_entry_point(self):
        """The command an MCP client would launch actually serves."""
        from mcp.client.stdio import StdioServerParameters

        async def go():
            params = StdioServerParameters(command=sys.executable,
                                           args=["-m", "openkapitza.agent.mcp_server",
                                                 "--samples", "50"], cwd=str(ROOT))
            async with Client(params) as c:
                r = await c.call_tool("lookup", {"query": "fix ave/chunk", "k": 1})
                self.assertFalse(r.is_error)
                self.assertTrue(json.loads(r.content[0].text)["hits"])
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
