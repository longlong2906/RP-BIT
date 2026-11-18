"""Runner recovery and archive tests. No registered pilot solve is launched."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))

from corridor_v4_execution import append_journal, atomic_json as dump, digest, file_hash, read
from native_paper_v4_storage import ArtifactStore
from run_corridor_v4 import _commit_spool, raw_inventory
from run_sem_exposure_pilot import (_audited_summary, _groups, _journal, _prepare_map, _read_sealed,
    _recover_map, _seal, _verify_map, _waves, analyze, prepare, protocol, run)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="pilot-", dir=ROOT/"tests"/"fixtures")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_frozen_inventory_has_no_attempts_and_is_analyzable(self):
        config = protocol.load_config()
        manifest = prepare(self.root)
        self.assertEqual([4]*12+[2], [len(maps) for _, maps in _waves(manifest["entries"])])
        with self.assertRaises(FileExistsError):
            prepare(self.root)
        frozen, summary = _audited_summary(self.root/"tune", config)
        self.assertEqual((2500, 0, 0), (summary["planned"], summary["attempted"], summary["validated"]))
        self.assertEqual("PREPARED", frozen["status"])
        self.assertEqual("INCOMPLETE", analyze(self.root, "tune")["selection"]["status"])
        self.assertEqual("INCOMPLETE", analyze(self.root, "tune")["selection"]["status"])
        with self.assertRaises(ValueError):
            run(self.root, "tune", None, None, None, None)
        with self.assertRaises(ValueError):
            prepare(self.root, phase="confirm")
        self.assertFalse((self.root/"confirm"/"manifest.json").exists())
        frozen["entries"][0]["planner_seed"] = 11
        dump(self.root/"tune"/"manifest.json", frozen)
        with self.assertRaisesRegex(ValueError, "schedule"):
            _audited_summary(self.root/"tune", config)

    def test_started_interruption_remains_a_failure_and_seal_is_audited(self):
        store = ArtifactStore(self.root)
        wave, mid = "wave_000", "terrain"
        folder = store._paths(wave, mid)[0]
        inputs = folder/"inputs"
        inputs.mkdir(parents=True)
        model, cloud = inputs/"map.t25d", inputs/"map.cloud.npy"
        model.write_bytes(b"synthetic runner test")
        cloud.write_bytes(b"synthetic runner test cloud")
        query = dict(query_id="q0", query_hash="test", start=[1., 2.], goal=[3., 4.])
        dump(model.with_suffix(".json"), dict(queries=[query], map_id=mid,
             generator_seed=37001, family="traversable_hill"))
        info = dict(map_id=mid, map_path=str(model), cloud_path=str(cloud), map_sha256=file_hash(model),
                    metadata_sha256=file_hash(model.with_suffix(".json")), cloud_sha256=file_hash(cloud),
                    map_seed=37001, family="traversable_hill", bounds=[0., 0., 40., 40.], **query)
        entry = dict(trial_id="attempt", map_id=mid, method=protocol.CONTROL, sem_lambda=0.,
                     family="traversable_hill", planner_seed=376001, map_seed=37001, query_id="q0",
                     phase="tune", alpha=None, policy="full_safe_exposure", status="NOT_RUN")
        manifest = dict(config_sha256="config", sources={"test": "source"}, entries=[entry],
                        maps=[info], shards={}, phase="tune")
        journal = folder/"execution_control"/"start_journal.jsonl"
        append_journal(journal, dict(trial_id="attempt", config_sha256="config", sources_sha256=digest(manifest["sources"]),
                                    started_utc="2026-11-12T00:00:00Z", lane=0, core=1, queue_wait_s=0.))
        _recover_map(folder, [entry], manifest)
        self.assertEqual("RAW_READY", entry["status"])
        trial = folder/"trials"/"attempt"
        self.assertEqual("INTERRUPTED", read(trial/"execution_recovery.json")["status"])
        self.assertEqual(raw_inventory(trial), read(trial/"raw_inventory.json"))
        before = entry["raw_inventory_sha256"]
        _recover_map(folder, [entry], manifest)
        self.assertEqual(before, entry["raw_inventory_sha256"])
        spool = folder/"validation_spool"
        row = {key: entry[key] for key in ("trial_id", "map_id", "method", "family", "planner_seed", "sem_lambda")}
        row.update(success=False, execution_status="INTERRUPTED", execution_launched=False, correctness_errors=[])
        dump(spool/"attempt.json", dict(row=row, identity=dict(trial_id="attempt", raw_inventory_sha256=before,
            sources_sha256=digest(manifest["sources"]), info_sha256=digest(info))))
        _commit_spool(folder, entry, manifest, spool)
        self.assertEqual("INTERRUPTED", entry["status"])
        # Simulate interruption after the shard transaction, before the phase
        # manifest receives the shard receipt.
        store.pack_map(wave, mid, cleanup=True)
        self.assertFalse(folder.exists())
        _seal(store, manifest, wave, mid)
        observed = _read_sealed(store, wave, mid, [entry], manifest, info)
        self.assertEqual([(False, "INTERRUPTED")], [(r["success"], r["execution_status"]) for r in observed])
        with self.assertRaisesRegex(ValueError, "map/query"):
            _read_sealed(store, wave, mid, [entry], manifest, dict(info, goal=[2., 4.]))
        journal_line = json.dumps(dict(trial_id="attempt", config_sha256="config",
                                      sources_sha256=digest(manifest["sources"])))
        with self.assertRaisesRegex(ValueError, "journal"):
            _journal(journal_line+"\n"+journal_line, [entry], manifest)
        archive = store._paths(wave, mid)[1]
        with archive.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "archive identity"):
            _read_sealed(store, wave, mid, [entry], manifest, info)

    def test_registered_map_preparation_uses_real_q0_and_frozen_cloud(self):
        config = protocol.load_config()
        entries = protocol.inventory(config)["entries"]
        first = next(iter(_groups(entries).values()))[0]
        result = _prepare_map(self.root/"map", first)
        _verify_map(result)
        meta = read(Path(result["map_path"]).with_suffix(".json"))
        self.assertEqual("q0", meta["queries"][0]["query_id"])
        self.assertEqual(result["query_hash"], meta["queries"][0]["query_hash"])
        self.assertEqual(result["cloud_sha256"], file_hash(result["cloud_path"]))
        self.assertEqual([0., 0., 40., 40.], result["bounds"])


if __name__ == "__main__":
    unittest.main()
