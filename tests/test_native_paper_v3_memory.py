"""Resource-counter fixtures for promoted absolute-headroom admission."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from native_paper_v3_memory import admission_snapshot, AdmissionError, cache_credit, GIB


class V3MemoryTests(unittest.TestCase):
    def test_credit_only_clean_inactive_file(self):
        stat = dict(inactive_file=8*GIB, file=12*GIB, shmem=2*GIB,
                    file_dirty=GIB, file_writeback=GIB)
        self.assertEqual(cache_credit(stat, 20*GIB, 2)[0], 6*GIB)
        stat["shmem"] = 13*GIB
        self.assertEqual(cache_credit(stat, 20*GIB, 2)[0], 0)

    def test_min_absolute_headroom_across_ancestors(self):
        runtime = dict(memory_limit_bytes=20*GIB, memory_sources=[
            dict(limit_bytes=20*GIB, current_file="/child/memory.current"),
            dict(limit_bytes=30*GIB, current_file="/parent/memory.current")])
        files = {
            "/child/memory.current":str(18*GIB), "/child/memory.max":str(20*GIB),
            "/child/memory.stat":f"inactive_file {8*GIB}\nfile {10*GIB}\nshmem 0\nfile_dirty 0\nfile_writeback 0\n",
            "/parent/memory.current":str(25*GIB), "/parent/memory.max":str(30*GIB),
            "/parent/memory.stat":"inactive_file 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n"}
        value = admission_snapshot(runtime, lambda:dict(total=64*GIB, available=40*GIB), lambda path:files[str(path).replace('\\', '/')])
        self.assertEqual(value["absolute_headroom_bytes"], 5*GIB)
        self.assertEqual(value["reservation_slots"], 2)

    def test_missing_required_current_counter_fails_closed(self):
        runtime = dict(memory_limit_bytes=20*GIB, memory_sources=[dict(limit_bytes=20*GIB, current_file="/c/memory.current")])
        def unreadable(path):
            raise OSError("counter unavailable")
        with self.assertRaises(AdmissionError):
            admission_snapshot(runtime, lambda:dict(total=64*GIB, available=40*GIB), unreadable)


if __name__ == "__main__":
    unittest.main()
