"""Real SEM worker/C++ bridge check on a labelled synthetic test map only."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
from terrain_io import write_map
from native_point_cloud import generate_cloud
from native_paper_protocol import default_config, request_for, sha256
from native_paper_analysis import _validate_group, validation_code_identity
from experiment_common import dump

folder = Path(tempfile.mkdtemp(prefix='synthetic_real_sem_', dir=Path(__file__).parent))
map_path = folder/'map.t25d'
write_map(map_path, [[0.]*31 for _ in range(31)], spacing=(.2,.2),
          metadata={'synthetic_test_fixture': True})
cloud_path = folder/'cloud.npy'
np.save(cloud_path, generate_cloud(map_path), allow_pickle=False)
request = request_for('sem', {'start':[1.,3.], 'goal':[5.,3.]}, map_path,
                      cloud_path, sha256(cloud_path), 91823, 2., default_config())
request.update(bridge_library=str(ROOT/'build-native-paper/libterrain_compare_c.dll'),
               library_search_paths=[str(ROOT/'build-native-paper'), 'C:/mingw64/bin'],
               work_iterations=100)
trial = folder/'synthetic_sem_worker'
trial.mkdir()
dump(trial/'request.json', request)
command = [sys.executable, str(ROOT/'tools/native_paper_worker.py'), str(trial/'request.json'), str(trial)]
began=time.perf_counter()
ran = subprocess.run(command, capture_output=True, text=True, encoding='utf8', timeout=60)
(trial/'stdout.log').write_text(ran.stdout,encoding='utf8')
(trial/'stderr.log').write_text(ran.stderr,encoding='utf8')
dump(folder/'command.json',dict(argv=command,exit_code=ran.returncode,wall_s=time.perf_counter()-began))
assert ran.returncode==0, ran.stdout+ran.stderr
worker=json.loads((trial/'worker.json').read_text())
assert worker['status']=='EXACT_SOLUTION', worker
entry=dict(trial_id=trial.name,backend='sem',track='native',family='synthetic',map_id='synthetic',
           query_id='q0',planner_seed=91823,execution_status='COMPLETED',
           map_sha256=sha256(map_path),request=request)
row=_validate_group((folder,[entry],validation_code_identity()))[0]
dump(folder/'review_validation.json',row)
assert row['success'],row['errors']
print(json.dumps({'fixture':str(folder),'worker_status':worker['status'],'independently_valid_success':row['success'],
                  'snapshots':len(worker['snapshots']),'iterations':worker['counters']['iterations'],
                  'native_raster_artifacts':worker['native_model']['raster_artifacts'],
                  'phase_timings':worker['phase_timings']},indent=2))
