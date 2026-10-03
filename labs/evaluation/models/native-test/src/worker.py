"""Fresh managed process for a single locked test configuration."""
import json
from pathlib import Path
import sys

from .test_lib import run

if __name__ == '__main__':
    request=json.loads(Path(sys.argv[1]).read_bytes())
    result=run(request['inputs'],request['plan'],request['root'],weights_pin=request['weights_pin'],
               work_seconds=request['work_seconds'],lock_pin=request['lock_pin'],lock_digest=request['lock_digest'],
               evaluation_bundle=request['evaluation_bundle'],phase=request['phase'],baseline_pin=request['baseline_pin'])
    Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False))
