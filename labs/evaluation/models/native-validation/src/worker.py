"""Fresh managed process entry point for native original-image validation."""
import json
from pathlib import Path
import sys

from .validation_lib import run

if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    result = run(request['inputs'],request['plan'],request['root'],
                 weights_pin=request['weights_pin'],work_seconds=request['work_seconds'])
    Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False))
