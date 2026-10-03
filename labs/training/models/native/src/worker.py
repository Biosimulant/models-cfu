"""Fresh managed worker; scientific inputs remain retained MCP File references."""

import json
from pathlib import Path
import sys
import time

from .benchmark import run


def main():
    started = time.monotonic()
    request_path, receipt_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_bytes())
    root = request_path.parent
    receipt = run(request['inputs'], request['plan'], root, started=started, timed_steps=request['timed_steps'], precision=request.get('precision','float32'))
    receipt['environment']['execution_isolation'] = 'fresh managed Python subprocess after pinned dependency resolution'
    receipt_path.write_text(json.dumps(receipt, allow_nan=False))
    receipt_path.chmod(0o600)


if __name__ == '__main__':
    main()
