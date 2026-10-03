"""Fresh managed native training worker after pinned dependency resolution."""

import json
from pathlib import Path
import sys

from .training_lib import run


def main():
    request_path,result_path=map(Path,sys.argv[1:])
    request=json.loads(request_path.read_bytes())
    outputs=run(request['inputs'],request['plan'],request_path.parent,work_seconds=request['work_seconds'],
                checkpoint_pin=request['checkpoint_pin'],weights_pin=request['weights_pin'])
    result_path.write_text(json.dumps(outputs,allow_nan=False));result_path.chmod(0o600)


if __name__=='__main__':main()
