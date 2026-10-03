"""Execute the immutable uploaded-image inference in a fresh managed interpreter."""
import json
from pathlib import Path
import sys

from .inference import run

if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_bytes())
    outputs = run(request, Path(__file__).resolve().parent.parent)
    destination = Path(sys.argv[2]); destination.write_text(json.dumps(outputs, allow_nan=False))
    destination.chmod(0o600)
