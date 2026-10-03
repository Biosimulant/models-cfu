"""Load the byte-verified native augmentation file without dataset initializers."""
import importlib.util
from pathlib import Path


def load_preproc(verified_source_root):
    root = Path(verified_source_root).resolve(strict=True)
    path = root/'yolox/data/data_augment.py'
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
            or not 0 < path.stat().st_size <= 128*1024):
        raise ValueError('Native preprocessing must come from its verified regular source file')
    # Same direct-file import method already executed by native training.
    # The source extractor verifies the complete archive/member digests first.
    spec = importlib.util.spec_from_file_location('_cfu_verified_native_preprocessing',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module,'preproc',None)):
        raise ValueError('Verified native source has no preprocessing function')
    return module.preproc
