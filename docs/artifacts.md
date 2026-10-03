# Required counter assets

The immutable Hub release includes these assets under `models/native/artifacts/`.
They are deliberately excluded from Git. Restore them from the exact published
Biosimulant release, then verify length and SHA-256 against the contract before
execution. A raw checkout alone is not an inference-ready package.

Release: `demi/plate-colony-counter@0.1.1`. Package SHA-256:
`77cb99f47f032e9921ce653875a38d5dbd2b0e3757b2d322d61631a670c4d57d`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `ema-weights.pth` | 20372446 | `27068af634b80c51d3c318d8fd7e8c1d79269fa00ce62b557f5b9cf3e5d9cd90` |
| `yolox-source.zip` | 111995 | `12871d9c8e7d3f45e67995bfec9a175b9f27496741ef197a604c0653670ae4dc` |
| `yolox-source.json` | 63830 | `03cd32cc8b56f144d9f63e173e5b28826af2c401def5a94af9d56f38b1248349` |
| `sahi-source.zip` | 114361 | `31fa3dab37284e5cf0e404e74505781523a7faab59f9b2e1f847a436d9b6cf11` |
| `sahi-source.json` | 11972 | `43cf50681fbdacd5aa4c8216516c30b5a5b1e2fce78e6e065b6e64715335489a` |

The checked-in `inference-contract.json` pins all 14 authored source files,
the five assets, fixed preprocessing/postprocessing settings and required
scientific environment. The model verifies these identities before accepting
outputs. Repository relocation does not relax these checks.

Training/evaluation components also require their declared retained dataset,
source and checkpoint inputs. Source IDs and recipe contracts are provenance;
they do not make private account-owned files available to another account.

No generated predictions, plate photographs, private operational logs,
signed download links, credentials or local model caches are included here.
