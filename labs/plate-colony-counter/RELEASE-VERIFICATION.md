# Single-image release verification

Status: private package delivered; hosted deployment pending new production approval.

Source baseline: backend main e35efb4bdae0e19226895ffd88a1db9989606b1e. Staging commits: 4b00e3e10a312bad219de4f6c624c312e0651d6e, then 8fc45168113df65d6f6e238710d538edefca248e. Staging API and worker both report 8fc4516, SDK 0.0.34, ready. Production API/worker remained at e35efb4 when checked. No backend production deployment or provider change was performed.

The original JPEG preview run returned exact saved detections but its image was not retained; the run Passport was BLOCKED for visual asset delivery. PNG preview revision 3 corrected retention without changing detector or count semantics.

Private workspace a0e446f6-9716-44c2-b8d1-281a974a52e5, science-tested revision d0f9f8a5-d00d-4aed-9a14-ac6c73365931, SHA b1218a5983ade2da0ac015241028ea245ab8b56d1cddd84d8077e97ae0e8908d. Study 1cae9e15-fd0b-49cb-baea-349bb392e760 is completed.

| Plate | Returned count | Exact saved box/score parity | Inference including source setup and PNG | Managed command |
|---|---:|---|---:|---:|
| sp01_img04.jpg | 7 | Passed | 21.91 s | 56.68 s |
| sp22_img03.jpg | 71 | Passed | 16.04 s | 51.59 s |
| sp10_img20.jpg | 620 | Passed | 17.85 s | 52.89 s |

All three result JSON and annotated PNG byte lengths/SHA-256 were independently verified. All previews used durable artifact IDs; image and text visualization checks passed. Source/runtime guards confirmed Python 3.12.3, Torch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 1.26.4, CUDA 12.4 and cuDNN 90100. No weights were deserialized or detector executed locally. Local staging checks: 30 passed; Ruff F/E9 passed. The portable test subset is included separately.

This proves serving parity on these three fixtures, not fresh accuracy or completed hosting. It retains the failed original p95 gate, sparse false positives and uncompleted independent recreation. Browser upload, cold/warm hosted execution and separate-account access remain pending.

Private release demi/plate-colony-counter@0.1.0, artifact 2ff2abf5-dad4-44bb-8ab2-55b72003b700, ZIP SHA 82424834d0aa6b25446d4fd32c2a1000428aab42b43f41285f7862cc8391cb92, 18,919,566 bytes. No public release was executed.

New compute prepared upper estimates total $1.028160, added to the retained previous conservative envelope $22.641276, giving $23.669436 against the $50 cap. Actual provider invoices and full infrastructure durations were not obtained. All four managed runs are terminal. Private hosting uses one T4, scales to zero after 300 seconds idle and needs administrator GPU approval. No hosting container was requested yet.

Hub public Results/Experiments are separate from owned workspace records. Publication alone does not expose private runs; native MCP currently lacks run-public-sharing controls. Do not report those public tabs populated. The guide records verified evidence while public sharing capability is resolved.
