# Plate Colony Counter

Upload one plate image; the experimental counter returns the number of detected visible colonies and an annotated image. Inspect every outlined detection. Open the receipt for original-image coordinates, scores, fixed weights/configuration and artifact identities. Counts are not CFU/mL.

The saved model was evaluated on 37 original ADBC plates with 6,409 reference colonies. Aggregate count error was 2.9022%, precision 91.4467% and recall 92.4169%. The 95th-percentile symmetric per-plate error was 22.2222%, exceeding the approved 20% limit. This tool is experimental research assistance and has not passed the original qualification gate. Sparse plates can have substantial relative error. New laboratories, blank controls and merged growth remain unvalidated.

This separate single-image composition uses the already-trained YOLOX-Tiny weights and fixed pretest selection from [Recorded Training and Evaluation](https://hub.biosimulant.com/labs/70c5eb8d-6822-4ebe-8247-0619ee05fd29). It has no annotation, frozen-split, source or weights upload ports. Thirty local staging software checks passed. Actual Modal T4 runs on sparse, moderate and dense plates returned 7, 71 and 620 colonies; every box and score exactly matched the saved evaluator predictions, and annotated PNG and text visuals were durably retained. Hosted/browser execution is still pending. No additional training or tuning against the opened final test occurs.

YOLOX source: Apache-2.0, commit 6ddff4824372906469a7fae2dc3206c7aa4bbaee, retained license and unchanged regular runtime files. SAHI 0.11.21: MIT. ADBC dataset evidence: CC-BY-4.0, with attribution in the research release. This adapter: MIT.
