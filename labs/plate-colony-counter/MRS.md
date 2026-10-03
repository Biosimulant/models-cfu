# Plate Colony Counter requirements

Authorized scope: experimental single-image serving of the saved research candidate. This supplements the recorded CFU requirements without changing their scientific acceptance gates.

Accept one account-authorized JPEG/PNG plate image (20 MiB, 25 million pixels maximum); return an integer visible-colony count equal to the complete retained detections, a bounded annotated PNG, and an auditable receipt. Preserve EXIF-normalized original coordinates and the immutable selected weights and configuration. Display the count, outlined colonies and research limitations from actual computed outputs; reopening must retain the image. Reject corrupt, animated, oversized or unauthorized inputs and execution failures without manufacturing a count.

Before serving, compare complete detections and counts against saved fixed evaluator predictions on representative sparse, moderate and dense plates. This checks serving parity, not fresh model accuracy. No threshold tuning against the opened test is permitted. Actual managed and hosted execution, artifact checks and user upload/rendering remain required; local software fixtures are insufficient.

Retain the original failed test: WAPE 2.9022%, p95 symmetric per-plate error 22.2222% (approved maximum 20%), precision 91.4467%, recall 92.4169%. Label the tool experimental and require human review. Do not claim species, CFU/mL, sterility, clinical suitability, validation on new laboratories, calibrated uncertainty or completed independent training recreation.

Use Modal and the existing $50 total compute cap. Public release and hosting require their exact prepared approvals.
