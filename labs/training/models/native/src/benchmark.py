"""Finite managed GPU timing of native YOLOX loss/backward on retained crops."""

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time

from biosimulant import BioModule, BioSignal, ExecutionPolicy, SignalSpec

from .data import TrainingArchive, loss_targets
from .source import extract_native_source
from .process import absolute_inputs, launch

SCHEMA = {"stage": "str", "source": "json", "environment": "json", "cases": "json",
          "training_steps_executed": "int", "test_outcomes_inspected": "bool",
          "scientific_acceptance_established": "bool"}


def file_path(inputs, name):
    value = inputs.get(name)
    value = value.value if isinstance(value, BioSignal) else value
    if not isinstance(value, str) or not value:
        raise ValueError(f"Managed benchmark requires retained File input {name}")
    return value


def tensors(data, batch_size, torch, numpy):
    tiles = data.tiles[:batch_size]
    if len(tiles) != batch_size:
        raise ValueError("Benchmark cannot silently repeat crops to fill its batch")
    # Native inference/training use BGR float32 0..255, with bottom/right padding114.
    # All prepared windows are <=640; no resizing, HSV/mosaic or target filtering.
    images = numpy.full((batch_size, 3, 640, 640), 114, dtype=numpy.float32)
    rows = [loss_targets(t) for t in tiles]
    capacity = max(1, max(map(len, rows)))
    labels = numpy.zeros((batch_size, capacity, 5), dtype=numpy.float32)
    for i, tile in enumerate(tiles):
        with data.image(tile) as image:
            rgb = numpy.asarray(image)
            images[i, :, :tile["height"], :tile["width"]] = rgb[:, :, ::-1].transpose(2, 0, 1)
        if rows[i]:
            labels[i, :len(rows[i])] = rows[i]
    return torch.from_numpy(images).cuda(), torch.from_numpy(labels).cuda(), [len(r) for r in rows]


def run(inputs, plan, root, *, started, timed_steps, precision="float32"):
    source = extract_native_source(file_path(inputs, "source_archive"), file_path(inputs, "source_receipt"),
                                   root / "native", archive_pin=plan["source_archive"], manifest_pin=plan["source_receipt"])
    data = TrainingArchive(file_path(inputs, "prepared_archive"), file_path(inputs, "frozen_split"), plan)
    sys.path.insert(0, str(root / "native"))
    try:
        if any(n == "yolox" or n.startswith("yolox.") for n in sys.modules):
            raise RuntimeError("Native source import would reuse an unverified cached YOLOX module")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy
        import torch
        import torchvision
        from yolox.exp import Exp

        if not torch.cuda.is_available():
            raise RuntimeError("Managed benchmark requires declared NVIDIA GPU execution; CPU fallback is forbidden")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        if numpy.__version__ != "1.26.4" or importlib.metadata.version("numpy") != numpy.__version__:
            raise RuntimeError("Loaded NumPy differs from the pinned native worker environment")
        cv = importlib.metadata.version("torch")
        if cv.split("+")[0] != "2.6.0" or torchvision.__version__.split("+")[0] != "0.21.0":
            raise RuntimeError("Managed Torch/torchvision differs from the pinned compatibility candidate")
        cases, executed = [], 0
        amp = precision == "amp_float16"
        for batch_size in ((8, 16) if amp else (2, 4, 8)):
            if time.monotonic() - started >= 480:
                cases.append({"batch_size": batch_size, "status": "not_started_before_internal_deadline"})
                continue
            random.seed(20261001)
            numpy.random.seed(20261001)
            torch.manual_seed(20261001)
            torch.cuda.manual_seed_all(20261001)
            exp = Exp()
            exp.depth, exp.width, exp.num_classes = 0.33, 0.375, 1
            exp.input_size, exp.test_size = (640, 640), (640, 640)
            exp.warmup_epochs = 0  # Positive native scaled LR for real optimizer updates.
            model = exp.get_model().cuda()
            optimizer = exp.get_optimizer(batch_size)
            scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128)
            torch.cuda.reset_peak_memory_stats()
            durations, losses, target_counts = [], [], []
            initial_weight = next(model.parameters()).detach().clone()
            try:
                for step in range(timed_steps + 2):
                    if time.monotonic() - started >= 480:
                        break
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    images, targets, target_counts = tensors(data, batch_size, torch, numpy)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
                        output = model(images, targets)
                        loss = output["total_loss"]
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("Native training loss is nonfinite")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                        raise RuntimeError("Native training gradient is nonfinite")
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - before
                    executed += 1
                    if step >= 2:
                        durations.append(elapsed)
                        losses.append(float(loss.item()))
                    del images, targets, output, loss
                changed = not torch.equal(initial_weight, next(model.parameters()).detach())
                if not durations or not changed:
                    raise RuntimeError("Benchmark has no timed optimizer updates or its model weights did not change")
                cases.append({"batch_size": batch_size, "status": "measured", "timed_steps": len(durations),
                              "seconds_per_step_mean": sum(durations) / len(durations),
                              "seconds_per_step_maximum": max(durations), "end_to_end_crops_per_second": batch_size * len(durations) / sum(durations),
                              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "target_counts_per_crop": target_counts,
                              "targets_truncated": 0, "native_total_losses": losses, "weights_changed": changed,
                              "precision": precision, "final_gradient_scale": scaler.get_scale(),
                              "optimizer": {"name": "native_SGD", "lr": exp.basic_lr_per_img * batch_size,
                                            "momentum": exp.momentum, "weight_decay": exp.weight_decay, "nesterov": True}})
            except torch.cuda.OutOfMemoryError:
                # Retain the actual failed case; never manufacture a successful capacity.
                cases.append({"batch_size": batch_size, "status": "cuda_out_of_memory", "timed_steps": len(durations)})
                break
            finally:
                del model, optimizer, scaler, exp, initial_weight
                gc.collect()
                torch.cuda.empty_cache()
        if not any(c["status"] == "measured" for c in cases):
            raise RuntimeError("No native training benchmark case completed")
        environment = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                       "numpy": numpy.__version__, "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "device": torch.cuda.get_device_name(0), "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                       "seed": 20261001, "precision": precision, "deterministic_algorithms": True,
                       "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
        source.update({"code_executed": True, "initialization": "random; no third-party checkpoint", "preparation_batch": plan["preparation_batch"],
                       "archive_sha256": plan["archive"]["sha256"], "training_tiles_in_archive": len(data.tiles),
                       "maximum_training_targets_in_archive": max(len(t["targets"]) for t in data.tiles),
                       "dense_crop_order": "descending target count then SHA256 then name", "augmentation": "none for compatibility/timing",
                       "throughput_scope": "dense crops; includes JPEG decode, BGR packing, transfer, native loss, backward, gradient checks and SGD; excludes full epoch augmentation/validation"})
        return {"stage": "native_gpu_training_compatibility_benchmark", "source": source, "environment": environment, "cases": cases,
                "training_steps_executed": executed, "test_outcomes_inspected": False, "scientific_acceptance_established": False}
    finally:
        data.close()
        sys.path.remove(str(root / "native"))


class BenchmarkNativeDetector(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self, timed_steps=12, precision="float32"):
        if isinstance(timed_steps, bool) or not isinstance(timed_steps, int) or not 4 <= timed_steps <= 30:
            raise ValueError("Benchmark timed steps must be between4 and30 per batch size")
        if precision not in {"float32", "amp_float16"}:
            raise ValueError("Benchmark precision must be float32 or amp_float16")
        self.timed_steps = timed_steps
        self.precision = precision

    def inputs(self):
        return {n: SignalSpec.scalar(dtype="str", value_type="file", format=kind)
                for n, kind in [("source_archive", "zip"), ("source_receipt", "json"), ("prepared_archive", "zip"), ("frozen_split", "json")]}

    def outputs(self):
        return {"receipt": SignalSpec.record(schema=SCHEMA, emitted_unit="1")}

    def execute(self, inputs, *, context):
        root = Path(tempfile.mkdtemp(prefix="cfu-benchmark-", dir=Path.cwd()))
        root.chmod(0o700)
        plan = json.loads((Path(__file__).resolve().parent.parent / "benchmark-plan.json").read_text())
        values = absolute_inputs({name: file_path(inputs, name) for name in self.inputs()})
        request = {"inputs": values, "plan": plan, "timed_steps": self.timed_steps, "precision": self.precision}
        return {"receipt": launch(request, root, Path(__file__).resolve().parent.parent)}
