"""Whole-set crop identity checks, separate from training and biological gates."""


def check_indices(indices, original_members, expected_tiles_by_batch):
    originals = {m["sample_id"]: m for m in original_members if m["role"] == "image"}
    eligible = {n for n, m in originals.items() if m["split"] in {"train", "validation"}}
    seen, parents, by_batch, hashes = set(), set(), {}, {}
    group_splits, parent_batches = {}, {}
    total_bytes = 0
    for index in indices:
        if not isinstance(index, list) or not index:
            raise ValueError("Every retained crop index must contain actual verified members")
        for tile in index:
            name = tile["sample_id"]
            if name not in eligible:
                raise ValueError("Crop index contains an unknown or held-out parent")
            parent = originals[name]
            if tile["parent_file_id"] != parent["file_id"] or any(tile[k] != parent[k] for k in ["group_id", "split"]):
                raise ValueError("Crop parent/group/split differs from the frozen original")
            batch = tile["preparation_batch"]
            if isinstance(batch, bool) or not isinstance(batch, int) or batch not in expected_tiles_by_batch:
                raise ValueError("Crop belongs to an unplanned preparation batch")
            identity = (tile["parent_file_id"], tile["file_name"])
            if identity in seen:
                raise ValueError("A derived crop appears more than once in the whole-set index")
            seen.add(identity)
            if name in parent_batches and parent_batches[name] != batch:
                raise ValueError("A parent was transformed in multiple selected batches")
            parent_batches[name] = batch
            sha = tile["sha256"]
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("Crop identity is not a SHA-256 digest")
            size = tile["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 20 * 1024 * 1024:
                raise ValueError("Crop byte length is outside the retained input contract")
            if sha in hashes and hashes[sha][0] != tile["split"]:
                raise ValueError("Byte-identical derived crops cross frozen partitions")
            if sha in hashes and hashes[sha][1] != size:
                raise ValueError("A repeated crop digest has conflicting byte lengths")
            hashes[sha] = (tile["split"], size)
            split = group_splits.setdefault(tile["group_id"], tile["split"])
            if split != tile["split"]:
                raise ValueError("A frozen group crosses partitions")
            parents.add(name)
            by_batch[batch] = by_batch.get(batch, 0) + 1
            total_bytes += size
    if parents != eligible:
        raise ValueError("Whole-set verification does not cover every frozen training/validation original")
    if by_batch != expected_tiles_by_batch:
        raise ValueError("Whole-set crop counts differ from the completed per-archive verification receipts")
    return {"source_plates": len(parents), "derived_tiles": len(seen), "unique_crop_byte_digests": len(hashes),
            "derived_image_bytes": total_bytes, "cross_split_identical_crops": 0,
            "original_plates_excluded": 0, "test_outcomes_inspected": False, "training_executed": False}
