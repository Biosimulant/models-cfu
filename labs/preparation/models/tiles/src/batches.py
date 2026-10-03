"""Keep complete frozen parent groups in each native transformation Run."""

from collections import defaultdict


def grouped_batches(members, pilot, maximum=16):
    images = {m["sample_id"]: m for m in members if m["role"] == "image"}
    eligible = {n: m for n, m in images.items() if m["split"] in {"train", "validation"}}
    groups = defaultdict(list)
    for name, member in eligible.items():
        groups[member["group_id"]].append(name)
    pilot = list(pilot)
    if not pilot or len(pilot) != len(set(pilot)) or len(pilot) > maximum or not set(pilot) <= set(eligible):
        raise ValueError("Pilot must contain distinct eligible originals within the batch bound")
    selected_groups = {eligible[n]["group_id"] for n in pilot}
    if any(set(groups[g]) - set(pilot) for g in selected_groups):
        raise ValueError("Pilot must retain every original in its selected parent groups")
    remaining = [(g, sorted(names)) for g, names in groups.items() if g not in selected_groups]
    if any(len(names) > maximum for _, names in remaining):
        raise ValueError("A parent group exceeds the transformation batch bound")
    bins = []
    for _, names in sorted(remaining, key=lambda item: (-len(item[1]), item[0])):
        target = next((batch for batch in bins if len(batch) + len(names) <= maximum), None)
        if target is None:
            bins.append(list(names))
        else:
            target.extend(names)
    return [pilot] + [sorted(batch) for batch in bins]
