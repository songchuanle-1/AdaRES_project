"""Deterministic train-only grouping with annotation identity preservation."""

from collections import Counter, defaultdict


def target_ids(row):
    return tuple(sorted(int(v) for v in row.get("object_ids", [row.get("object_id")])))


def identity(row):
    return str(row["scene_id"]), target_ids(row), str(row["ann_id"])


def paired_chunks(chunks):
    grouped, sizes = defaultdict(list), defaultdict(list)
    for chunk in chunks:
        scans = {str(row["scene_id"]) for row in chunk}
        if len(scans) != 1:
            raise ValueError("Pairing requires a single scan per chunk")
        scan = next(iter(scans))
        grouped[scan].extend(chunk)
        sizes[scan].append(len(chunk))
    result = []
    for scan, rows in grouped.items():
        targets = defaultdict(list)
        for row in rows:
            targets[target_ids(row)].append(row)
        units, leftovers = [], []
        for _, group in sorted(targets.items()):
            available = sorted(group, key=lambda r: str(r["ann_id"]))
            while available:
                left = available.pop(0)
                peer = next((j for j, right in enumerate(available)
                             if tuple(right["token"]) != tuple(left["token"])), None)
                if peer is None:
                    leftovers.append(left)
                else:
                    units.extend([left, available.pop(peer)])
        ordered = units + leftovers
        if Counter(map(identity, rows)) != Counter(map(identity, ordered)):
            raise ValueError("Pairing changed annotations")
        start = 0
        for size in sizes[scan]:
            result.append(ordered[start:start+size])
            start += size
    return result


def encoded_pairs(object_idss, tokens, scenes_len):
    result, offset = [], 0
    for count in scenes_len:
        count = int(count)
        unused = list(range(offset, offset+count))
        pairs = []
        while unused:
            left = unused.pop(0)
            peer = next((j for j, right in enumerate(unused)
                         if tuple(sorted(object_idss[left])) ==
                         tuple(sorted(object_idss[right]))
                         and not bool((tokens[left] == tokens[right]).all())), None)
            if peer is not None:
                pairs.append((left, unused.pop(peer)))
        result.append(pairs)
        offset += count
    if offset != len(object_idss) or offset != len(tokens):
        raise ValueError("Pairing expression count mismatch")
    return result
