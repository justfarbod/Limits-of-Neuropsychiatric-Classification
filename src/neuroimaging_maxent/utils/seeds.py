import hashlib


def stable_seed(*identifiers: object, base: int = 42) -> int:
    digest = hashlib.sha256("|".join(map(str, (base, *identifiers))).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF
