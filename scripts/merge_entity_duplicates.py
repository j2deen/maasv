#!/usr/bin/env python3
"""
maasv Entity Near-Duplicate Detection + Merging

Detects near-duplicate entities using normalization, fuzzy matching (rapidfuzz),
and substring containment. Clusters duplicates with Union-Find, selects keepers,
and merges via store.merge_entity().

Usage (from the doris project directory, with its venv active):
    cd /Users/macmini/Projects/doris
    python /Users/macmini/Projects/maasv/scripts/merge_entity_duplicates.py --dry-run -v
    python /Users/macmini/Projects/maasv/scripts/merge_entity_duplicates.py

Options:
    --dry-run    Show what would be merged without modifying the DB
    -v           Verbose: show detailed matching info
"""

import argparse
import logging
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/Users/macmini/Projects/maasv")
sys.path.insert(0, "/Users/macmini/Projects/doris")
os.chdir("/Users/macmini/Projects/doris")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("merge_entity_duplicates")


# ── Normalization ──────────────────────────────────────────────────────

def normalize_name(canonical_name: str) -> str:
    """
    Normalize a canonical_name for duplicate detection.

    Steps:
    1. Replace hyphens with underscores
    2. Strip trailing "s" if len > 4 (basic depluralization)
    3. Strip domain suffixes (.sh, .dev, .js, .io, .ai, .py)
    4. Strip parenthetical qualifiers: "foo_(bar_baz)" → "foo"
    """
    name = canonical_name.lower().strip()

    # 1. Normalize separators
    name = name.replace("-", "_")

    # 4. Strip parenthetical qualifiers (before other steps to avoid partial matches)
    name = re.sub(r"_?\(.*?\)$", "", name)

    # 3. Strip domain suffixes
    for suffix in (".sh", ".dev", ".js", ".io", ".ai", ".py", ".rs", ".go"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    # 2. Strip trailing "s" for depluralization (only if result is long enough)
    if len(name) > 4 and name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]

    return name


# ── Detection Phases ───────────────────────────────────────────────────

def detect_normalization_pairs(entities: list[dict]) -> list[tuple[str, str, str]]:
    """
    Phase A: Find entities that normalize to the same string.

    Returns list of (id_a, id_b, method) tuples.
    """
    by_normalized = defaultdict(list)
    for ent in entities:
        norm = normalize_name(ent["canonical_name"])
        by_normalized[norm].append(ent)

    pairs = []
    for norm, group in by_normalized.items():
        if len(group) < 2:
            continue
        # Pair all combinations within the group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append((group[i]["id"], group[j]["id"], "normalization"))

    return pairs


def _extract_version(name: str) -> str | None:
    """Extract a version-like pattern from a name (e.g., '4.5' from 'haiku_4.5')."""
    match = re.search(r'[\d]+(?:\.\d+)*', name)
    return match.group() if match else None


def _has_version_mismatch(name_a: str, name_b: str) -> bool:
    """
    Check if two names differ primarily in version number.

    Returns True if both have version patterns and the versions differ,
    while the non-version parts are similar.
    e.g., "haiku_3.5" vs "haiku_4.5" → True
          "claude_code" vs "claude_api" → False (no versions)
    """
    ver_a = _extract_version(name_a)
    ver_b = _extract_version(name_b)

    if not ver_a or not ver_b:
        return False

    if ver_a == ver_b:
        return False  # Same version, probably same thing

    # Strip the version numbers and compare the base names
    base_a = re.sub(r'[\d]+(?:\.\d+)*', '', name_a).strip("_- ")
    base_b = re.sub(r'[\d]+(?:\.\d+)*', '', name_b).strip("_- ")

    # If base names are very similar but versions differ → version mismatch
    from rapidfuzz import fuzz
    return fuzz.ratio(base_a, base_b) >= 80


def _has_platform_prefix_mismatch(name_a: str, name_b: str) -> bool:
    """
    Check if two names differ only in platform prefix.

    e.g., "ios_keychain" vs "macos_keychain" → True (different platforms)
    """
    PLATFORM_PREFIXES = ["ios_", "macos_", "android_", "windows_", "linux_"]

    prefix_a = next((p for p in PLATFORM_PREFIXES if name_a.startswith(p)), None)
    prefix_b = next((p for p in PLATFORM_PREFIXES if name_b.startswith(p)), None)

    if prefix_a and prefix_b and prefix_a != prefix_b:
        base_a = name_a[len(prefix_a):]
        base_b = name_b[len(prefix_b):]
        from rapidfuzz import fuzz
        return fuzz.ratio(base_a, base_b) >= 80

    return False


def detect_fuzzy_pairs(entities: list[dict], paired_ids: set, verbose: bool) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, int]]]:
    """
    Phase B: Character-level fuzzy matching with rapidfuzz.

    Uses fuzz.ratio (strict Levenshtein-based) instead of token_set_ratio
    to avoid false positives where one name's tokens are a subset of another's.

    Guards:
    - Length ratio >= 0.6
    - Minimum name length 6 chars (avoids "sql"/"psql", "http"/"httpx")
    - Version number mismatch detection (avoids "haiku_3.5"/"haiku_4.5")
    - Platform prefix mismatch detection (avoids "ios_X"/"macos_X")

    Returns:
        - auto_pairs: list of (id_a, id_b, method) for score >= 82
        - review_pairs: list of (name_a, name_b, score) for score 70-81
    """
    from rapidfuzz import fuzz

    unpaired = [e for e in entities if e["id"] not in paired_ids]
    auto_pairs = []
    review_pairs = []

    for i in range(len(unpaired)):
        for j in range(i + 1, len(unpaired)):
            a, b = unpaired[i], unpaired[j]

            if a["id"] in paired_ids or b["id"] in paired_ids:
                continue

            name_a = a["canonical_name"].replace("_", " ")
            name_b = b["canonical_name"].replace("_", " ")

            # Minimum length guard: avoid matching very short distinct names
            if len(name_a) < 6 or len(name_b) < 6:
                continue

            # Length ratio guard
            shorter, longer = (name_a, name_b) if len(name_a) <= len(name_b) else (name_b, name_a)
            if len(longer) > 0 and len(shorter) / len(longer) < 0.6:
                continue

            # Version mismatch guard: different version numbers = different entities
            cn_a = a["canonical_name"]
            cn_b = b["canonical_name"]
            if _has_version_mismatch(cn_a, cn_b):
                continue

            # Platform prefix mismatch guard
            if _has_platform_prefix_mismatch(cn_a, cn_b):
                continue

            score = fuzz.ratio(name_a, name_b)

            if score >= 82:
                auto_pairs.append((a["id"], b["id"], f"fuzzy-{score}"))
                paired_ids.add(a["id"])
                paired_ids.add(b["id"])
            elif score >= 70:
                review_pairs.append((a["canonical_name"], b["canonical_name"], score))

    return auto_pairs, review_pairs


def detect_substring_pairs(entities: list[dict], paired_ids: set) -> list[tuple[str, str, str]]:
    """
    Phase C: Qualified substring containment.

    The shorter name is fully contained in the longer, AND the longer name
    looks like a qualified version (e.g., parenthetical, prefix like "ios_",
    or suffix like "_api").

    Tighter than naive substring to avoid merging genuinely distinct entities
    like "swift"/"swiftui", "sqlite"/"sqlite-vec", "react"/"react_flow".
    """
    unpaired = [e for e in entities if e["id"] not in paired_ids]
    pairs = []

    # Known qualifier prefixes/suffixes that indicate "same thing, qualified"
    QUALIFIER_PATTERNS = [
        # Prefixes: platform qualifiers
        "ios_", "macos_", "apple_", "google_", "openai_",
        # Suffixes: role qualifiers
        "_api", "_sdk", "_cli", "_mcp", "_server", "_client",
    ]

    for i in range(len(unpaired)):
        for j in range(i + 1, len(unpaired)):
            a, b = unpaired[i], unpaired[j]
            if a["id"] in paired_ids or b["id"] in paired_ids:
                continue

            name_a = a["canonical_name"].replace("-", "_").lower()
            name_b = b["canonical_name"].replace("-", "_").lower()

            shorter, longer = (name_a, name_b) if len(name_a) <= len(name_b) else (name_b, name_a)

            # Guards
            if len(shorter) < 6:
                continue
            if len(shorter) / len(longer) < 0.5:
                continue

            if shorter not in longer:
                continue

            # Extra guard: the longer name must be the shorter name with a known qualifier,
            # OR the longer name must contain the shorter name plus parenthetical only
            remainder = longer.replace(shorter, "", 1).strip("_").strip()
            is_parenthetical = remainder.startswith("(") and remainder.endswith(")")
            is_known_qualifier = any(
                longer.startswith(prefix + shorter) or longer.endswith(shorter + suffix.lstrip("_"))
                for prefix in ["ios_", "macos_", "apple_", "google_", "openai_"]
                for suffix in []  # checked separately below
            ) or any(
                longer == shorter + suffix
                for suffix in ["_api", "_sdk", "_cli", "_mcp", "_server", "_client", "_tts"]
            ) or any(
                longer == prefix + shorter
                for prefix in ["ios_", "macos_", "apple_", "google_", "openai_"]
            )

            if is_parenthetical or is_known_qualifier:
                pairs.append((a["id"], b["id"], "substring"))
                paired_ids.add(a["id"])
                paired_ids.add(b["id"])

    return pairs


# ── Union-Find Clustering ─────────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self, ids: set) -> dict[str, set]:
        groups = defaultdict(set)
        for x in ids:
            groups[self.find(x)].add(x)
        return {root: members for root, members in groups.items() if len(members) > 1}


# ── Keeper Selection ──────────────────────────────────────────────────

def select_keeper(members: list[dict], rel_counts: dict[str, int]) -> tuple[dict, list[dict]]:
    """
    Select the best entity to keep from a cluster.

    Sort by:
    1. Most relationships (more connected = more canonical)
    2. Shortest canonical_name (more concise = more standard)
    3. Highest access_count
    4. Most recent created_at
    """
    members.sort(key=lambda m: (
        rel_counts.get(m["id"], 0),          # most relationships
        -len(m.get("canonical_name") or ""),  # shortest name (negative for desc)
        m.get("access_count") or 0,           # highest access
        m.get("created_at") or "",            # most recent
    ), reverse=True)

    return members[0], members[1:]


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect and merge near-duplicate entities")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying DB")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed matching info")
    args = parser.parse_args()

    from maasv_bridge import init_maasv
    init_maasv()

    from maasv.core.store import get_db, merge_entity

    db = get_db()

    db_path = db.execute("PRAGMA database_list").fetchone()[2]
    logger.info(f"=== Entity Near-Duplicate Detection ===")
    logger.info(f"Database: {db_path}")

    # Get all entities
    all_entities = [dict(r) for r in db.execute(
        "SELECT id, name, entity_type, canonical_name, metadata, access_count, created_at FROM entities"
    ).fetchall()]

    # Get relationship counts for all entities
    rel_count_rows = db.execute("""
        SELECT entity_id, SUM(cnt) as total FROM (
            SELECT subject_id as entity_id, COUNT(*) as cnt FROM relationships WHERE valid_to IS NULL GROUP BY subject_id
            UNION ALL
            SELECT object_id as entity_id, COUNT(*) as cnt FROM relationships WHERE valid_to IS NULL AND object_id IS NOT NULL GROUP BY object_id
        ) GROUP BY entity_id
    """).fetchall()
    rel_counts = {r["entity_id"]: r["total"] for r in rel_count_rows}

    # Group entities by type
    by_type = defaultdict(list)
    for ent in all_entities:
        by_type[ent["entity_type"]].append(ent)

    entity_by_id = {e["id"]: e for e in all_entities}

    logger.info(f"Total entities: {len(all_entities)}")
    for etype, ents in sorted(by_type.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {etype}: {len(ents)}")

    # Process each entity type
    all_pairs = []              # (id_a, id_b, method)
    all_review = []             # (name_a, name_b, score)
    total_norm = 0
    total_fuzzy = 0
    total_substring = 0

    for entity_type, entities in sorted(by_type.items()):
        if len(entities) < 2:
            continue

        paired_ids = set()

        # Phase A: Normalization
        norm_pairs = detect_normalization_pairs(entities)
        for a, b, _ in norm_pairs:
            paired_ids.add(a)
            paired_ids.add(b)
        total_norm += len(norm_pairs)

        # Phase B: Fuzzy
        fuzzy_pairs, review = detect_fuzzy_pairs(entities, paired_ids, args.verbose)
        total_fuzzy += len(fuzzy_pairs)

        # Phase C: Substring
        sub_pairs = detect_substring_pairs(entities, paired_ids)
        total_substring += len(sub_pairs)

        all_pairs.extend(norm_pairs)
        all_pairs.extend(fuzzy_pairs)
        all_pairs.extend(sub_pairs)
        all_review.extend(review)

    logger.info(f"\nCandidate pairs: {len(all_pairs)} ({total_norm} normalization, {total_fuzzy} fuzzy, {total_substring} substring)")

    if not all_pairs:
        logger.info("No duplicates found!")
        db.close()
        return

    # Build clusters with Union-Find
    uf = UnionFind()
    all_involved_ids = set()
    pair_methods = {}  # (id_a, id_b) → method

    for id_a, id_b, method in all_pairs:
        uf.union(id_a, id_b)
        all_involved_ids.add(id_a)
        all_involved_ids.add(id_b)
        pair_key = tuple(sorted([id_a, id_b]))
        pair_methods[pair_key] = method

    clusters = uf.clusters(all_involved_ids)
    logger.info(f"Clusters: {len(clusters)}")

    # Safety check: skip clusters where a single entity matches too many others
    # (suggests it's too generic, like "api" matching everything)
    safe_clusters = {}
    for root, member_ids in clusters.items():
        if len(member_ids) > 5:
            names = [entity_by_id[mid]["canonical_name"] for mid in member_ids if mid in entity_by_id]
            logger.warning(f"  SKIPPING large cluster ({len(member_ids)} members): {names[:5]}...")
            continue
        safe_clusters[root] = member_ids

    # Display clusters and select keepers
    logger.info(f"\n=== Merge Plan ===\n")
    merge_plan = []  # (keeper_id, [duplicate_ids])

    for i, (root, member_ids) in enumerate(sorted(safe_clusters.items()), 1):
        members = [entity_by_id[mid] for mid in member_ids if mid in entity_by_id]
        if len(members) < 2:
            continue

        keeper, duplicates = select_keeper(members, rel_counts)

        # Find the method used for each duplicate
        dup_info = []
        for dup in duplicates:
            pair_key = tuple(sorted([keeper["id"], dup["id"]]))
            method = pair_methods.get(pair_key, "transitive")
            dup_info.append((dup, method))

        keeper_rels = rel_counts.get(keeper["id"], 0)
        names = [m["canonical_name"] for m in members]
        logger.info(f"Cluster {i}: {names} → keeper: {keeper['canonical_name']} ({keeper_rels} rels)")

        for dup, method in dup_info:
            dup_rels = rel_counts.get(dup["id"], 0)
            logger.info(f"  - Merge: {dup['canonical_name']} ({dup_rels} rels, method={method})")

        merge_plan.append((keeper["id"], [d["id"] for d, _ in dup_info]))

    # Flagged for manual review
    if all_review:
        logger.info(f"\n=== Flagged for Manual Review (score 70-84) ===")
        for name_a, name_b, score in sorted(all_review, key=lambda x: -x[2]):
            logger.info(f"  - \"{name_a}\" vs \"{name_b}\" (score={score}) — REVIEW")

    # Summary
    total_to_merge = sum(len(dups) for _, dups in merge_plan)
    total_rels_affected = sum(
        sum(rel_counts.get(did, 0) for did in dups)
        for _, dups in merge_plan
    )

    logger.info(f"\n=== Summary ===")
    action = "Would merge" if args.dry_run else "Will merge"
    logger.info(f"{action}: {total_to_merge} entities into {len(merge_plan)} keepers")
    logger.info(f"Relationships affected: ~{total_rels_affected}")
    logger.info(f"Flagged for review: {len(all_review)} pairs")

    db.close()

    # Execute merges
    if not args.dry_run and merge_plan:
        # Backup first
        import maasv
        config = maasv.get_config()
        backup_dir = config.backup_dir / "entity_merge" if config.backup_dir else None

        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"pre_entity_merge_{timestamp}.db"
            shutil.copy2(config.db_path, backup_path)
            logger.info(f"\nBackup created: {backup_path}")

        logger.info(f"\nExecuting merges...")
        total_stats = {"relationships_updated": 0, "entities_deleted": 0, "rel_dupes_removed": 0}

        for keeper_id, dup_ids in merge_plan:
            keeper_name = entity_by_id.get(keeper_id, {}).get("canonical_name", keeper_id)
            try:
                stats = merge_entity(keeper_id, dup_ids)
                total_stats["relationships_updated"] += stats["relationships_updated"]
                total_stats["entities_deleted"] += stats["entities_deleted"]
                total_stats["rel_dupes_removed"] += stats["rel_dupes_removed"]

                if args.verbose:
                    logger.info(f"  Merged into {keeper_name}: {stats}")
            except Exception as e:
                logger.error(f"  FAILED merging into {keeper_name}: {e}", exc_info=True)

        logger.info(f"\n=== Merge Complete ===")
        logger.info(f"Entities deleted: {total_stats['entities_deleted']}")
        logger.info(f"Relationships reassigned: {total_stats['relationships_updated']}")
        logger.info(f"Duplicate relationships removed: {total_stats['rel_dupes_removed']}")

        # Final entity count
        db2 = get_db()
        final_count = db2.execute("SELECT count(*) FROM entities").fetchone()[0]
        final_rels = db2.execute("SELECT count(*) FROM relationships WHERE valid_to IS NULL").fetchone()[0]
        logger.info(f"Final entities: {final_count}")
        logger.info(f"Final active relationships: {final_rels}")

        # Orphan check
        orphans = db2.execute("""
            SELECT COUNT(*) FROM relationships r
            WHERE r.valid_to IS NULL
            AND (r.subject_id NOT IN (SELECT id FROM entities)
                 OR (r.object_id IS NOT NULL AND r.object_id NOT IN (SELECT id FROM entities)))
        """).fetchone()[0]
        if orphans:
            logger.error(f"ORPHANED RELATIONSHIPS: {orphans} — investigate!")
        else:
            logger.info("Relationship integrity: OK (no orphans)")

        db2.close()


if __name__ == "__main__":
    main()
