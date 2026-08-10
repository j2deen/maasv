"""
Personalized PageRank over the knowledge graph (HippoRAG-style).

Multi-hop graph retrieval: instead of expanding one hop from query entities,
run PPR with restart mass pinned to the query's seed entities. Score flows
along relationship edges (confidence-weighted, undirected), so an entity two
or three hops out still earns retrieval weight — proportionally less, which
is exactly what 1-hop expansion can't express.

Pure Python, no dependencies. Works on a bounded subgraph (BFS neighborhood
of the seeds) so cost stays flat regardless of total graph size.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

MAX_HOPS = 3


def build_subgraph(
    db,
    seed_ids: list[str],
    max_nodes: int = 500,
) -> tuple[set[str], list[tuple[str, str, float]]]:
    """
    Collect the BFS neighborhood of the seeds (up to MAX_HOPS, bounded by
    max_nodes) over active entity-to-entity relationships.

    Returns (node_ids, edges) where edges are (subject_id, object_id, weight)
    with weight = relationship confidence.
    """
    nodes: set[str] = set(seed_ids)
    frontier: set[str] = set(seed_ids)
    edges: list[tuple[str, str, float]] = []
    seen_edges: set[tuple[str, str]] = set()

    for _hop in range(MAX_HOPS):
        if not frontier or len(nodes) >= max_nodes:
            break
        placeholders = ",".join("?" * len(frontier))
        params = list(frontier) * 2
        try:
            rows = db.execute(f"""
                SELECT subject_id, object_id, confidence
                FROM relationships
                WHERE (subject_id IN ({placeholders}) OR object_id IN ({placeholders}))
                AND valid_to IS NULL
                AND object_id IS NOT NULL
            """, params).fetchall()
        except Exception:
            logger.debug("PPR subgraph query failed", exc_info=True)
            break

        next_frontier: set[str] = set()
        for row in rows:
            a, b = row["subject_id"], row["object_id"]
            key = (a, b) if a <= b else (b, a)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            weight = row["confidence"] if row["confidence"] is not None else 1.0
            edges.append((a, b, max(0.0, min(1.0, weight))))
            for node in (a, b):
                if node not in nodes:
                    next_frontier.add(node)

        room = max_nodes - len(nodes)
        if room <= 0:
            break
        next_frontier = set(sorted(next_frontier)[:room])  # deterministic cap
        nodes |= next_frontier
        frontier = next_frontier

    return nodes, edges


def personalized_pagerank(
    db,
    seed_ids: list[str],
    alpha: float = 0.5,
    iterations: int = 20,
    max_nodes: int = 500,
) -> dict[str, float]:
    """
    PPR scores for the seed neighborhood. alpha is the walk-continuation
    probability (1 - alpha restarts at the seeds); low alpha keeps mass near
    the seeds, which suits retrieval better than web-style 0.85.

    Returns {entity_id: score}, scores summing to ~1 over the subgraph.
    """
    if not seed_ids:
        return {}

    nodes, edges = build_subgraph(db, seed_ids, max_nodes=max_nodes)
    if not nodes:
        return {}

    # Undirected weighted adjacency
    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for a, b, w in edges:
        if w <= 0.0:
            continue
        neighbors[a].append((b, w))
        neighbors[b].append((a, w))

    weight_sum = {n: sum(w for _, w in neighbors[n]) for n in nodes}

    seeds_in_graph = [s for s in seed_ids if s in nodes]
    if not seeds_in_graph:
        return {}
    restart = 1.0 / len(seeds_in_graph)

    scores = {n: 0.0 for n in nodes}
    for s in seeds_in_graph:
        scores[s] = restart

    for _ in range(iterations):
        nxt = {n: 0.0 for n in nodes}
        dangling_mass = 0.0
        for n, score in scores.items():
            if score == 0.0:
                continue
            out = weight_sum.get(n, 0.0)
            if out <= 0.0:
                dangling_mass += alpha * score
                continue
            share = alpha * score / out
            for m, w in neighbors[n]:
                nxt[m] += share * w
        # Restart mass + dangling mass returns to the seeds
        returned = (1.0 - alpha) + dangling_mass
        per_seed = returned / len(seeds_in_graph)
        for s in seeds_in_graph:
            nxt[s] += per_seed
        # Convergence check (L1)
        delta = sum(abs(nxt[n] - scores[n]) for n in nodes)
        scores = nxt
        if delta < 1e-6:
            break

    return scores
