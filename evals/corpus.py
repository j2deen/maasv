"""Synthetic eval corpus: a small team/project world plus distractors.

Every memory, entity, relationship, and QA pair is static and deterministic.
QA pairs are tagged by retrieval mechanism so per-type metrics show which
signal a change helped or hurt:

- keyword:    exact terms present, BM25 should nail it
- paraphrase: gold uses different words than the query (synonym-bridged),
              vector signal should carry it
- graph_1hop: answering requires one entity-graph hop from a query entity
- graph_2hop: requires two hops — 1-hop expansion can't reach the gold subject
"""

from dataclasses import dataclass, field


@dataclass
class Memory:
    key: str  # stable key used by QA gold references
    content: str
    category: str
    subject: str | None = None


@dataclass
class QA:
    question: str
    gold_keys: list[str]  # any of these memories counts as a correct retrieval
    qa_type: str  # keyword | paraphrase | graph_1hop | graph_2hop


@dataclass
class Corpus:
    memories: list[Memory] = field(default_factory=list)
    entities: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)  # (subj, pred, obj)
    qas: list[QA] = field(default_factory=list)


# Distractor templates: plausible workplace noise with deliberate vocabulary
# overlap (team, project, leads, depends, review, launch, database...) so
# distractors compete with gold memories in vector and BM25 space. No template
# mentions a graph entity — distractors add noise, never alternate answers.
_DISTRACTOR_TEMPLATES = [
    "Note {i}: the team leads a project review on level {i} every other week",
    "Reminder {i}: the deploy checklist depends on sign-off from the release team",
    "Update {i}: database maintenance is scheduled for the {i}th at midnight",
    "Item {i}: the platform launch retrospective notes are in the shared drive",
    "Log {i}: the engineer on call rotated after the incident review",
    "Memo {i}: office desk {i} is reserved for the visiting project manager",
    "Note {i}: the quarterly security training must be completed by all teams",
    "Ticket {i}: the build pipeline flaked twice and was retried by the team",
    "Update {i}: the vendor contract for tooling renews in month {i}",
    "Item {i}: meeting room {i} has a broken display awaiting repair",
    "Log {i}: the newsletter goes out after the marketing review approves it",
    "Memo {i}: expense reports are due before the end of the month",
    "Note {i}: the intern program review happens each summer with the leads",
    "Ticket {i}: printer {i} jams when the paper tray is overfilled",
]


def _distractors(count: int) -> list[Memory]:
    """Deterministic templated noise; category rotates to spread across tiers."""
    categories = ["history", "project", "person", "learning"]
    out = []
    for i in range(count):
        template = _DISTRACTOR_TEMPLATES[i % len(_DISTRACTOR_TEMPLATES)]
        out.append(Memory(
            f"noise{i}",
            template.format(i=i),
            categories[i % len(categories)],
        ))
    return out


def build_corpus(distractor_count: int = 140) -> Corpus:
    memories = [
        # --- Atlas cluster ---
        Memory("priya_leads_atlas", "Priya Sharma leads the Atlas project", "project", "Priya"),
        Memory("atlas_rust", "Atlas is written in Rust", "project", "Atlas"),
        Memory("atlas_postgres", "Atlas depends on the Postgres database for storage", "project", "Atlas"),
        Memory("atlas_deadline", "Atlas has a hard security-audit deadline in November", "project", "Atlas"),
        # --- Beacon cluster ---
        Memory("marcus_beacon", "Marcus Chen works on the Beacon project", "project", "Marcus"),
        Memory("beacon_kotlin", "Beacon is a mobile app built with Kotlin", "project", "Beacon"),
        Memory("beacon_slip", "The Q3 launch of Beacon slipped two weeks because of app store review", "history", "Beacon"),
        Memory("marcus_toronto", "Marcus is based in Toronto", "person", "Marcus"),
        # --- Casper cluster ---
        Memory("elena_platform", "Elena Rodriguez manages the data platform team", "person", "Elena"),
        Memory("casper_owner", "The data platform team owns the Casper pipeline", "project", "Casper"),
        Memory("casper_nightly", "Casper ingests billing events nightly", "project", "Casper"),
        Memory("casper_airflow", "Elena's team migrated Casper from cron to Airflow in May", "history", "Casper"),
        # --- Helix cluster ---
        Memory("dana_helix", "Dana Osei runs the Helix training pipeline", "project", "Dana"),
        Memory("helix_k8s", "Helix runs on Kubernetes in the eu-west cluster", "project", "Helix"),
        Memory("helix_weekly", "Helix retrains ranking models every Sunday night", "project", "Helix"),
        # --- Quill cluster ---
        Memory("sam_quill", "Sam Porter created the Quill documentation generator", "project", "Sam"),
        Memory("quill_ts", "Quill is written in TypeScript", "project", "Quill"),
        Memory("quill_cf", "Quill is hosted on Cloudflare Workers", "project", "Quill"),
        # --- Preferences / person facts ---
        Memory("priya_async", "Priya prefers async communication over meetings", "preference", "Priya"),
        Memory("elena_espresso", "Elena drinks a double espresso every morning before standup", "person", "Elena"),
        Memory("marcus_keyboard", "Marcus uses a split ergonomic keyboard", "preference", "Marcus"),
        # --- Distractors: plausible, vocabulary-overlapping noise ---
        Memory("d1", "The Orion project was cancelled last year", "history", "Orion"),
        Memory("d2", "Zephyr is a prototype weather dashboard", "project", "Zephyr"),
        Memory("d3", "The design team prefers Figma over Sketch", "preference", None),
        Memory("d4", "Quarterly planning happens the first week of each quarter", "history", None),
        Memory("d5", "The office espresso machine was repaired on Tuesday", "history", None),
        Memory("d6", "Sofia joined the mobile team as an intern", "person", "Sofia"),
        Memory("d7", "The billing service was rewritten in Go two years ago", "history", None),
        Memory("d8", "Toronto office lease renews in January", "history", None),
        Memory("d9", "The security team runs a phishing drill every quarter", "history", None),
        Memory("d10", "Postgres 16 upgrade is planned for the staging cluster", "project", None),
        Memory("d11", "The app store developer account needs its certificate renewed", "history", None),
        Memory("d12", "Rust meetup happens downtown on the last Thursday of the month", "history", None),
        Memory("d13", "Kotlin coroutines caused a subtle bug in the notifications service", "history", None),
        Memory("d14", "The Airflow web UI is behind the VPN", "history", None),
        Memory("d15", "Meetings on Friday afternoons are discouraged", "preference", None),
    ]

    entities = [
        ("Priya", "person"),
        ("Marcus", "person"),
        ("Elena", "person"),
        ("Sofia", "person"),
        ("Dana", "person"),
        ("Sam", "person"),
        ("Atlas", "project"),
        ("Beacon", "project"),
        ("Casper", "project"),
        ("Zephyr", "project"),
        ("Helix", "project"),
        ("Quill", "project"),
        ("Rust", "technology"),
        ("Kotlin", "technology"),
        ("Postgres", "technology"),
        ("Airflow", "technology"),
        ("Kubernetes", "technology"),
        ("TypeScript", "technology"),
        ("Cloudflare", "technology"),
        ("Toronto", "place"),
    ]

    relationships = [
        ("Priya", "manages", "Atlas"),
        ("Atlas", "written_in", "Rust"),
        ("Atlas", "depends_on", "Postgres"),
        ("Marcus", "works_on", "Beacon"),
        ("Beacon", "built_with", "Kotlin"),
        ("Marcus", "lives_in", "Toronto"),
        ("Elena", "manages", "Casper"),
        ("Casper", "runs_on", "Airflow"),
        ("Sofia", "works_on", "Beacon"),
        ("Dana", "manages", "Helix"),
        ("Helix", "runs_on", "Kubernetes"),
        ("Sam", "created", "Quill"),
        ("Quill", "written_in", "TypeScript"),
        ("Quill", "hosted_on", "Cloudflare"),
    ]

    qas = [
        # Keyword: exact terms in gold memory
        QA("Who leads the Atlas project?", ["priya_leads_atlas"], "keyword"),
        QA("When does Casper ingest billing events?", ["casper_nightly"], "keyword"),
        QA("Why did the Beacon launch slip?", ["beacon_slip"], "keyword"),
        QA("What deadline does Atlas have in November?", ["atlas_deadline"], "keyword"),
        # Paraphrase: synonym-bridged, low keyword overlap with gold
        QA("What programming language is Atlas implemented in?", ["atlas_rust"], "paraphrase"),
        QA("Which developer is located in Toronto?", ["marcus_toronto"], "paraphrase"),
        QA("How does Priya like to communicate?", ["priya_async"], "paraphrase"),
        QA("Who heads the data platform team?", ["elena_platform"], "paraphrase"),
        # Graph 1-hop: query names an entity one hop from the gold subject
        QA("Who works on the project written in Rust?", ["priya_leads_atlas", "atlas_rust"], "graph_1hop"),
        QA("Which project does the Toronto-based engineer work on?", ["marcus_beacon"], "graph_1hop"),
        QA("What pipeline does Elena's team look after?", ["casper_owner", "elena_platform"], "graph_1hop"),
        QA("What schedule does Helix retrain on?", ["helix_weekly"], "keyword"),
        QA("Which developer built the docs generator?", ["sam_quill"], "paraphrase"),
        QA("Where is the Helix pipeline deployed?", ["helix_k8s"], "graph_1hop"),
        # Graph 2-hop: gold subject is two hops from the only entity in the query
        QA("Who leads the project that depends on Postgres?", ["priya_leads_atlas"], "graph_2hop"),
        QA("Which person's project is built with Kotlin?", ["marcus_beacon"], "graph_2hop"),
        QA("Who manages the pipeline that runs on Airflow?", ["elena_platform", "casper_owner"], "graph_2hop"),
        QA("Who runs the thing deployed on Kubernetes?", ["dana_helix"], "graph_2hop"),
        QA("Who created the tool written in TypeScript?", ["sam_quill"], "graph_2hop"),
    ]

    memories.extend(_distractors(distractor_count))

    return Corpus(memories=memories, entities=entities, relationships=relationships, qas=qas)
