"""Synthetic incident generation.

Everything here is synthetic data generated from code: no proprietary
system names, customer data, or real log content is used anywhere in this
project. Categories and on-call tracks are generic labels that any data
engineering organization would recognize (database capacity, DAG
orchestration, Kubernetes infrastructure, data quality, network
connectivity).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

CATEGORIES: List[str] = [
    "db_capacity",
    "dag_orchestration",
    "k8s_infra",
    "data_quality",
    "network_connectivity",
]

SEVERITIES: List[str] = ["P1", "P2", "P3"]

TIME_BUCKETS: List[str] = ["business_hours", "evening", "overnight", "weekend"]

# A fixed, cyclic "confusor" mapping: the track that actually resolves an
# incident fastest is not always the one its surface category suggests.
# This is what makes the routing problem non-trivial even with a perfect
# category classifier: the bandit still has to learn, from reward
# feedback, which arm truly performs best for each observed context.
CONFUSOR_TRACK: Dict[str, str] = {
    "db_capacity": "dag_orchestration",
    "dag_orchestration": "k8s_infra",
    "k8s_infra": "network_connectivity",
    "data_quality": "dag_orchestration",
    "network_connectivity": "db_capacity",
}

CONFUSOR_PROBABILITY = 0.25

# Keyword pools used both to render incident text and to drive the
# deterministic rule-based classifier baseline in keyword_classifier.py.
KEYWORD_POOL: Dict[str, List[str]] = {
    "db_capacity": [
        "tablespace usage above threshold",
        "disk free space critically low on the database volume",
        "autoextend disabled on data file",
        "connection pool near max_connections limit",
        "archive log destination filling up",
        "storage utilization trending toward capacity ceiling",
    ],
    "dag_orchestration": [
        "task instance stuck in queued state",
        "DAG run missed its scheduling window",
        "upstream sensor timed out waiting for partition",
        "scheduler heartbeat lagging behind",
        "retry storm on a downstream task group",
        "backfill run overlapping with the scheduled run",
    ],
    "k8s_infra": [
        "pod stuck in CrashLoopBackOff",
        "node pool autoscaler failed to add capacity",
        "pod evicted due to memory pressure",
        "readiness probe failing after deploy",
        "persistent volume claim stuck in pending",
        "container OOMKilled repeatedly",
    ],
    "data_quality": [
        "null rate spike in a required column",
        "row count dropped below the expected watermark",
        "schema change detected without a version bump",
        "duplicate primary keys found in the landing table",
        "checksum mismatch between source and target",
        "freshness SLA breached for a gold table",
    ],
    "network_connectivity": [
        "intermittent DNS resolution failures",
        "TLS handshake timeouts to the upstream endpoint",
        "packet loss detected between availability zones",
        "load balancer returning 503 for a subset of requests",
        "VPN tunnel flapping between regions",
        "connection reset by peer on the egress proxy",
    ],
}

# Distractor phrases are drawn from a *different* category than the true
# one, to make the text genuinely ambiguous some of the time.
DISTRACTOR_PROBABILITY = 0.35


@dataclass
class Incident:
    incident_id: str
    true_category: str
    optimal_track: str
    severity: str
    time_bucket: str
    title: str
    description: str


def _render_text(rng: random.Random, true_category: str) -> (str, str):
    other_categories = [c for c in CATEGORIES if c != true_category]
    primary_phrases = rng.sample(
        KEYWORD_POOL[true_category], k=rng.randint(2, 3)
    )
    phrases = list(primary_phrases)
    if rng.random() < DISTRACTOR_PROBABILITY:
        distractor_category = rng.choice(other_categories)
        phrases.append(rng.choice(KEYWORD_POOL[distractor_category]))
    rng.shuffle(phrases)
    title = phrases[0][:1].upper() + phrases[0][1:]
    description = ". ".join(p[:1].upper() + p[1:] for p in phrases) + "."
    return title, description


def sample_optimal_track(rng: random.Random, true_category: str) -> str:
    if rng.random() < CONFUSOR_PROBABILITY:
        return CONFUSOR_TRACK[true_category]
    return true_category


def generate_incident(rng: random.Random, incident_id: str) -> Incident:
    true_category = rng.choice(CATEGORIES)
    optimal_track = sample_optimal_track(rng, true_category)
    severity = rng.choice(SEVERITIES)
    time_bucket = rng.choice(TIME_BUCKETS)
    title, description = _render_text(rng, true_category)
    return Incident(
        incident_id=incident_id,
        true_category=true_category,
        optimal_track=optimal_track,
        severity=severity,
        time_bucket=time_bucket,
        title=title,
        description=description,
    )


def generate_eval_dataset(n_per_category: int, seed: int) -> List[Incident]:
    """Generate a balanced evaluation set, used for the offline classifier
    comparison between the LLM zero-shot classifier and the deterministic
    keyword matcher. Balanced by design so raw accuracy is directly
    comparable to the 1/5 chance baseline."""
    rng = random.Random(seed)
    incidents: List[Incident] = []
    counter = 0
    for category in CATEGORIES:
        made = 0
        while made < n_per_category:
            candidate_rng = random.Random(seed * 100_003 + counter)
            true_category = category
            optimal_track = sample_optimal_track(candidate_rng, true_category)
            severity = candidate_rng.choice(SEVERITIES)
            time_bucket = candidate_rng.choice(TIME_BUCKETS)
            title, description = _render_text(candidate_rng, true_category)
            incidents.append(
                Incident(
                    incident_id=f"EVAL-{counter:04d}",
                    true_category=true_category,
                    optimal_track=optimal_track,
                    severity=severity,
                    time_bucket=time_bucket,
                    title=title,
                    description=description,
                )
            )
            counter += 1
            made += 1
    rng.shuffle(incidents)
    return incidents
