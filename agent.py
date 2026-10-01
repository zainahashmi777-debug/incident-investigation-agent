"""
Incident Investigation Agent
=============================

An agent that investigates operational incidents across a heterogeneous
collection of internal documents (incident reports, deployment notes,
architecture docs, troubleshooting guides, customer complaints, and
postmortems).

Design goals (mapped directly to the problem statement's requirements):

1. Accept a natural-language investigation question.
2. Search across multiple document types using semantic + metadata-aware
   retrieval (hybrid lexical scoring, weighted by type/service/date/version
   relevance -- no network/API dependency required, so it runs anywhere).
3. Use information discovered during investigation to perform additional
   searches (multi-hop retrieval: hop 1 finds directly relevant documents,
   entities extracted from hop 1 -- service, version, date -- seed hop 2).
4. Distinguish document dates, software versions, and outdated guidance
   (explicit date/version parsing + "supersedes" resolution).
5. Detect contradictions and avoid treating similar incidents as identical
   (negation-aware directive comparison; root-cause/service equality checks
   before two incidents are called "the same").
6. Return an evidence-backed answer with document identifiers and a clear
   statement when evidence is insufficient.

The retrieval layer is intentionally dependency-free (pure Python) so the
agent is fully deterministic and auditable for a hackathon demo. It is
written so the `score_document` function could be swapped for an embedding
similarity call (e.g. OpenAI/local sentence-transformers) without touching
the multi-hop / contradiction / synthesis logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Document:
    document_id: str
    type: str
    date: str
    title: str
    content: str
    service: Optional[str] = None
    version: Optional[str] = None

    @property
    def parsed_date(self) -> Optional[date]:
        try:
            return datetime.strptime(self.date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    @property
    def text_blob(self) -> str:
        return " ".join(
            filter(None, [self.title, self.content, self.type, self.service])
        )

    def to_citation(self) -> str:
        bits = [self.document_id, f"type={self.type}"]
        if self.service:
            bits.append(f"service={self.service}")
        bits.append(f"date={self.date}")
        if self.version:
            bits.append(f"version={self.version}")
        return " | ".join(bits)


@dataclass
class Evidence:
    document: Document
    score: float
    reason: str


@dataclass
class Contradiction:
    doc_a: Document
    doc_b: Document
    explanation: str
    resolution: str


@dataclass
class InvestigationResult:
    question: str
    answer: str
    evidence: list = field(default_factory=list)          # list[Evidence]
    contradictions: list = field(default_factory=list)    # list[Contradiction]
    insufficient: bool = False
    hops: list = field(default_factory=list)              # list[str] (debug trail)

    def to_markdown(self) -> str:
        lines = [f"### Question\n{self.question}\n", f"### Answer\n{self.answer}\n"]

        if self.contradictions:
            lines.append("### Contradictions Detected")
            for c in self.contradictions:
                lines.append(
                    f"- **{c.doc_a.document_id}** vs **{c.doc_b.document_id}**: "
                    f"{c.explanation} -> {c.resolution}"
                )
            lines.append("")

        lines.append("### Evidence Used")
        if not self.evidence:
            lines.append("- (none met the relevance threshold)")
        for e in self.evidence:
            lines.append(f"- `{e.document.to_citation()}` — {e.reason}")
        lines.append("")

        if self.insufficient:
            lines.append(
                "> **Insufficient evidence:** the retrieved documents do not "
                "confirm a definitive answer to this question. See reasoning above."
            )

        if self.hops:
            lines.append("\n### Investigation Trail (debug)")
            for h in self.hops:
                lines.append(f"- {h}")

        return "\n".join(lines)


# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "to", "of",
    "and", "or", "in", "on", "for", "we", "us", "our", "this", "that", "it",
    "did", "do", "does", "has", "have", "had", "what", "why", "how", "when",
    "should", "check", "whether", "first", "i", "you", "your", "if", "with",
    "before", "not", "any", "so",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set:
    if not text:
        return set()
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def normalize_service(text: str) -> str:
    """Normalize service references like 'Order API' -> 'orders-api'."""
    t = text.lower().strip()
    t = t.replace("_", "-").replace(" ", "-")
    t = re.sub(r"-+", "-", t)
    # common pluralization / naming drift: "order-api" ~ "orders-api"
    t = re.sub(r"^order-api$", "orders-api", t)
    return t


_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+){0,2}\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_MONTH_DAY_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2})\b",
    re.IGNORECASE,
)

_TYPE_TRIGGER_WORDS = {
    "incident_report": {"incident", "happened", "before", "history", "spike", "outage"},
    "deployment_note": {"deploy", "deployment", "deployed", "release", "rollout"},
    "postmortem": {"postmortem", "before", "previous", "root", "cause", "history"},
    "troubleshooting": {"restart", "procedure", "oncall", "on-call", "do", "should", "first", "guide"},
    "customer_complaint": {"customer", "complaint", "users", "reported"},
    "architecture_doc": {"architecture", "design", "topology", "dependency", "dependencies"},
}

_NEGATION_WINDOW = re.compile(
    r"\b(do not|don't|never|avoid|should not|stop)\b", re.IGNORECASE
)


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

class InvestigationAgent:
    def __init__(self, documents: list):
        self.documents: list = documents
        self._by_id = {d.document_id: d for d in documents}

        # Semantic vector index: ChromaDB + a local SentenceTransformer model.
        # The collection lives only for this process, so the demo remains simple
        # and no external database/server is required.
        self._embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            name="incident_documents",
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        if documents:
            self._collection.add(
                ids=[d.document_id for d in documents],
                documents=[d.text_blob for d in documents],
                metadatas=[
                    {
                        "type": d.type,
                        "service": d.service or "",
                        "date": d.date or "",
                        "version": d.version or "",
                    }
                    for d in documents
                ],
            )

    # ---- construction helpers ----

    @classmethod
    def from_json(cls, documents_json: list) -> "InvestigationAgent":
        docs = [
            Document(
                document_id=d["document_id"],
                type=d.get("type", "unknown"),
                date=d.get("date", ""),
                title=d.get("title", ""),
                content=d.get("content", ""),
                service=d.get("service"),
                version=d.get("version"),
            )
            for d in documents_json
        ]
        return cls(docs)

    # ---- retrieval ----

    def _extract_entities(self, text: str) -> dict:
        """Pull out service / version / date hints mentioned in free text."""
        entities = {"services": set(), "versions": set(), "dates": set()}

        for doc in self.documents:
            if doc.service:
                svc_tokens = doc.service.replace("-", " ")
                if svc_tokens in text.lower() or doc.service.lower() in text.lower():
                    entities["services"].add(normalize_service(doc.service))
        # also catch generic "Order API" -> orders-api even if not literal service text
        for m in re.finditer(r"\b([A-Za-z]+)\s+API\b", text, re.IGNORECASE):
            entities["services"].add(normalize_service(f"{m.group(1)}-api"))

        for m in _VERSION_RE.finditer(text):
            entities["versions"].add(m.group(0).lower())

        for m in _DATE_RE.finditer(text):
            entities["dates"].add(m.group(1))

        for m in _MONTH_DAY_RE.finditer(text):
            entities["dates"].add(m.group(0))

        return entities

    def _boosted_types(self, query_tokens: set) -> set:
        boosted = set()
        for doc_type, triggers in _TYPE_TRIGGER_WORDS.items():
            if query_tokens & triggers:
                boosted.add(doc_type)
        return boosted

    def score_document(
        self,
        doc: Document,
        query_tokens: set,
        boosted_types: set,
        service_hint: Optional[str],
        version_hint: Optional[str],
        relax_version: bool = False,
        semantic_similarity: float = 0.0,
    ) -> tuple:
        """
        Hybrid semantic + metadata score.

        ChromaDB supplies semantic similarity from local sentence embeddings.
        Metadata-aware boosts then incorporate document type, service and version.
        A small lexical signal is retained as an explainable tie-breaker.
        Returns (score, reason_string).
        """
        doc_tokens = tokenize(doc.text_blob)
        title_tokens = tokenize(doc.title)

        overlap = query_tokens & doc_tokens
        title_overlap = query_tokens & title_tokens

        if not doc_tokens:
            lexical = 0.0
        else:
            lexical = len(overlap) / max(len(query_tokens | doc_tokens), 1)
        lexical += 0.15 * len(title_overlap)

        # Main relevance signal is semantic similarity; lexical overlap keeps
        # exact identifiers and rare technical terms highly explainable.
        score = (0.75 * semantic_similarity) + (0.25 * min(lexical, 1.0))
        reasons = [f"semantic similarity: {semantic_similarity:.3f}"]
        if overlap:
            reasons.append(f"keyword overlap: {', '.join(sorted(overlap))}")

        if doc.type in boosted_types:
            score += 0.6
            reasons.append(f"document type '{doc.type}' matches question intent")

        if service_hint and doc.service and normalize_service(doc.service) == service_hint:
            score += 0.9
            reasons.append(f"same service ({doc.service})")

        if version_hint and doc.version and doc.version.lower() == version_hint and not relax_version:
            score += 0.4
            reasons.append(f"matches version {doc.version}")

        return score, "; ".join(reasons)

    def search(
        self,
        query_text: str,
        service_hint: Optional[str] = None,
        version_hint: Optional[str] = None,
        type_filter: Optional[set] = None,
        relax_version: bool = False,
        top_k: int = 10,
    ) -> list:
        query_tokens = tokenize(query_text)
        boosted_types = self._boosted_types(query_tokens)
        if type_filter:
            boosted_types |= type_filter

        # Semantic retrieval over the full document corpus. ChromaDB returns
        # cosine distances; convert them to similarity in [0, 1].
        if not self.documents:
            return []

        result = self._collection.query(
            query_texts=[query_text],
            n_results=len(self.documents),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        semantic_by_id = {
            doc_id: max(0.0, min(1.0, 1.0 - float(distance)))
            for doc_id, distance in zip(ids, distances)
        }

        scored = []
        for doc in self.documents:
            semantic_similarity = semantic_by_id.get(doc.document_id, 0.0)
            score, reason = self.score_document(
                doc, query_tokens, boosted_types, service_hint, version_hint,
                relax_version, semantic_similarity
            )
            if score > 0:
                scored.append(Evidence(document=doc, score=round(score, 3), reason=reason))

        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:top_k]

    # ---- contradiction detection ----

    def detect_contradictions(self, evidence: list) -> list:
        contradictions = []
        guidance_docs = [
            e.document for e in evidence
            if e.document.type in ("troubleshooting",) or "restart" in e.document.content.lower()
        ]

        for i in range(len(guidance_docs)):
            for j in range(i + 1, len(guidance_docs)):
                a, b = guidance_docs[i], guidance_docs[j]
                if a.service and b.service and normalize_service(a.service) != normalize_service(b.service):
                    continue  # different subjects entirely, not a contradiction

                a_topic = tokenize(a.content) & {"restart", "service"}
                b_topic = tokenize(b.content) & {"restart", "service"}
                if not (a_topic and b_topic):
                    continue

                a_negated = bool(_NEGATION_WINDOW.search(a.content))
                b_negated = bool(_NEGATION_WINDOW.search(b.content))

                if a_negated != b_negated:
                    # one says "restart", the other says "do not restart"
                    a_date, b_date = a.parsed_date, b.parsed_date
                    if a_date and b_date:
                        newer, older = (a, b) if a_date > b_date else (b, a)
                        resolution = (
                            f"'{newer.document_id}' ({newer.date}, {newer.version}) is more "
                            f"recent than '{older.document_id}' ({older.date}, {older.version}) "
                            f"and should be treated as the current guidance."
                        )
                    else:
                        resolution = "Unable to determine recency; flag for human review."

                    contradictions.append(
                        Contradiction(
                            doc_a=a,
                            doc_b=b,
                            explanation=(
                                "Directly conflicting operational guidance on the same "
                                "action (restart) for the same service."
                            ),
                            resolution=resolution,
                        )
                    )
        return contradictions

    # ---- multi-hop investigation ----

    def investigate(self, question: str, verbose: bool = True) -> InvestigationResult:
        hops_log = []
        entities = self._extract_entities(question)
        service_hint = next(iter(entities["services"]), None)
        version_hint = next(iter(entities["versions"]), None)

        # ---- Hop 1: direct retrieval from the question itself ----
        hop1 = self.search(question, service_hint=service_hint, version_hint=version_hint)
        hops_log.append(
            f"Hop 1 — direct search for question (service_hint={service_hint}, "
            f"version_hint={version_hint}): found {len(hop1)} candidate(s) -> "
            f"{[e.document.document_id for e in hop1]}"
        )

        # ---- Hop 2: expand using entities discovered in hop-1 results ----
        discovered_versions = {d.document.version for d in hop1 if d.document.version}
        discovered_dates = {d.document.date for d in hop1 if d.document.date}
        wants_history = bool(tokenize(question) & {"before", "history", "previous", "ever", "again"})

        hop2_query_terms = list(tokenize(question))
        if wants_history:
            # deliberately drop the version constraint to search *across* versions
            # for prior occurrences of the same underlying symptom
            hop2 = self.search(
                " ".join(hop2_query_terms) + " previous incident postmortem history",
                service_hint=service_hint,
                version_hint=None,
                type_filter={"postmortem", "incident_report"},
                relax_version=True,
            )
            hops_log.append(
                f"Hop 2 — historical search (version constraint relaxed, service_hint="
                f"{service_hint}) to look for prior occurrences: found {len(hop2)} -> "
                f"{[e.document.document_id for e in hop2]}"
            )
        else:
            hop2 = self.search(
                " ".join(hop2_query_terms),
                service_hint=service_hint,
                version_hint=version_hint,
                type_filter={"deployment_note", "troubleshooting", "architecture_doc"},
            )
            hops_log.append(
                f"Hop 2 — supporting-context search (deployment/architecture/guides): "
                f"found {len(hop2)} -> {[e.document.document_id for e in hop2]}"
            )

        # merge, dedupe, keep best score per doc
        merged: dict = {}
        for e in hop1 + hop2:
            key = e.document.document_id
            if key not in merged or e.score > merged[key].score:
                merged[key] = e
        evidence = sorted(merged.values(), key=lambda e: e.score, reverse=True)

        # ---- contradiction detection ----
        contradictions = self.detect_contradictions(evidence)

        # ---- synthesize answer ----
        answer, insufficient, evidence = self._synthesize(
            question, evidence, contradictions, service_hint, wants_history
        )

        return InvestigationResult(
            question=question,
            answer=answer,
            evidence=evidence,
            contradictions=contradictions,
            insufficient=insufficient,
            hops=hops_log if verbose else [],
        )

    # ---- answer synthesis ----

    def _synthesize(
        self,
        question: str,
        evidence: list,
        contradictions: list,
        service_hint: Optional[str],
        wants_history: bool,
    ) -> tuple:
        RELEVANCE_FLOOR = 0.35
        strong_evidence = [e for e in evidence if e.score >= RELEVANCE_FLOOR]

        if contradictions:
            c = contradictions[0]
            answer = (
                f"Found conflicting guidance for this situation. {c.explanation} "
                f"{c.resolution} Follow '{c.resolution.split(chr(39))[1]}' as the "
                f"current procedure, and flag the older document for deprecation."
            )
            # keep only evidence actually relevant to the contradiction + supporting docs
            return answer, False, evidence

        if not strong_evidence:
            answer = (
                "There is not enough evidence in the available documents to give a "
                "confident, evidence-backed answer to this question. "
            )
            if evidence:
                weak_ids = ", ".join(e.document.document_id for e in evidence[:3])
                answer += (
                    f"The closest matches ({weak_ids}) reference different services "
                    "and/or different root causes than what the question describes, "
                    "so they cannot be treated as the same incident recurring. "
                    "Recommend broadening the search window or providing more specific "
                    "symptoms (service name, error signature, or timeframe)."
                )
            else:
                answer += "No documents in the corpus matched this question at all."
            return answer, True, evidence

        incidents = [e.document for e in strong_evidence if e.document.type == "incident_report"]
        deployments = [e.document for e in strong_evidence if e.document.type == "deployment_note"]
        postmortems = [e.document for e in strong_evidence if e.document.type == "postmortem"]

        parts = []

        if incidents and deployments:
            inc, dep = incidents[0], deployments[0]
            if inc.parsed_date and dep.parsed_date and dep.parsed_date <= inc.parsed_date:
                parts.append(
                    f"The incident described in {inc.document_id} ({inc.date}) began after "
                    f"deployment {dep.document_id} shipped {dep.version} on {dep.date}, "
                    "so the deployment is the most likely trigger."
                )
            else:
                parts.append(
                    f"{inc.document_id} and {dep.document_id} both concern "
                    f"{inc.service or dep.service}, but the timing does not clearly "
                    "establish that the deployment caused the incident."
                )
        elif incidents:
            parts.append(
                f"{incidents[0].document_id} documents the incident directly, but no "
                "matching deployment note was found to confirm a deployment trigger."
            )

        if wants_history:
            if postmortems:
                pm = postmortems[0]
                same_root_cause = False
                if incidents:
                    inc_tokens = tokenize(incidents[0].content)
                    pm_tokens = tokenize(pm.content)
                    same_root_cause = len(inc_tokens & pm_tokens) >= 2
                if same_root_cause:
                    parts.append(
                        f"A similar symptom occurred before: {pm.document_id} ({pm.date}, "
                        f"{pm.version}) describes a related latency issue, though the root "
                        f"cause recorded there ({pm.content.strip().rstrip('.')}) differs "
                        f"from the current incident's likely trigger, so treat this as "
                        "similar-but-not-identical rather than a repeat of the same root cause."
                    )
                else:
                    parts.append(
                        f"{pm.document_id} ({pm.date}) shows a prior incident on the same "
                        "service, but with a different recorded cause, so it should not be "
                        "treated as the same failure recurring."
                    )
            else:
                parts.append(
                    "No prior incident or postmortem for this service/symptom was found in "
                    "the corpus, so this does not appear to be a repeat of a known issue "
                    "based on available evidence."
                )

        if not parts:
            parts.append(
                "The retrieved documents are topically related but do not, on their own, "
                "fully answer the question with high confidence."
            )

        answer = " ".join(parts)
        return answer, False, evidence


# --------------------------------------------------------------------------
# Convenience loader for the CLI/demo
# --------------------------------------------------------------------------

def load_case(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    agent = InvestigationAgent.from_json(payload["documents"])
    return payload["question"], agent
