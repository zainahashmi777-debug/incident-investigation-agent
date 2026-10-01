# Incident Investigation Agent

A working solution to **"The Investigation Nobody Could Answer"**: an agent that
investigates operational incidents by connecting evidence scattered across
incident reports, deployment notes, architecture docs, troubleshooting guides,
customer complaints, and postmortems.

Two deliverables are included:

1. **`agent.py`** — the reference implementation (pure Python, zero external
   dependencies, fully deterministic and offline).
2. **`demo.html`** — a client-side "case file" reimplementation of the same
   logic, for a live, judge-friendly, in-browser demo. Pick one of the four
   sample cases or paste your own `documents.json`.
3. **`requirements.txt`** — ChromaDB + local sentence-transformer dependencies for
   semantic retrieval.

## Semantic + metadata-aware retrieval

The retrieval layer uses **ChromaDB with a local SentenceTransformer
(`all-MiniLM-L6-v2`)** to perform semantic similarity search, then combines
that signal with explainable metadata-aware boosts for document type, service,
and version. A small lexical signal is retained for exact technical terms and
identifiers.

The vector index is created in memory when the agent starts, so no API key or
separate database server is required. The first run may download the local
embedding model; after that, the model can be reused from the local cache.

## Architecture

```mermaid
flowchart TD
    Q[Natural-language question] --> E[Entity extraction<br/>service / version / date]
    E --> H1[Hop 1: direct search<br/>question text + entity hints]
    H1 --> D{Question implies<br/>"has this happened before"?}
    D -- yes --> H2a[Hop 2: historical search<br/>relax version constraint,<br/>boost postmortem/incident_report]
    D -- no --> H2b[Hop 2: supporting-context search<br/>boost deployment_note/troubleshooting/architecture_doc]
    H2a --> M[Merge + de-dupe<br/>keep best score per doc]
    H2b --> M
    M --> C[Contradiction detection<br/>negation-aware directive comparison]
    C --> S[Answer synthesis<br/>evidence floor + recency/version resolution]
    S --> R[Evidence-backed answer<br/>+ document IDs<br/>+ insufficient-evidence flag]
```

## How each core requirement is met

| Requirement | Implementation |
|---|---|
| Accept a natural-language question | `InvestigationAgent.investigate(question)` |
| Search multiple document types, semantic + metadata-aware | ChromaDB semantic similarity from local sentence embeddings + lexical signal + type/service/version metadata boosts |
| Use discovered info to search again | Two-hop retrieval: hop 1 finds direct matches; hop 2 is seeded with entities pulled from hop 1 (or relaxes the version filter when the question is historical) |
| Distinguish dates, versions, outdated guidance | `Document.parsed_date`, explicit `version` field comparisons, and the contradiction resolver picks the more recent document as current guidance |
| Detect contradictions, avoid treating similar incidents as identical | `detect_contradictions()` does negation-aware comparison ("restart" vs "do not restart") on same-service guidance docs; `_synthesize()` explicitly checks root-cause token overlap before calling two incidents "the same" |
| Evidence-backed answer with document IDs + insufficiency statement | Every answer sentence traces back to cited `document_id`s; a relevance floor (`RELEVANCE_FLOOR = 0.35`) triggers an explicit "insufficient evidence" response with reasoning, as demonstrated by Test Input C |

## Running it

```bash
cd incident-investigation-agent
pip install -r requirements.txt
python3 demo.py            # runs all four sample cases
python3 demo.py --case b   # run just the contradictory-guidance case
python3 demo.py --case d   # run the 10-document evidence corpus
python3 demo.py --json data/my_case.json   # bring your own documents.json
```

Each `data/my_case.json` should look like:

```json
{
  "question": "...",
  "documents": [ { "document_id": "...", "type": "...", "service": "...", "date": "YYYY-MM-DD", "version": "...", "title": "...", "content": "..." } ]
}
```

For the live browser demo, open `demo.html` directly (no server needed) or use
the published artifact.

## Sample results (from `python3 demo.py`)

**Test A (deployment-related):**
> The incident described in INC-1042 (2026-09-16) began after deployment
> DEP-882 shipped v2.8.1 on 2026-09-15, so the deployment is the most likely
> trigger. A similar symptom occurred before: PM-211 (2026-05-03, v2.6.0)
> describes a related latency issue, though the root cause recorded there
> (database connection saturation during a schema migration) differs from
> the current incident's likely trigger, so treat this as
> similar-but-not-identical rather than a repeat of the same root cause.

**Test B (contradictory guidance):**
> Found conflicting guidance for this situation. Directly conflicting
> operational guidance on the same action (restart) for the same service.
> 'GUIDE-41' (2026-08-10, v3) is more recent than 'GUIDE-12' (2024-02-01, v1)
> and should be treated as the current guidance. Follow 'GUIDE-41' as the
> current procedure, and flag the older document for deprecation.

**Test C (insufficient evidence):**
> There is not enough evidence in the available documents to give a
> confident, evidence-backed answer to this question. No documents in the
> corpus matched this question at all.

**Test D (10-document evidence corpus — deployment, dependency, history, and
contradiction all in one case):** `data/test_d.json` packs ten documents
spanning six document types around a single `payments-api` checkout
incident — an incident report, a deployment note, an architecture/dependency
doc, two conflicting troubleshooting guides, a customer complaint, a
postmortem, a related incident on a *dependency* service
(`fraud-check-service`), and two clearly unrelated documents from other
services (`orders-api`, `catalog-api`) included specifically to prove the
retrieval layer filters out noise rather than dumping the whole corpus into
the answer. It's designed to exercise every requirement at once:

> Found conflicting guidance for this situation. Directly conflicting
> operational guidance on the same action (restart) for the same service.
> 'GUIDE-95' (2026-08-10, v4) is more recent than 'GUIDE-88' (2025-01-10,
> v3) and should be treated as the current guidance. Follow 'GUIDE-95' as
> the current procedure, and flag the older document for deprecation.

Ten documents are pulled into evidence (10/10, none dropped outright), but
they aren't treated equally: the two `payments-api` troubleshooting guides
surface as a genuine same-service, same-action contradiction and resolve to
the newer one; the deployment note, incident report, postmortem, and
architecture doc all score high on service + type relevance and back up the
deployment-trigger and historical-recurrence reasoning that would show in
the answer if no contradiction had taken priority; and the two decoy
documents (`DEP-700` on `orders-api`, `CC-050` on `catalog-api`) still show
up in the evidence list — each with a visibly weaker "weak topical match" or
partial keyword-overlap reason and no service-match boost — which is the
point: the reasoning trail makes it obvious *why* they're weak instead of
silently hiding them. Run it yourself with `python3 demo.py --case d` to see
the full evidence list and hop-by-hop trail.

## Extending it

- **Swap in real embeddings**: replace the body of `score_document()` with a
  cosine-similarity call against an embedding index; keep the `(score,
  reason)` return contract so multi-hop/contradiction/synthesis logic is
  untouched.
- **More document types**: add trigger keywords to `_TYPE_TRIGGER_WORDS`.
- **Richer contradiction detection**: currently negation-based on "restart";
  generalize by comparing any imperative verb phrase with/without negation
  across same-topic documents.
- **N-hop instead of 2-hop**: `investigate()` can be turned into a loop that
  keeps expanding the query with newly discovered entities until no new
  documents are found or a hop budget is exhausted.
