"""Evidence-ledger deep research miner for Harnyx.

Pipeline (each stage bounded by a wall-clock deadline and the session budget):

1. Contract:   a fast model turns the query into required facts + search queries,
               while a seed search on the raw query runs concurrently.
2. Retrieve:   all planned queries run in parallel; results are deduplicated and
               ranked lexically against the query and required facts.
3. Read:       the best pages are fetched in parallel and split into passages with
               exact character offsets into the fetched note text.
4. Ledger:     passages are scored with BM25 per required fact so every fact keeps
               its own best evidence instead of one global top-k.
5. Gap check:  a fast model maps facts to ledger ids and names missing facts; one
               targeted retrieval round fills them when time and budget allow.
6. Write:      the writer answers only from ledger ids ([E7]); ids are rewritten
               into exact [[n]] pointers with sliced CitationRefs.

Every stage degrades gracefully: tool failures shrink the ledger, model failures
fall through to the next route, and the entrypoint always returns a Response.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from harnyx_miner_sdk.api import ToolCallResponse, fetch_page, llm_chat, search_web
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, CitationSlice, Query, Response
from harnyx_miner_sdk.tools.search_models import SearchWebSearchResponse

# ---------------------------------------------------------------------------
# Configuration. Store a platform credential for every provider listed here
# (harnyx-miner-config --provider <name> --api-key ...). Routes are tried in
# order; a provider that fails is skipped for the rest of the query.
# ---------------------------------------------------------------------------

SEARCH_PROVIDER = "desearch"

FAST_ROUTES = (
    ("chutes", "google/gemma-4-31B-turbo-TEE", None),
    ("openrouter", "openai/gpt-oss-120b", {"provider": {"only": ["cerebras"]}}),
    ("chutes", "deepseek-ai/DeepSeek-V3.2-TEE", None),
)
WRITER_ROUTES = (
    ("chutes", "deepseek-ai/DeepSeek-V3.2-TEE", None),
    ("openrouter", "deepseek/deepseek-v4-pro", None),
    ("chutes", "moonshotai/Kimi-K2.6-TEE", None),
)

SOFT_DEADLINE_S = 150.0
WRITER_RESERVE_S = 75.0
GAP_ROUND_CUTOFF_S = 60.0
MIN_BUDGET_FOR_RETRIEVAL_USD = 0.05

SEARCH_RESULTS_PER_QUERY = 8
MAX_PLANNED_QUERIES = 5
MAX_FETCHES = 6
MAX_GAP_FETCHES = 3
SEARCH_TIMEOUT_S = 15.0
FETCH_TIMEOUT_S = 20.0
FAST_LLM_TIMEOUT_S = 30.0
WRITER_LLM_TIMEOUT_S = 70.0

PASSAGE_TARGET_CHARS = 900
PASSAGE_MIN_CHARS = 120
MAX_PAGE_CHARS = 60000
MAX_LEDGER_ENTRIES = 36
MAX_LEDGER_CHARS = 42000
PER_FACT_PASSAGES = 4

STOPWORDS = frozenset(
    "a an the and or of to in on for with by from at as is are was were be been being "
    "what which who whom whose when where why how that this these those it its into than "
    "then there their about over under between does do did has have had not no can could "
    "should would will may might also any all each more most other such only same so very "
    "much many some your you we our they them he she his her i me my vs versus per".split()
)
AUTHORITY_HINTS = (".gov", ".edu", ".int", "wikipedia.org", ".org/", "who.int", "europa.eu", "sec.gov")
LOW_VALUE_HINTS = ("pinterest.", "facebook.com", "instagram.com", "tiktok.com", "quora.com", "/tag/", "/login")


@dataclass
class Candidate:
    url: str
    title: str
    snippet: str
    receipt_id: str
    result_id: str
    hits: int = 1
    score: float = 0.0


@dataclass
class Passage:
    text: str
    url: str
    title: str
    receipt_id: str
    result_id: str
    start: int
    end: int
    whole_note: bool
    fetched: bool
    score: float = 0.0


@dataclass
class Run:
    query: Query
    started: float
    today: str
    dead_providers: set = field(default_factory=set)
    remaining_budget: float = 1.0
    candidates: dict = field(default_factory=dict)
    fetched_urls: set = field(default_factory=set)
    passages: list = field(default_factory=list)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def time_left(self) -> float:
        return SOFT_DEADLINE_S - self.elapsed()

    def note_budget(self, envelope: object) -> None:
        budget = getattr(envelope, "budget", None)
        if budget is not None:
            self.remaining_budget = float(budget.session_remaining_budget_usd)


@entrypoint("query")
async def query(query: Query) -> Response:
    run = Run(
        query=query,
        started=time.monotonic(),
        today=datetime.now(UTC).strftime("%Y-%m-%d"),
    )
    try:
        return await _research(run)
    except Exception:
        return _fallback_response(run)


async def _research(run: Run) -> Response:
    question = run.query.text
    plan_task = asyncio.create_task(_plan(run))
    seed_task = asyncio.create_task(_search(run, [_clip(question, 380)]))
    plan = await plan_task
    planned = [q for q in plan["queries"] if q.lower() != question.lower()][:MAX_PLANNED_QUERIES]
    await _search(run, planned)
    await seed_task

    facts = plan["facts"] or [question]
    terms = _query_terms(question, facts)
    _rank_candidates(run, terms)
    await _fetch_best(run, MAX_FETCHES)
    ledger = _build_ledger(run, question, facts)

    if run.elapsed() < GAP_ROUND_CUTOFF_S and run.remaining_budget > MIN_BUDGET_FOR_RETRIEVAL_USD:
        gaps = await _find_gaps(run, facts, ledger)
        if gaps:
            await _search(run, gaps[:3])
            _rank_candidates(run, _query_terms(question, facts + gaps))
            await _fetch_best(run, MAX_GAP_FETCHES)
            ledger = _build_ledger(run, question, facts + gaps)

    if not ledger:
        ledger = _build_ledger(run, question, facts)
    return await _write(run, plan, ledger)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def _chat(run: Run, routes: tuple, messages: list, *, max_output_tokens: int, timeout: float) -> str | None:
    for provider, model, extra in routes:
        if provider in run.dead_providers:
            continue
        budget_left = run.time_left() + WRITER_RESERVE_S
        if budget_left < 8.0:
            return None
        kwargs = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_output_tokens": max_output_tokens,
            "timeout": max(8.0, min(timeout, budget_left)),
        }
        if extra is not None:
            kwargs["provider_extra"] = extra
        if provider == "openrouter":
            kwargs["thinking"] = {"enabled": True, "effort": "low"}
        elif "gemma" in model.lower():
            kwargs["thinking"] = {"enabled": False}
        try:
            result = await llm_chat(**kwargs)
        except Exception as exc:
            if _looks_like_credential_error(exc):
                run.dead_providers.add(provider)
            continue
        run.note_budget(result)
        text = result.llm.raw_text
        if text:
            return text
    return None


def _looks_like_credential_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("credential", "api key", "api_key", "unauthorized", "401", "403"))


def _parse_json_object(text: str | None) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(_clip(item.strip(), 300))
        if len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# Stage 1: answer contract
# ---------------------------------------------------------------------------

PLAN_PROMPT = """Today is {today}. You plan web research for one question.
Return JSON only:
{{"facts": [...], "queries": [...], "recency": true|false}}

- "facts": 2-6 atomic facts the final answer must establish. Cover every part of the
  question (each entity, metric, date, comparison side). Name traps such as a false
  premise or an easily confused entity as their own fact ("verify whether ...").
- "queries": 3-5 diverse keyword web searches (not sentences) that together find
  primary or authoritative evidence for every fact. Include entity names, years,
  and official terms. Use the question's language plus English when helpful.
- "recency": true when the answer depends on current or recent status.

Question:
{question}"""


async def _plan(run: Run) -> dict:
    prompt = PLAN_PROMPT.format(today=run.today, question=_clip(run.query.text, 6000))
    text = await _chat(
        run,
        FAST_ROUTES,
        [{"role": "user", "content": prompt}],
        max_output_tokens=900,
        timeout=FAST_LLM_TIMEOUT_S,
    )
    parsed = _parse_json_object(text) or {}
    facts = _string_list(parsed.get("facts"), 6)
    queries = _string_list(parsed.get("queries"), MAX_PLANNED_QUERIES)
    if not queries:
        queries = _fallback_queries(run.query.text)
    recency = parsed.get("recency") is True
    if recency:
        year = run.today[:4]
        queries = [q if year in q else f"{q} {year}" for q in queries[:2]] + queries[2:]
    return {"facts": facts, "queries": queries, "recency": recency}


def _fallback_queries(question: str) -> list[str]:
    words = [w for w in re.findall(r"\w+", question) if w.lower() not in STOPWORDS]
    queries = [" ".join(words[:12])] if words else []
    for sentence in re.split(r"[?.;\n]", question):
        sentence = sentence.strip()
        if len(sentence) > 20 and sentence not in queries:
            queries.append(_clip(sentence, 200))
    return queries[:MAX_PLANNED_QUERIES] or [_clip(question, 200)]


# ---------------------------------------------------------------------------
# Stage 2/3: retrieval and reading
# ---------------------------------------------------------------------------


async def _search(run: Run, queries: list[str]) -> None:
    queries = [q for q in queries if q.strip()]
    if not queries or run.time_left() < 20.0 or run.remaining_budget < MIN_BUDGET_FOR_RETRIEVAL_USD:
        return
    outcomes = await asyncio.gather(*[_search_one(run, q) for q in queries], return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException) or outcome is None:
            continue
        run.note_budget(outcome)
        for result in outcome.results:
            url = result.url
            if not url:
                continue
            key = _url_key(url)
            existing = run.candidates.get(key)
            if existing is not None:
                existing.hits += 1
                continue
            run.candidates[key] = Candidate(
                url=url,
                title=result.title or "",
                snippet=result.note or "",
                receipt_id=outcome.receipt_id,
                result_id=result.result_id,
            )


async def _search_one(run: Run, text: str) -> ToolCallResponse[SearchWebSearchResponse]:
    return await search_web(
        _clip(text, 380),
        provider=SEARCH_PROVIDER,
        num=SEARCH_RESULTS_PER_QUERY,
        timeout=min(SEARCH_TIMEOUT_S, max(5.0, run.time_left())),
    )


def _rank_candidates(run: Run, terms: list[str]) -> None:
    term_set = set(terms)
    for candidate in run.candidates.values():
        tokens = _tokens(f"{candidate.title} {candidate.snippet}")
        overlap = len(term_set.intersection(tokens))
        url = candidate.url.lower()
        bonus = 1.5 if any(hint in url for hint in AUTHORITY_HINTS) else 0.0
        penalty = 3.0 if any(hint in url for hint in LOW_VALUE_HINTS) else 0.0
        candidate.score = overlap + 1.5 * (candidate.hits - 1) + bonus - penalty


async def _fetch_best(run: Run, limit: int) -> None:
    if run.time_left() < 25.0 or run.remaining_budget < MIN_BUDGET_FOR_RETRIEVAL_USD:
        return
    ranked = sorted(run.candidates.values(), key=lambda c: c.score, reverse=True)
    chosen: list[Candidate] = []
    hosts: dict[str, int] = {}
    for candidate in ranked:
        if candidate.url in run.fetched_urls or candidate.url.lower().endswith((".mp4", ".zip", ".jpg", ".png")):
            continue
        host = _host(candidate.url)
        if hosts.get(host, 0) >= 2:
            continue
        hosts[host] = hosts.get(host, 0) + 1
        chosen.append(candidate)
        if len(chosen) >= limit:
            break
    for candidate in chosen:
        run.fetched_urls.add(candidate.url)
    timeout = min(FETCH_TIMEOUT_S, max(6.0, run.time_left() - 10.0))
    outcomes = await asyncio.gather(
        *[fetch_page(c.url, provider=SEARCH_PROVIDER, timeout=timeout) for c in chosen],
        return_exceptions=True,
    )
    for candidate, outcome in zip(chosen, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            continue
        run.note_budget(outcome)
        for result in outcome.results:
            note = result.note or ""
            if len(note) < PASSAGE_MIN_CHARS:
                continue
            run.passages.extend(
                _split_passages(
                    note,
                    url=result.url or candidate.url,
                    title=result.title or candidate.title,
                    receipt_id=outcome.receipt_id,
                    result_id=result.result_id,
                )
            )


def _split_passages(note: str, *, url: str, title: str, receipt_id: str, result_id: str) -> list[Passage]:
    limit = min(len(note), MAX_PAGE_CHARS)
    passages: list[Passage] = []
    cursor = 0
    while cursor < limit:
        end = min(limit, cursor + PASSAGE_TARGET_CHARS)
        if end < limit:
            window = note[cursor:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > PASSAGE_TARGET_CHARS // 2:
                end = cursor + cut + 1
        text = note[cursor:end]
        if len(text.strip()) >= PASSAGE_MIN_CHARS and _looks_like_prose(text):
            passages.append(
                Passage(
                    text=text,
                    url=url,
                    title=title,
                    receipt_id=receipt_id,
                    result_id=result_id,
                    start=cursor,
                    end=end,
                    whole_note=False,
                    fetched=True,
                )
            )
        cursor = end
    return passages


def _looks_like_prose(text: str) -> bool:
    words = re.findall(r"\w+", text)
    if len(words) < 15:
        return False
    link_noise = text.count("](") + text.count("http")
    return link_noise * 12 < len(words)


# ---------------------------------------------------------------------------
# Stage 4: per-fact evidence ledger
# ---------------------------------------------------------------------------


def _build_ledger(run: Run, question: str, facts: list[str]) -> list[Passage]:
    pool: list[Passage] = list(run.passages)
    for candidate in run.candidates.values():
        if len(candidate.snippet) >= 40:
            pool.append(
                Passage(
                    text=candidate.snippet,
                    url=candidate.url,
                    title=candidate.title,
                    receipt_id=candidate.receipt_id,
                    result_id=candidate.result_id,
                    start=0,
                    end=len(candidate.snippet),
                    whole_note=True,
                    fetched=False,
                )
            )
    if not pool:
        return []
    scorer = _Bm25([_tokens(f"{p.title} {p.text}") for p in pool])
    selected: list[int] = []
    seen: set[int] = set()
    question_scores = scorer.scores(_tokens(question))
    for fact in facts:
        fact_scores = scorer.scores(_tokens(fact))
        combined = [
            fact_scores[i] + 0.35 * question_scores[i] + (0.4 if pool[i].fetched else 0.0) for i in range(len(pool))
        ]
        order = sorted(range(len(pool)), key=lambda i: combined[i], reverse=True)
        taken = 0
        for index in order:
            if combined[index] <= 0.0 or taken >= PER_FACT_PASSAGES:
                break
            if index in seen:
                taken += 1
                continue
            seen.add(index)
            selected.append(index)
            taken += 1
    for index in sorted(range(len(pool)), key=lambda i: question_scores[i], reverse=True):
        if len(selected) >= MAX_LEDGER_ENTRIES or question_scores[index] <= 0.0:
            break
        if index not in seen:
            seen.add(index)
            selected.append(index)
    ledger: list[Passage] = []
    fingerprints: set[str] = set()
    total = 0
    for index in selected:
        if len(ledger) >= MAX_LEDGER_ENTRIES:
            break
        passage = pool[index]
        fingerprint = " ".join(_tokens(passage.text)[:30])
        if fingerprint in fingerprints or total + len(passage.text) > MAX_LEDGER_CHARS:
            continue
        fingerprints.add(fingerprint)
        total += len(passage.text)
        ledger.append(passage)
    return ledger


class _Bm25:
    def __init__(self, documents: list[list[str]]) -> None:
        self.documents = documents
        self.lengths = [len(doc) or 1 for doc in documents]
        self.average = sum(self.lengths) / max(1, len(self.lengths))
        frequencies: dict[str, int] = {}
        for doc in documents:
            for token in set(doc):
                frequencies[token] = frequencies.get(token, 0) + 1
        count = len(documents)
        self.idf = {token: math.log(1.0 + (count - df + 0.5) / (df + 0.5)) for token, df in frequencies.items()}
        self.counts = []
        for doc in documents:
            counter: dict[str, int] = {}
            for token in doc:
                counter[token] = counter.get(token, 0) + 1
            self.counts.append(counter)

    def scores(self, terms: list[str]) -> list[float]:
        unique = [t for t in set(terms) if t in self.idf]
        results: list[float] = []
        for counter, length in zip(self.counts, self.lengths, strict=True):
            total = 0.0
            for term in unique:
                tf = counter.get(term, 0)
                if tf:
                    total += self.idf[term] * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / self.average))
            results.append(total)
        return results


# ---------------------------------------------------------------------------
# Stage 5: gap check
# ---------------------------------------------------------------------------

GAP_PROMPT = """Today is {today}. Decide which required facts are NOT yet supported by the evidence.
Return JSON only: {{"missing": [{{"fact": "...", "query": "keyword web search"}}]}}
Return an empty list when the evidence already supports every fact. At most 3 items.

Question:
{question}

Required facts:
{facts}

Evidence:
{evidence}"""


async def _find_gaps(run: Run, facts: list[str], ledger: list[Passage]) -> list[str]:
    evidence = "\n".join(f"[E{i + 1}] {_clip(p.text, 500)}" for i, p in enumerate(ledger[:24]))
    prompt = GAP_PROMPT.format(
        today=run.today,
        question=_clip(run.query.text, 4000),
        facts="\n".join(f"- {f}" for f in facts),
        evidence=evidence or "(no evidence found)",
    )
    text = await _chat(
        run,
        FAST_ROUTES,
        [{"role": "user", "content": prompt}],
        max_output_tokens=500,
        timeout=FAST_LLM_TIMEOUT_S,
    )
    parsed = _parse_json_object(text) or {}
    missing = parsed.get("missing")
    queries: list[str] = []
    if isinstance(missing, list):
        for item in missing[:3]:
            if isinstance(item, dict) and isinstance(item.get("query"), str) and item["query"].strip():
                queries.append(_clip(item["query"].strip(), 300))
    return queries


# ---------------------------------------------------------------------------
# Stage 6: grounded writing
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You are a meticulous research analyst. Today is {today}.
Answer the user's question using ONLY the numbered evidence passages.

Rules:
- Answer every part of the question directly first, then give the key supporting detail.
- After each factual sentence, cite the passage ids that directly contain that fact, e.g. [E3] or [E2][E7].
  Cite only passages whose text actually states the claim. Never invent ids.
- If evidence conflicts, say so and prefer primary/official and more recent sources.
- If the question contains a false premise, correct it explicitly.
- If a required fact is not in the evidence, say clearly that it could not be verified;
  do not guess numbers, dates, or names. Common knowledge needs no citation.
- Follow any format, length, or language the question requests; otherwise answer in the
  question's language as clear, compact Markdown (short paragraphs or bullets, no title).
- Do not add a sources or references section and do not paste raw URLs."""

WRITER_JSON_SYSTEM_SUFFIX = """
The caller requires structured output. Return JSON only, shaped as:
{{"answer": <value matching the JSON Schema below>, "evidence": [<ids such as "E3" that support the answer>]}}
Do not put citation markers inside atomic fields; only use [E#] markers inside prose
fields if the schema or question explicitly asks for citations.
JSON Schema:
{schema}"""


async def _write(run: Run, plan: dict, ledger: list[Passage]) -> Response:
    evidence = "\n\n".join(
        f"[E{i + 1}] {p.title or _host(p.url)} ({_host(p.url)})\n{p.text.strip()}" for i, p in enumerate(ledger)
    )
    facts = plan["facts"]
    user = (
        f"Question:\n{run.query.text}\n\n"
        + (("Facts the answer must cover:\n" + "\n".join(f"- {f}" for f in facts) + "\n\n") if facts else "")
        + f"Evidence:\n{evidence or '(no evidence was retrieved)'}"
    )
    system = WRITER_SYSTEM.format(today=run.today)
    schema = run.query.output_schema
    if schema is not None:
        system += WRITER_JSON_SYSTEM_SUFFIX.format(schema=json.dumps(schema, ensure_ascii=False)[:20000])
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = await _chat(run, WRITER_ROUTES, messages, max_output_tokens=3000, timeout=WRITER_LLM_TIMEOUT_S)
    if text is None:
        text = await _chat(run, FAST_ROUTES, messages, max_output_tokens=2500, timeout=WRITER_LLM_TIMEOUT_S)
    if schema is not None:
        return _structured_response(text, ledger, schema)
    if not text:
        return _fallback_response(run, ledger)
    return _prose_response(text, ledger)


EVIDENCE_MARKER = re.compile(r"\[\s*E\d+(?:\s*[,;]\s*E?\d+)*\s*\](?:\s*\[\s*E\d+(?:\s*[,;]\s*E?\d+)*\s*\])*")


def _prose_response(text: str, ledger: list[Passage]) -> Response:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    citations: list[CitationRef] = []
    positions: dict[int, int] = {}

    def replace_group(match: re.Match) -> str:
        ids = [int(n) for n in re.findall(r"\d+", match.group(0))]
        pointers: list[str] = []
        for evidence_id in ids:
            index = evidence_id - 1
            if index < 0 or index >= len(ledger):
                continue
            if index not in positions:
                if len(citations) >= 190:
                    continue
                citations.append(_citation_for(ledger[index]))
                positions[index] = len(citations)
            pointer = f"[[{positions[index]}]]"
            if pointer not in pointers:
                pointers.append(pointer)
        return "".join(pointers)

    rewritten = EVIDENCE_MARKER.sub(replace_group, text)
    rewritten = re.sub(r"[ \t]+(\[\[)", r" \1", rewritten)
    rewritten = re.sub(r"[ \t]+([.,;:!?)])", r"\1", rewritten)
    rewritten = rewritten.strip()[:79000]
    if not rewritten:
        rewritten = text[:79000] or "No answer could be produced."
    return Response(text=rewritten, citations=citations or None)


def _structured_response(text: str | None, ledger: list[Passage], schema: dict) -> Response:
    parsed = _parse_json_object(text)
    answer = None
    evidence_ids: list[int] = []
    if parsed is not None and "answer" in parsed:
        answer = parsed["answer"]
        for item in parsed.get("evidence") or []:
            digits = re.findall(r"\d+", str(item))
            if digits:
                evidence_ids.append(int(digits[0]))
    elif parsed is not None:
        answer = parsed
    if answer is None:
        answer = _empty_for_schema(schema)
    citations: list[CitationRef] = []
    for evidence_id in evidence_ids:
        index = evidence_id - 1
        if 0 <= index < len(ledger) and len(citations) < 40:
            ref = _citation_for(ledger[index])
            if ref not in citations:
                citations.append(ref)
    answer = _strip_markers(answer)
    return Response(output=answer, citations=citations or None)


def _strip_markers(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"\s*\[E\d+\]", "", value)
    if isinstance(value, list):
        return [_strip_markers(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_markers(item) for key, item in value.items()}
    return value


def _empty_for_schema(schema: dict) -> object:
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object":
        result = {}
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            sub = properties.get(name)
            result[name] = _empty_for_schema(sub) if isinstance(sub, dict) else ""
        return result
    if kind == "array":
        return []
    if kind == "integer":
        return 0
    if kind == "number":
        return 0
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    return "Not verifiable from available evidence."


def _citation_for(passage: Passage) -> CitationRef:
    if passage.whole_note:
        return CitationRef(receipt_id=passage.receipt_id, result_id=passage.result_id)
    return CitationRef(
        receipt_id=passage.receipt_id,
        result_id=passage.result_id,
        slices=[CitationSlice(start=passage.start, end=passage.end)],
    )


def _fallback_response(run: Run, ledger: list[Passage] | None = None) -> Response:
    items = ledger if ledger is not None else _build_ledger(run, run.query.text, [run.query.text])
    if run.query.output_schema is not None:
        return Response(output=_empty_for_schema(run.query.output_schema))
    lines: list[str] = []
    citations: list[CitationRef] = []
    for passage in items[:5]:
        snippet = re.sub(r"\s+", " ", passage.text).strip()
        citations.append(_citation_for(passage))
        lines.append(f"- {_clip(snippet, 400)} [[{len(citations)}]]")
    if not lines:
        return Response(text="I could not retrieve evidence to answer this question reliably.")
    return Response(text="Relevant findings:\n\n" + "\n".join(lines), citations=citations)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def _query_terms(question: str, facts: list[str]) -> list[str]:
    return _tokens(question + " " + " ".join(facts))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _host(url: str) -> str:
    match = re.match(r"^[a-z]+://([^/]+)", url.lower())
    host = match.group(1) if match else url.lower()
    return host[4:] if host.startswith("www.") else host


def _url_key(url: str) -> str:
    key = re.sub(r"#.*$", "", url.strip().lower())
    key = re.sub(r"^https?://(www\.)?", "", key)
    return key.rstrip("/")
