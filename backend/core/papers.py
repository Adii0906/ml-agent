"""arXiv search. Returns original paper metadata verbatim (no AI rewriting)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


class PaperSearchError(RuntimeError):
    pass


_cache: Dict[Tuple[str, int, str], Tuple[float, List[Dict[str, Any]]]] = {}
_lock = threading.Lock()
CACHE_SECONDS = 1800


def search(query: str, limit: int = 10, sort: str = "relevance") -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if len(query) < 2:
        raise PaperSearchError("Enter a longer search query.")
    limit = max(1, min(int(limit), 30))
    key = (query.lower(), limit, sort)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
    try:
        import arxiv
    except ImportError as exc:  # pragma: no cover
        raise PaperSearchError("The 'arxiv' package is not installed on the server.") from exc
    criterion = {"relevance": arxiv.SortCriterion.Relevance,
                 "date": arxiv.SortCriterion.SubmittedDate}.get(sort, arxiv.SortCriterion.Relevance)
    try:
        client = arxiv.Client(page_size=limit, delay_seconds=3, num_retries=2)
        results = client.results(arxiv.Search(query=query, max_results=limit, sort_by=criterion))
        papers = []
        for r in results:
            papers.append({
                "arxiv_id": r.get_short_id(),
                "title": " ".join(r.title.split()),
                "authors": [str(a) for a in r.authors],
                "published": r.published.isoformat() if r.published else None,
                "updated": r.updated.isoformat() if r.updated else None,
                "abstract": " ".join(r.summary.split()),
                "categories": list(r.categories or []),
                "primary_category": r.primary_category,
                "abs_url": r.entry_id,
                "pdf_url": r.pdf_url,
                "doi": r.doi,
                "journal_ref": r.journal_ref,
                "comment": r.comment,
            })
    except Exception as exc:
        log.warning("arXiv search failed: %s", exc)
        raise PaperSearchError(f"arXiv could not be reached: {exc}") from exc
    with _lock:
        _cache[key] = (time.time(), papers)
    return papers
