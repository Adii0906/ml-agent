"""Optional LLM layer (Groq or local Ollama).

The LLM is used ONLY to explain or summarise facts computed elsewhere. It never
selects the target, decides compatibility, or produces metrics. When no
provider is configured every core workflow still works; callers receive
``LLMUnavailable`` and the UI shows that AI text is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import requests

from .store import Store

log = logging.getLogger(__name__)

AI_LABEL = "AI-generated"
DISCLAIMER = ("AI-generated recommendations and explanations may contain errors. Verify your dataset, "
              "preprocessing decisions and evaluation methodology independently. Model performance depends "
              "on data quality, validation strategy and real-world conditions.")

DEFAULTS = {
    "llm_provider": "auto",  # auto | groq | ollama | none
    "groq_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
}

GUARDRAIL = (
    "You are assisting a machine learning engineer. Use ONLY the facts provided. "
    "Do not invent numbers, metrics, datasets, citations or performance guarantees. "
    "If something is not in the facts, say it is unknown. Be concise and technical."
)


class LLMUnavailable(RuntimeError):
    pass


class LLM:
    def __init__(self, store: Store):
        self.store = store
        self._groq = None
        self._groq_key = None

    # -------------------------------------------------------------- config
    def config(self) -> Dict[str, Any]:
        cfg = dict(DEFAULTS)
        stored = self.store.get_settings()
        cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
        if os.getenv("USE_OLLAMA", "").lower() == "true" and "llm_provider" not in stored:
            cfg["llm_provider"] = "ollama"
        return cfg

    def update_config(self, values: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}
        for k, v in values.items():
            if k not in DEFAULTS:
                raise ValueError(f"Unknown setting '{k}'.")
            if k == "llm_provider" and v not in ("auto", "groq", "ollama", "none"):
                raise ValueError("llm_provider must be auto, groq, ollama or none.")
            if k == "ollama_url" and not re.match(r"^https?://", str(v)):
                raise ValueError("ollama_url must start with http:// or https://")
            clean[k] = str(v).strip()
        self.store.set_settings(clean)
        return self.config()

    def _groq_client(self):
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            return None
        if self._groq is None or self._groq_key != key:
            from groq import Groq  # imported lazily so the app runs without the package
            self._groq = Groq(api_key=key, timeout=45)
            self._groq_key = key
        return self._groq

    def provider(self) -> Optional[str]:
        cfg = self.config()
        p = cfg["llm_provider"]
        if p == "none":
            return None
        if p == "groq":
            return "groq" if os.getenv("GROQ_API_KEY") else None
        if p == "ollama":
            return "ollama"
        # auto: Groq when a key is configured, otherwise nothing (Ollama must be chosen explicitly)
        return "groq" if os.getenv("GROQ_API_KEY") else None

    def status(self, probe: bool = False) -> Dict[str, Any]:
        cfg = self.config()
        prov = self.provider()
        out = {"configured_provider": cfg["llm_provider"], "active_provider": prov,
               "groq_key_present": bool(os.getenv("GROQ_API_KEY")),
               "groq_model": cfg["groq_model"], "ollama_url": cfg["ollama_url"], "ollama_model": cfg["ollama_model"],
               "available": prov is not None, "reason": None}
        if prov is None:
            out["reason"] = ("AI features are disabled." if cfg["llm_provider"] == "none" else
                             "Set GROQ_API_KEY in .env, or choose Ollama in Settings.")
        if probe and prov:
            try:
                self.complete("Reply with the single word: ok", json_mode=False, max_tokens=5)
                out["reachable"] = True
            except Exception as exc:
                out["reachable"] = False
                out["reason"] = str(exc)[:300]
        return out

    # -------------------------------------------------------------- calls
    def complete(self, prompt: str, json_mode: bool = True, max_tokens: int = 900) -> str:
        prov = self.provider()
        if prov is None:
            raise LLMUnavailable(self.status()["reason"])
        cfg = self.config()
        if prov == "groq":
            client = self._groq_client()
            kwargs: Dict[str, Any] = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            res = client.chat.completions.create(
                model=cfg["groq_model"],
                messages=[{"role": "system", "content": GUARDRAIL}, {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=max_tokens, **kwargs)
            return res.choices[0].message.content or ""
        resp = requests.post(f"{cfg['ollama_url'].rstrip('/')}/api/generate", timeout=90, json={
            "model": cfg["ollama_model"], "prompt": f"{GUARDRAIL}\n\n{prompt}", "stream": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens},
            **({"format": "json"} if json_mode else {})})
        if resp.status_code != 200:
            raise LLMUnavailable(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("response", "")

    def complete_json(self, prompt: str, max_tokens: int = 900) -> Dict[str, Any]:
        text = self.complete(prompt, json_mode=True, max_tokens=max_tokens)
        try:
            return json.loads(text)
        except ValueError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except ValueError:
                    pass
        raise LLMUnavailable("The model returned text that is not valid JSON.")

    def generated(self, content: Any) -> Dict[str, Any]:
        cfg = self.config()
        prov = self.provider()
        model = cfg["groq_model"] if prov == "groq" else cfg["ollama_model"]
        return {"label": AI_LABEL, "provider": prov, "model": model, "disclaimer": DISCLAIMER, "content": content}

    # ---------------------------------------------------------- use cases
    def explain_recommendations(self, rec: Dict[str, Any], target: str) -> Dict[str, Any]:
        top = [{k: a[k] for k in ("name", "score", "reasons", "cautions")} for a in rec["algorithms"] if a.get("compatible")][:5]
        prompt = (
            "Explain, for the engineer, why these algorithms were ranked this way for the dataset. "
            "The ranking was computed by deterministic rules; do not re-rank and do not predict accuracy.\n\n"
            f"Facts: {json.dumps({'target': target, 'task': rec['task'], 'dataset': rec['facts'], 'ranked': top}, default=str)}\n\n"
            'Return JSON: {"overview": "2-3 sentences", "per_algorithm": [{"name": "...", "note": "1-2 sentences"}], '
            '"watch_out": ["short caution", ...]}')
        return self.generated(self.complete_json(prompt))

    def report_narrative(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "Write a short narrative for an experiment report using ONLY these computed facts. "
            "Quote numbers exactly as given. Do not claim the model will perform the same in production.\n\n"
            f"Facts: {json.dumps(facts, default=str)}\n\n"
            'Return JSON: {"summary": "3 sentences", "observations": ["..."], "next_steps": ["..."]}')
        return self.generated(self.complete_json(prompt, max_tokens=1000))

    def paper_relevance(self, query: str, papers: list) -> Dict[str, Any]:
        items = [{"i": i, "title": p["title"], "abstract": p["abstract"][:700]} for i, p in enumerate(papers)]
        prompt = (
            f'For the research query "{query}", explain in one sentence per paper how it relates to the query, '
            "based only on the title and abstract provided.\n\n"
            f"Papers: {json.dumps(items)}\n\n"
            'Return JSON: {"notes": [{"i": 0, "relevance": "one sentence"}]}')
        return self.generated(self.complete_json(prompt, max_tokens=1200))

    def summarize_paper(self, paper: Dict[str, Any], context: Optional[str]) -> Dict[str, Any]:
        prompt = (
            "Summarise this paper for an ML engineer from its abstract only (you have not read the full text). "
            + (f"Relate it to the engineer's context: {context}. " if context else "")
            + f"\n\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
            'Return JSON: {"tldr": "1 sentence", "key_points": ["..."], "methods": ["..."], '
            '"practical_takeaways": ["..."], "limitations_unknown": "what cannot be judged from the abstract"}')
        return self.generated(self.complete_json(prompt))
