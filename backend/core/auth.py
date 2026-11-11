"""Optional token-based access control.

ML-Agent has no user accounts. For shared deployments, set
``MLAGENT_ACCESS_TOKENS="alice:<token>,bob:<token>"`` (owner:token pairs). Requests to
protected endpoints must then send ``Authorization: Bearer <token>``; saved models are
owned by the token's owner and are invisible to other owners.

With no tokens configured the server runs in single-user *local* mode: protected
endpoints are open and every saved model belongs to the owner ``local``.
"""
from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from typing import Dict, Optional

LOCAL_OWNER = "local"
_OWNER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class AuthConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    owner: str
    authenticated: bool


def parse_tokens(raw: str) -> Dict[str, str]:
    """'alice:tok1,bob:tok2' → {'tok1': 'alice', 'tok2': 'bob'}. A bare token maps to 'default'."""
    tokens: Dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        owner, _, token = part.partition(":") if ":" in part else ("default", "", part)
        owner, token = owner.strip(), token.strip()
        if not _OWNER_RE.match(owner):
            raise AuthConfigError(f"Invalid owner name in MLAGENT_ACCESS_TOKENS: {owner!r}")
        if len(token) < 16:
            raise AuthConfigError(f"Access token for {owner!r} must be at least 16 characters.")
        tokens[token] = owner
    return tokens


class Authenticator:
    def __init__(self, raw_tokens: str):
        self.tokens = parse_tokens(raw_tokens)

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)

    def resolve(self, authorization: Optional[str]) -> Optional[Principal]:
        """Return the principal, or None when credentials are required but missing/invalid."""
        if not self.enabled:
            return Principal(LOCAL_OWNER, authenticated=False)
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        presented = authorization[7:].strip()
        for token, owner in self.tokens.items():
            if hmac.compare_digest(presented.encode(), token.encode()):
                return Principal(owner, authenticated=True)
        return None
