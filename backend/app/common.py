"""Shared helpers: pagination, normalisation, audit logging, Indian states list."""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, TypeVar

from fastapi import Query, Request
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from .models import AuditLog, User

T = TypeVar("T")

INDIAN_STATES_AND_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep",
    "Puducherry",
]
_STATE_LOOKUP = {s.lower(): s for s in INDIAN_STATES_AND_UTS}
_STATE_LOOKUP.update({"nct of delhi": "Delhi", "new delhi": "Delhi", "orissa": "Odisha", "pondicherry": "Puducherry",
                      "j&k": "Jammu and Kashmir", "uttaranchal": "Uttarakhand"})


def normalize_state(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return _STATE_LOOKUP.get(re.sub(r"\s+", " ", value.strip().lower()))


def normalize_name(value: str) -> str:
    """Case/whitespace/punctuation-insensitive key used for duplicate detection."""
    v = unicodedata.normalize("NFKC", value).lower()
    v = v.replace("&", " and ")
    v = re.sub(r"[^\w\s]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def slugify(value: str) -> str:
    v = normalize_name(value.replace("+", " plus ").replace("#", " sharp "))
    return re.sub(r"\s+", "-", v)


class PageParams:
    def __init__(
        self,
        page: int = Query(1, ge=1, description="1-based page number"),
        page_size: int = Query(20, ge=1, le=200),
    ):
        self.page = page
        self.page_size = page_size


def paginate(db: Session, stmt: Select, params: PageParams, serializer=None) -> dict[str, Any]:
    total = db.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0
    rows = db.scalars(stmt.offset((params.page - 1) * params.page_size).limit(params.page_size)).all()
    items = [serializer(r) if serializer else r for r in rows]
    return {
        "items": items,
        "total": total,
        "page": params.page,
        "page_size": params.page_size,
        "pages": max(1, math.ceil(total / params.page_size)) if total else 0,
    }


def audit(
    db: Session,
    user: User | None,
    action: str,
    entity_type: str,
    entity_id: int | None = None,
    details: dict | None = None,
    request: Request | None = None,
) -> None:
    db.add(
        AuditLog(
            user_id=user.id if user else None,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
            ip_address=request.client.host if request and request.client else None,
        )
    )


def model_dict(obj: Any, fields: list[str]) -> dict[str, Any]:
    out = {}
    for f in fields:
        v = getattr(obj, f, None)
        out[f] = v.isoformat() if hasattr(v, "isoformat") else v
    return out
