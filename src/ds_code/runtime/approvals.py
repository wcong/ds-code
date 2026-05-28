from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ApprovalRecord:
    approval_id: str
    status: str
    reason: Optional[str] = None


class ApprovalStore:
    def __init__(self) -> None:
        self._records: Dict[str, ApprovalRecord] = {}

    def record(self, approval_id: str, status: str, reason: Optional[str] = None) -> None:
        self._records[approval_id] = ApprovalRecord(
            approval_id=approval_id,
            status=status,
            reason=reason,
        )

    def get(self, approval_id: str) -> Optional[ApprovalRecord]:
        return self._records.get(approval_id)
