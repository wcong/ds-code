"""Runtime API."""

from typing import TYPE_CHECKING

from ds_code.runtime.approvals import ApprovalRecord, ApprovalStore

if TYPE_CHECKING:
	from ds_code.runtime.api import RuntimeApi

__all__ = [
	"RuntimeApi",
	"ApprovalRecord",
	"ApprovalStore",
]


def __getattr__(name: str):
	if name == "RuntimeApi":
		from ds_code.runtime.api import RuntimeApi

		return RuntimeApi
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
