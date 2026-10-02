"""Contracts isolated from the scalar comparison harness."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
Status = Literal['ok', 'invalid', 'error', 'unavailable']
@dataclass(frozen=True)
class StructuredCase:
 case_id: str
 context: str
 schema: dict[str, dict[str, Any]]
 gold: dict[str, Any]
 metadata: dict[str, Any] = field(default_factory=dict)
@dataclass
class StructuredPrediction:
 provider: str
 model: str
 status: Status
 prediction: dict[str, Any] | None = None
 latency_ms: float | None = None
 exact_valid: bool = False
 error_code: str | None = None
 usage: dict[str, Any] = field(default_factory=dict)
 diagnostics: dict[str, Any] = field(default_factory=dict)
 @classmethod
 def unavailable(cls,provider:str,model:str,reason:str): return cls(provider,model,'unavailable',error_code=reason)
 @classmethod
 def invalid(cls,provider:str,model:str,code:str,latency_ms:float|None=None): return cls(provider,model,'invalid',latency_ms=latency_ms,error_code=code)
 @classmethod
 def error(cls,provider:str,model:str,code:str,latency_ms:float|None=None): return cls(provider,model,'error',latency_ms=latency_ms,error_code=code)
 def to_dict(self)->dict[str,Any]: return asdict(self)
