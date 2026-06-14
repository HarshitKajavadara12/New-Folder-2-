"""
GIGA SYSTEM - Execution Instructions (Stage 4)
Phase 3 Foundation 2 (Lifecycle Stage 4)

These are the "Finite Actions".
The Execution Engine only understands these objects.
No probability, no Greek letters, no ambiguity.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any


class ActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CANCEL = "CANCEL"
    HALT = "HALT"  # Kill switch


class InstructionStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class ExecutionInstruction:
    """
    A deterministic execution command.
    Reduced from probabilistic signals to finite actions.
    """
    id: str
    action: ActionType
    asset: str
    quantity: float  # float for crypto fractional quantities
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: InstructionStatus = InstructionStatus.PENDING

    # Traceability back to the Artifact that caused this
    source_artifact_id: Optional[str] = None
    
    # Execution metadata
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate instruction is well-formed."""
        if not self.id or not self.asset:
            return False
        if self.quantity <= 0:
            return False
        if self.limit_price is not None and self.limit_price <= 0:
            return False
        if self.stop_price is not None and self.stop_price <= 0:
            return False
        if self.expires_at and self.expires_at <= self.created_at:
            return False
        return True

    @property
    def is_active(self) -> bool:
        """Check if instruction is still active."""
        if self.status in (InstructionStatus.FILLED, InstructionStatus.CANCELLED,
                          InstructionStatus.REJECTED, InstructionStatus.EXPIRED):
            return False
        if self.expires_at and datetime.now() >= self.expires_at:
            return False
        return True

    @property
    def fill_ratio(self) -> float:
        """Fraction of quantity filled."""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0

    def __repr__(self) -> str:
        return (f"ExecutionInstruction({self.action.value} {self.quantity} "
                f"{self.asset} @ {self.limit_price or 'MKT'}, "
                f"status={self.status.value})")
