"""
GIGA SYSTEM - Artifact Definitions (Phase 3 Foundation)
"Context is Part of Math"

This module defines the "Reduced" state of the artifact lifecycle (Stage 3).
All math must be reduced to these structures to pass through the Air-Gap.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

class MarketRegime(Enum):
    """
    Foundation 5: Context Contract
    Defines the market reality under which an artifact is valid.
    """
    UNKNOWN = "unknown"
    LOW_VOL_BULL = "low_vol_bull"
    LOW_VOL_BEAR = "low_vol_bear"
    HIGH_VOL_CRASH = "high_vol_crash"
    HIGH_VOL_RALLY = "high_vol_rally"
    SIDEWAYS_CHOP = "sideways_chop"

class TimeHorizon(Enum):
    """Execution time horizon."""
    HFT = "hft_micro"       # < 1ms
    INTRA_MINUTE = "1min"   # 1 min
    INTRA_HOUR = "1hour"    # 1 hour
    DAILY = "1day"          # 1 day

@dataclass
class Context:
    """
    Foundation 5: The Context Contract.
    A number without context is dangerous noise.
    """
    regime: MarketRegime
    horizon: TimeHorizon
    asset_class: str
    constraints: Dict[str, float] = field(default_factory=dict)
    valid_until: Optional[datetime] = None

@dataclass
class Artifact:
    """
    Base class for all Reduced Artifacts (Stage 3).
    Must be deterministic and contextualized.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_artifact"
    version: str = "0.1.0"
    created_at: datetime = field(default_factory=datetime.now)
    context: Context = None
    
    # The actual reduced content (parameters, weights, thresholds)
    # NOT executable code.
    content: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate artifact meets Phase 3 requirements."""
        if self.context is None:
            return False
        if not isinstance(self.context.regime, MarketRegime):
            return False
        if not isinstance(self.context.horizon, TimeHorizon):
            return False
        if not self.context.asset_class:
            return False
        if self.context.valid_until and self.context.valid_until < datetime.now():
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'context': {
                'regime': self.context.regime.value if self.context else None,
                'horizon': self.context.horizon.value if self.context else None,
                'asset_class': self.context.asset_class if self.context else None,
            },
            'content': self.content
        }
    
    def __repr__(self) -> str:
        ctx = self.context.regime.value if self.context else 'no_context'
        return f"Artifact(id={self.id[:8]}..., name={self.name}, context={ctx})"

@dataclass
class SignalArtifact(Artifact):
    """A specific type of artifact that suggests an action."""
    direction: float = 0.0  # -1.0 to 1.0
    strength: float = 0.0   # 0.0 to 1.0
    confidence: float = 0.0 # 0.0 to 1.0
    
    def validate(self) -> bool:
        """Validate signal artifact with range checks."""
        if not super().validate():
            return False
        if not (-1.0 <= self.direction <= 1.0):
            return False
        if not (0.0 <= self.strength <= 1.0):
            return False
        if not (0.0 <= self.confidence <= 1.0):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize signal artifact."""
        base = super().to_dict()
        base.update({
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence
        })
        return base
    
    def __repr__(self) -> str:
        return (f"SignalArtifact(dir={self.direction:+.2f}, "
                f"str={self.strength:.2f}, conf={self.confidence:.2f})")
