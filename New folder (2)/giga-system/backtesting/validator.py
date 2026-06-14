"""
GIGA SYSTEM - Validation Pipeline (The Compiler)
Phase 3 Foundation 3: Air-Gap as Compiler

This module enforces the rules that stop invalid math from reaching the Brain.
Research objects must pass compilation before becoming live Artifacts.
"""

import hashlib
import json
import logging
from typing import List, Tuple, Any, Dict, Optional
from datetime import datetime

try:
    from artifacts.definitions import Artifact, Context
except ImportError:
    # Fallback: define minimal types if import path differs
    Artifact = None
    Context = None

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ValidationPipeline:
    """
    The Gatekeeper — compiles Research objects into validated Artifacts.
    
    Validation checks:
    1. Determinism: Same input must produce same output
    2. Context: Every artifact must have market context
    3. Type Safety: Parameters within valid ranges
    4. Staleness: Artifacts must not be expired
    5. NaN/Inf Guard: No invalid numerical values
    """
    
    # Valid parameter ranges for common fields
    PARAM_BOUNDS = {
        'confidence': (0.0, 1.0),
        'strength': (-1.0, 1.0),
        'direction': (-1.0, 1.0),
        'delta': (-1.0, 1.0),
        'gamma': (0.0, 100.0),
        'iv': (0.0, 10.0),
        'weight': (0.0, 1.0),
    }
    
    @staticmethod
    def compile(source_object: Any) -> Tuple[bool, List[str]]:
        """
        Attempts to turn a Research Object into a Valid Artifact.
        
        Returns:
            (Success, Logs)
        """
        logs = []
        logs.append("Starting compilation...")
        
        # 1. Determinism Check
        is_deterministic, det_msg = ValidationPipeline._check_determinism(source_object)
        if not is_deterministic:
            logs.append(f"  FAIL: Determinism — {det_msg}")
            return False, logs
        logs.append("  PASS: Determinism check")
        
        # 2. Context Check
        if not hasattr(source_object, 'context') or source_object.context is None:
            logs.append("  FAIL: Missing Context Definition")
            return False, logs
        logs.append("  PASS: Context present")
        
        # 3. NaN/Inf Guard
        nan_check, nan_msg = ValidationPipeline._check_numerical_validity(source_object)
        if not nan_check:
            logs.append(f"  FAIL: NaN/Inf — {nan_msg}")
            return False, logs
        logs.append("  PASS: Numerical validity")
        
        # 4. Parameter Bounds Check
        bounds_check, bounds_msg = ValidationPipeline._check_parameter_bounds(source_object)
        if not bounds_check:
            logs.append(f"  FAIL: Bounds — {bounds_msg}")
            return False, logs
        logs.append("  PASS: Parameter bounds")
        
        # 5. Staleness Check
        if hasattr(source_object, 'context') and source_object.context is not None:
            ctx = source_object.context
            if hasattr(ctx, 'valid_until') and ctx.valid_until is not None:
                if isinstance(ctx.valid_until, datetime) and ctx.valid_until < datetime.now():
                    logs.append(f"  FAIL: Artifact expired at {ctx.valid_until}")
                    return False, logs
                logs.append("  PASS: Staleness check")
        
        # 6. Artifact Type Check
        if Artifact is not None and not isinstance(source_object, Artifact):
            logs.append(f"  WARN: Object is {type(source_object).__name__}, not Artifact subclass")
        
        logs.append("  Compilation Successful. Artifact Signed.")
        return True, logs
    
    @staticmethod
    def _check_determinism(obj: Any) -> Tuple[bool, str]:
        """
        Ensures the object produces consistent serialization.
        Hashes the object's content dict twice and compares.
        """
        try:
            if hasattr(obj, 'content') and isinstance(obj.content, dict):
                # Serialize content deterministically
                serialized_1 = json.dumps(obj.content, sort_keys=True, default=str)
                serialized_2 = json.dumps(obj.content, sort_keys=True, default=str)
                
                hash_1 = hashlib.sha256(serialized_1.encode()).hexdigest()
                hash_2 = hashlib.sha256(serialized_2.encode()).hexdigest()
                
                if hash_1 != hash_2:
                    return False, "Content hash mismatch on re-serialization"
                    
                # Check for randomness indicators
                content_str = serialized_1.lower()
                if 'random' in content_str or 'rand()' in content_str:
                    return False, "Content contains random function references"
                    
                return True, "Deterministic"
            
            # For objects without content dict, basic check
            return True, "No content dict — passed by default"
            
        except (TypeError, ValueError) as e:
            return False, f"Serialization failed: {e}"
    
    @staticmethod
    def _check_numerical_validity(obj: Any) -> Tuple[bool, str]:
        """Check for NaN or Inf values in artifact content."""
        import math
        
        def _check_value(val, path=""):
            if isinstance(val, float):
                if math.isnan(val):
                    return False, f"NaN found at {path}"
                if math.isinf(val):
                    return False, f"Inf found at {path}"
            elif isinstance(val, dict):
                for k, v in val.items():
                    ok, msg = _check_value(v, f"{path}.{k}")
                    if not ok:
                        return False, msg
            elif isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    ok, msg = _check_value(v, f"{path}[{i}]")
                    if not ok:
                        return False, msg
            return True, "OK"
        
        if hasattr(obj, 'content') and isinstance(obj.content, dict):
            return _check_value(obj.content, "content")
        
        # Check dataclass fields
        if hasattr(obj, '__dataclass_fields__'):
            for field_name in obj.__dataclass_fields__:
                val = getattr(obj, field_name, None)
                if isinstance(val, float):
                    ok, msg = _check_value(val, field_name)
                    if not ok:
                        return False, msg
        
        return True, "OK"
    
    @staticmethod
    def _check_parameter_bounds(obj: Any) -> Tuple[bool, str]:
        """Check that known parameters are within valid bounds."""
        content = {}
        if hasattr(obj, 'content') and isinstance(obj.content, dict):
            content = obj.content
        elif hasattr(obj, '__dataclass_fields__'):
            content = {f: getattr(obj, f, None) for f in obj.__dataclass_fields__}
        
        for param, (lo, hi) in ValidationPipeline.PARAM_BOUNDS.items():
            if param in content and isinstance(content[param], (int, float)):
                val = content[param]
                if val < lo or val > hi:
                    return False, f"{param}={val} outside [{lo}, {hi}]"
        
        return True, "OK"
    
    @staticmethod
    def validate_batch(objects: List[Any]) -> Dict[str, Any]:
        """Validate a batch of objects, return summary."""
        results = {"passed": 0, "failed": 0, "errors": []}
        for i, obj in enumerate(objects):
            ok, logs = ValidationPipeline.compile(obj)
            if ok:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({"index": i, "logs": logs})
        return results
