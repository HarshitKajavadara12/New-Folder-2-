"""
RESEARCH-LIVE BRIDGE MANAGER — Automated TOML Generation & Performance Tracking
=================================================================================

Addresses Missing Concepts 5.1-5.5:
  5.1 — Automated TOML Generation from validated backtest results
  5.2 — TOML Versioning with rollback support
  5.3 — Live Pipeline Reading & Applying TOML Parameters
  5.4 — Research Artifact Store
  5.5 — Research-Live Performance Comparison
"""

import numpy as np
import json
import logging
import hashlib
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tomli
    TOMLI_READ = True
except ImportError:
    TOMLI_READ = False

try:
    import tomli_w
    TOMLI_WRITE = True
except ImportError:
    TOMLI_WRITE = False


@dataclass
class ResearchArtifact:
    """5.4 — A single research run artifact."""
    run_id: str
    timestamp: datetime
    strategy_name: str
    parameters: Dict
    metrics: Dict[str, float]  # sharpe, max_dd, total_return, etc.
    data_hash: str
    validated: bool = False
    promoted_to_live: bool = False


@dataclass
class PerformanceComparison:
    """5.5 — Research vs Live performance comparison."""
    timestamp: datetime
    research_sharpe: float
    live_sharpe: float
    sharpe_divergence: float
    research_max_dd: float
    live_max_dd: float
    dd_divergence: float
    alert_level: str  # "green", "yellow", "red"
    recommendation: str


class TOMLGenerator:
    """
    5.1 — Automatically generate strategies_config.toml from validated backtest.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_backtest(
        self,
        strategy_name: str,
        backtest_metrics: Dict[str, float],
        optimized_params: Dict,
        regime_params: Optional[Dict] = None,
    ) -> str:
        """
        Generate TOML config from validated backtest results.
        Returns path to generated file.
        """
        timestamp = datetime.now().isoformat()

        config = {
            "meta": {
                "generated_at": timestamp,
                "source": f"alpha_research_pipeline ({strategy_name})",
                "sharpe_ratio": backtest_metrics.get("sharpe_ratio", 0.0),
                "max_drawdown": backtest_metrics.get("max_drawdown", 0.0),
                "total_return": backtest_metrics.get("total_return", 0.0),
                "kappa_score": backtest_metrics.get("kappa", 0.0),
            },
            "execution_params": {
                "max_slippage_bps": optimized_params.get("max_slippage_bps", 5),
                "chaos_mode": False,
                "position_size_pct": optimized_params.get("position_size_pct", 2.0),
                "max_positions": optimized_params.get("max_positions", 5),
            },
        }

        # Add regime-specific parameters
        if regime_params:
            config["regime_params"] = regime_params
        else:
            config["regime_params"] = {
                "LOW_VOL": {
                    "leverage": optimized_params.get("low_vol_leverage", 2.0),
                    "kappa": backtest_metrics.get("kappa", 5.0),
                },
                "HIGH_VOL": {
                    "leverage": optimized_params.get("high_vol_leverage", 0.5),
                    "kappa": backtest_metrics.get("kappa", 5.0) * 2,
                },
            }

        # Add strategy-specific params
        config["strategy"] = {
            "name": strategy_name,
            "kappa_threshold": optimized_params.get("kappa_threshold", 5.0),
            "entropy_threshold": optimized_params.get("entropy_threshold", 3.5),
            "kelly_fraction_cap": optimized_params.get("kelly_cap", 0.25),
            "rebalance_frequency": optimized_params.get("rebalance_freq", "daily"),
        }

        # Write TOML
        output_path = self.config_dir / "strategies_config.toml"

        toml_str = self._dict_to_toml(config)
        output_path.write_text(toml_str, encoding="utf-8")
        logger.info(f"Generated TOML config at {output_path}")

        return str(output_path)

    @staticmethod
    def _dict_to_toml(d: Dict, prefix: str = "") -> str:
        """Convert dict to TOML string (minimal implementation)."""
        lines = []
        simple_items = {}
        table_items = {}

        for k, v in d.items():
            if isinstance(v, dict):
                table_items[k] = v
            else:
                simple_items[k] = v

        # Write simple key-value pairs first
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in simple_items.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f'{k} = "{v}"')

        # Write nested tables
        for k, v in table_items.items():
            full_key = f"{prefix}.{k}" if prefix else k
            lines.append("")
            lines.append(TOMLGenerator._dict_to_toml(v, full_key))

        return "\n".join(lines)


class TOMLVersionManager:
    """
    5.2 — TOML Versioning with rollback.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self.versions_dir = self.config_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def save_version(self, config_path: Optional[Path] = None, label: str = "") -> str:
        """Save current config as a versioned snapshot."""
        config_path = config_path or self.config_dir / "strategies_config.toml"
        if not config_path.exists():
            logger.warning("No config to version")
            return ""

        content = config_path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"v_{timestamp}_{content_hash}"
        if label:
            version_name += f"_{label}"

        version_path = self.versions_dir / f"{version_name}.toml"
        shutil.copy2(config_path, version_path)

        # Save metadata
        meta = {
            "version": version_name,
            "timestamp": timestamp,
            "hash": content_hash,
            "label": label,
            "source_path": str(config_path),
        }
        meta_path = self.versions_dir / f"{version_name}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info(f"Saved config version: {version_name}")
        return version_name

    def list_versions(self) -> List[Dict]:
        """List all saved config versions."""
        versions = []
        for meta_file in sorted(self.versions_dir.glob("*.json")):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                versions.append(meta)
            except Exception:
                continue
        return versions

    def rollback(self, version_name: str) -> bool:
        """Rollback to a previous version."""
        version_path = self.versions_dir / f"{version_name}.toml"
        if not version_path.exists():
            logger.error(f"Version {version_name} not found")
            return False

        target = self.config_dir / "strategies_config.toml"
        # Save current as backup first
        self.save_version(target, label="pre_rollback")
        shutil.copy2(version_path, target)
        logger.info(f"Rolled back to {version_name}")
        return True


class TOMLParameterReader:
    """
    5.3 — Live pipeline reads and applies TOML parameters.
    Ensures live pipeline actually uses config, not defaults.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "strategies_config.toml"
        self._params: Dict = {}
        self._loaded = False

    def load(self) -> Dict:
        """Load and parse TOML config."""
        if not self.config_path.exists():
            logger.warning(f"Config not found at {self.config_path}, using defaults")
            self._params = self._defaults()
            self._loaded = True
            return self._params

        content = self.config_path.read_text(encoding="utf-8")

        # Parse TOML manually (minimal parser for our format)
        self._params = self._parse_toml(content)
        self._loaded = True
        logger.info(f"Loaded {len(self._params)} config sections from {self.config_path}")
        return self._params

    def get(self, key: str, default=None):
        """Get a config parameter with dot-notation: 'execution_params.max_slippage_bps'"""
        if not self._loaded:
            self.load()

        parts = key.split(".")
        current = self._params
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def verify_params_applied(self, live_params: Dict) -> Dict:
        """
        Verify that live pipeline is actually using TOML params, not defaults.
        Returns discrepancies.
        """
        if not self._loaded:
            self.load()

        discrepancies = {}
        config_flat = self._flatten(self._params)
        live_flat = self._flatten(live_params)

        for key in config_flat:
            if key in live_flat:
                if config_flat[key] != live_flat[key]:
                    discrepancies[key] = {
                        "config_value": config_flat[key],
                        "live_value": live_flat[key],
                        "status": "MISMATCH",
                    }
            else:
                discrepancies[key] = {
                    "config_value": config_flat[key],
                    "status": "NOT_IN_LIVE",
                }

        return {
            "discrepancies": discrepancies,
            "n_mismatch": len(discrepancies),
            "all_applied": len(discrepancies) == 0,
        }

    @staticmethod
    def _defaults() -> Dict:
        return {
            "execution_params": {"max_slippage_bps": 5, "chaos_mode": False},
            "regime_params": {"LOW_VOL": {"leverage": 2.0}, "HIGH_VOL": {"leverage": 0.5}},
            "strategy": {"kappa_threshold": 5.0, "entropy_threshold": 3.5},
        }

    @staticmethod
    def _parse_toml(content: str) -> Dict:
        """Minimal TOML parser for our config format."""
        result = {}
        current_section = result

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section_path = line[1:-1].split(".")
                current_section = result
                for part in section_path:
                    if part not in current_section:
                        current_section[part] = {}
                    current_section = current_section[part]
            elif "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                # Type conversion
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                else:
                    try:
                        value = float(value)
                        if value == int(value):
                            value = int(value)
                    except ValueError:
                        pass
                current_section[key] = value

        return result

    @staticmethod
    def _flatten(d: Dict, prefix: str = "") -> Dict:
        items = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(TOMLParameterReader._flatten(v, full_key))
            else:
                items[full_key] = v
        return items


class ResearchArtifactStore:
    """
    5.4 — Store research artifacts (backtest results, param sweeps, validation).
    """

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or Path(__file__).parent.parent.parent / "artifacts" / "research"
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: ResearchArtifact) -> str:
        """Save a research artifact."""
        data = {
            "run_id": artifact.run_id,
            "timestamp": artifact.timestamp.isoformat(),
            "strategy_name": artifact.strategy_name,
            "parameters": artifact.parameters,
            "metrics": artifact.metrics,
            "data_hash": artifact.data_hash,
            "validated": artifact.validated,
            "promoted_to_live": artifact.promoted_to_live,
        }

        path = self.store_dir / f"{artifact.run_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"Saved research artifact: {artifact.run_id}")
        return str(path)

    def load(self, run_id: str) -> Optional[ResearchArtifact]:
        """Load a research artifact by run_id."""
        path = self.store_dir / f"{run_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        return ResearchArtifact(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            strategy_name=data["strategy_name"],
            parameters=data["parameters"],
            metrics=data["metrics"],
            data_hash=data["data_hash"],
            validated=data.get("validated", False),
            promoted_to_live=data.get("promoted_to_live", False),
        )

    def list_artifacts(self, strategy_name: Optional[str] = None) -> List[Dict]:
        """List all artifacts, optionally filtered by strategy."""
        artifacts = []
        for f in sorted(self.store_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if strategy_name and data.get("strategy_name") != strategy_name:
                    continue
                artifacts.append(data)
            except Exception:
                continue
        return artifacts

    def find_best(self, metric: str = "sharpe_ratio", top_n: int = 5) -> List[Dict]:
        """Find top N artifacts by a given metric."""
        artifacts = self.list_artifacts()
        artifacts.sort(key=lambda x: x.get("metrics", {}).get(metric, -999), reverse=True)
        return artifacts[:top_n]


class ResearchLiveComparator:
    """
    5.5 — Compare research backtest results with live performance.
    Alert on significant divergence.
    """

    def __init__(self, divergence_threshold: float = 0.3):
        self.divergence_threshold = divergence_threshold
        self.comparisons: List[PerformanceComparison] = []

    def compare(
        self,
        research_metrics: Dict[str, float],
        live_metrics: Dict[str, float],
    ) -> PerformanceComparison:
        """
        Compare research vs live performance.
        """
        r_sharpe = research_metrics.get("sharpe_ratio", 0.0)
        l_sharpe = live_metrics.get("sharpe_ratio", 0.0)
        sharpe_div = abs(r_sharpe - l_sharpe) / (abs(r_sharpe) + 1e-15)

        r_dd = research_metrics.get("max_drawdown", 0.0)
        l_dd = live_metrics.get("max_drawdown", 0.0)
        dd_div = abs(r_dd - l_dd) / (abs(r_dd) + 1e-15) if abs(r_dd) > 0 else 0.0

        # Determine alert level
        if sharpe_div > self.divergence_threshold * 2 or l_dd > r_dd * 2:
            alert = "red"
            rec = "STOP LIVE TRADING — significant performance divergence"
        elif sharpe_div > self.divergence_threshold or l_dd > r_dd * 1.5:
            alert = "yellow"
            rec = "REDUCE POSITION SIZE — moderate divergence from research"
        else:
            alert = "green"
            rec = "CONTINUE — live performance within expected range"

        comp = PerformanceComparison(
            timestamp=datetime.now(),
            research_sharpe=r_sharpe, live_sharpe=l_sharpe,
            sharpe_divergence=float(sharpe_div),
            research_max_dd=r_dd, live_max_dd=l_dd,
            dd_divergence=float(dd_div),
            alert_level=alert, recommendation=rec,
        )
        self.comparisons.append(comp)
        return comp
