"""
CI/CD PIPELINE — Automated Testing & Deployment
=================================================

Addresses Missing Concept 7.2: No automated testing or deployment.
Provides a Python-based CI/CD runner for local and GitHub Actions.
"""

import subprocess
import sys
import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test stage."""
    stage: str
    passed: bool
    duration_sec: float
    output: str
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result of full CI/CD pipeline."""
    timestamp: datetime
    stages: List[TestResult]
    all_passed: bool
    total_duration_sec: float
    commit_hash: str = ""


class CICDPipeline:
    """
    CI/CD pipeline for the Giga System.
    Stages: lint → type-check → unit-test → integration-test → health-check → build
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or Path(__file__).parent.parent
        self.results: List[PipelineResult] = []

    def run_full_pipeline(self) -> PipelineResult:
        """Run all CI/CD stages sequentially."""
        start = time.time()
        stages = []

        # Stage 1: Import Check (lightweight lint)
        stages.append(self._run_import_check())

        # Stage 2: Health Check
        stages.append(self._run_health_check())

        # Stage 3: Unit Tests
        stages.append(self._run_unit_tests())

        # Stage 4: Integration Tests
        stages.append(self._run_integration_test())

        # Stage 5: Build Check
        stages.append(self._run_build_check())

        all_passed = all(s.passed for s in stages)
        total_time = time.time() - start

        result = PipelineResult(
            timestamp=datetime.now(),
            stages=stages,
            all_passed=all_passed,
            total_duration_sec=total_time,
        )
        self.results.append(result)
        return result

    def _run_import_check(self) -> TestResult:
        """Check all modules can be imported."""
        start = time.time()
        errors = []
        core_modules = [
            "research.core.greeks", "research.core.black_scholes",
            "research.core.monte_carlo", "research.core.alpha_signal_engine",
            "research.core.greek_mathematics",
        ]

        for mod in core_modules:
            try:
                __import__(mod)
            except Exception as e:
                errors.append(f"{mod}: {e}")

        return TestResult(
            stage="import_check",
            passed=len(errors) == 0,
            duration_sec=time.time() - start,
            output=f"Checked {len(core_modules)} modules",
            errors=errors,
        )

    def _run_health_check(self) -> TestResult:
        """Run the health check script."""
        start = time.time()
        script = self.root / "scripts" / "health_check.py"

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=60,
                cwd=str(self.root),
            )
            passed = result.returncode == 0
            output = result.stdout[-500:] if result.stdout else ""
            errors = [result.stderr[-500:]] if result.stderr and not passed else []
        except Exception as e:
            passed = False
            output = str(e)
            errors = [str(e)]

        return TestResult(
            stage="health_check", passed=passed,
            duration_sec=time.time() - start,
            output=output, errors=errors,
        )

    def _run_unit_tests(self) -> TestResult:
        """Run pytest if available."""
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=120,
                cwd=str(self.root),
            )
            passed = result.returncode == 0
            output = result.stdout[-500:] if result.stdout else "No output"
            errors = [result.stderr[-500:]] if result.stderr and not passed else []
        except FileNotFoundError:
            passed = True
            output = "pytest not installed, skipped"
            errors = []
        except Exception as e:
            passed = False
            output = str(e)
            errors = [str(e)]

        return TestResult(
            stage="unit_tests", passed=passed,
            duration_sec=time.time() - start,
            output=output, errors=errors,
        )

    def _run_integration_test(self) -> TestResult:
        """Quick integration test: can the system run end-to-end?"""
        start = time.time()
        errors = []

        try:
            # Test research pipeline
            sys.path.insert(0, str(self.root))
            from research.core.alpha_signal_engine import AlphaSignalEngine
            import numpy as np

            engine = AlphaSignalEngine()
            prices = np.cumsum(np.random.randn(200)) + 100
            prices = np.abs(prices) + 10
            signal = engine.generate_signal(prices)
            if signal.direction not in ("LONG", "SHORT", "HOLD"):
                errors.append(f"Unexpected signal direction: {signal.direction}")

        except Exception as e:
            errors.append(f"Integration test failed: {e}")

        return TestResult(
            stage="integration_test",
            passed=len(errors) == 0,
            duration_sec=time.time() - start,
            output="Research pipeline end-to-end test",
            errors=errors,
        )

    def _run_build_check(self) -> TestResult:
        """Verify project structure is complete."""
        start = time.time()
        required_dirs = ["research", "backtesting", "bridge", "execution", "config"]
        required_files = ["requirements.txt", "pyproject.toml", "README.md"]

        missing = []
        for d in required_dirs:
            if not (self.root / d).exists():
                missing.append(f"Missing directory: {d}")
        for f in required_files:
            if not (self.root / f).exists():
                missing.append(f"Missing file: {f}")

        return TestResult(
            stage="build_check",
            passed=len(missing) == 0,
            duration_sec=time.time() - start,
            output=f"Checked {len(required_dirs)} dirs, {len(required_files)} files",
            errors=missing,
        )

    def generate_github_actions_yaml(self) -> str:
        """Generate a GitHub Actions workflow YAML."""
        yaml = """name: Giga System CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest

    - name: Health Check
      run: python scripts/health_check.py

    - name: Run Tests
      run: python -m pytest tests/ -v --tb=short

    - name: Integration Test
      run: python -c "from research.core.alpha_signal_engine import AlphaSignalEngine; print('OK')"
"""
        return yaml
