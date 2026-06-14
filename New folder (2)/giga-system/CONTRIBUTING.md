# Contributing to GIGA SYSTEM

Thank you for your interest in contributing to GIGA SYSTEM! This document provides guidelines and instructions for contributing.

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn
- Credit others' work appropriately

---

## How to Contribute

### 1. Reporting Bugs

Create an issue with:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, R version)
- Relevant logs or screenshots

```markdown
**Bug Description**
Brief description of the bug

**To Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: Ubuntu 22.04
- Python: 3.11.5
- R: 4.3.1
- GIGA version: 1.0.0
```

### 2. Suggesting Features

Create an issue with:
- Clear use case description
- Why this would benefit users
- Potential implementation approach
- Any relevant examples or references

### 3. Contributing Code

#### Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/giga-system.git
cd giga-system
git remote add upstream https://github.com/ORIGINAL_OWNER/giga-system.git
```

#### Create Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

#### Make Changes

1. Write code following style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass

#### Commit

```bash
git add .
git commit -m "feat: add new feature description"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Test changes
- `refactor:` Code refactoring
- `perf:` Performance improvement

#### Push and PR

```bash
git push origin feature/your-feature-name
```

Create Pull Request on GitHub with:
- Clear description of changes
- Link to related issue
- Screenshots if UI changes
- Test results

---

## Development Guidelines

### Python Style

We follow PEP 8 with Black formatting:

```python
# Good
def calculate_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call"
) -> float:
    """
    Calculate option delta using Black-Scholes.
    
    Parameters
    ----------
    spot : float
        Current asset price
    strike : float
        Option strike price
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Annual risk-free rate
    volatility : float
        Annualized volatility
    option_type : str
        'call' or 'put'
    
    Returns
    -------
    float
        Option delta
    
    Examples
    --------
    >>> calculate_delta(100, 100, 0.25, 0.05, 0.2, 'call')
    0.5596...
    """
    d1 = _calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility)
    
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
```

### R Style

Follow tidyverse style guide:

```r
#' Calculate GARCH Volatility Forecast
#'
#' @param returns Numeric vector of returns
#' @param p GARCH p parameter
#' @param q GARCH q parameter
#' @param horizon Forecast horizon
#'
#' @return List with forecast and model
#' @export
#'
#' @examples
#' forecast_garch_volatility(rnorm(100), p = 1, q = 1, horizon = 10)
forecast_garch_volatility <- function(returns, p = 1, q = 1, horizon = 10) {
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(p, q)),
    mean.model = list(armaOrder = c(0, 0))
  )
  
  fit <- ugarchfit(spec, returns)
  forecast <- ugarchforecast(fit, n.ahead = horizon)
  
  list(
    forecast = sigma(forecast),
    model = fit
  )
}
```

### Testing

Every new feature must have tests:

```python
# tests/test_greeks.py

import pytest
import numpy as np
from giga_system.core import OptionGreeks

class TestDelta:
    """Tests for delta calculation."""
    
    def test_atm_call_delta_near_half(self):
        """ATM call delta should be approximately 0.5."""
        greeks = OptionGreeks()
        delta = greeks.delta(100, 100, 0.25, 0.05, 0.2, 'call')
        assert 0.45 < delta < 0.60
    
    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call delta should be close to 1."""
        greeks = OptionGreeks()
        delta = greeks.delta(150, 100, 0.25, 0.05, 0.2, 'call')
        assert delta > 0.95
    
    def test_put_call_delta_relationship(self):
        """Put delta = Call delta - 1."""
        greeks = OptionGreeks()
        call_delta = greeks.delta(100, 100, 0.25, 0.05, 0.2, 'call')
        put_delta = greeks.delta(100, 100, 0.25, 0.05, 0.2, 'put')
        assert np.isclose(put_delta, call_delta - 1, atol=1e-10)
    
    @pytest.mark.parametrize("spot", [80, 90, 100, 110, 120])
    def test_delta_range(self, spot):
        """Delta should always be in [0, 1] for calls."""
        greeks = OptionGreeks()
        delta = greeks.delta(spot, 100, 0.25, 0.05, 0.2, 'call')
        assert 0 <= delta <= 1
```

### Documentation

Update docs for any API changes:

```markdown
## New Function: `calculate_vanna`

### Description
Calculates the cross-derivative of option value with respect to 
spot price and volatility.

### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| spot | float | Current asset price |
| strike | float | Option strike price |
| ... | ... | ... |

### Returns
| Type | Description |
|------|-------------|
| float | Vanna value |

### Example
```python
from giga_system.core import OptionGreeks

greeks = OptionGreeks()
vanna = greeks.vanna(100, 100, 0.25, 0.05, 0.2)
print(f"Vanna: {vanna:.4f}")
```
```

---

## Performance Requirements

New code must meet performance standards:

| Operation | Maximum Time |
|-----------|--------------|
| Black-Scholes (single) | 1 μs |
| Greeks (full suite) | 5 μs |
| Monte Carlo (10K paths) | 10 ms |
| Vectorized ops (1M elements) | 100 ms |

Run benchmarks:

```bash
python scripts/benchmark_system.py
```

---

## Review Process

1. **Automated Checks**: CI must pass (tests, linting, type checks)
2. **Code Review**: At least one maintainer approval
3. **Documentation**: Updated if needed
4. **Performance**: No regressions

---

## File Count Constraint

We maintain a ~52 file limit. Before adding new files:

1. Can this be added to an existing file?
2. Is this functionality essential?
3. Can we remove/consolidate something else?

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md
- Release notes
- GitHub contributors page

---

## Questions?

- Open a Discussion on GitHub
- Tag maintainers in issues
- Check existing documentation

---

*Thank you for contributing to GIGA SYSTEM!*
