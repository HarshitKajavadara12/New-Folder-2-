# =============================================================================
# GIGA SYSTEM - Performance Analytics (R)
# Risk-Adjusted Returns, Attribution, Rolling Analysis
# =============================================================================

library(PerformanceAnalytics)
library(xts)
library(zoo)

# =============================================================================
# RETURN METRICS
# =============================================================================

#' Calculate comprehensive return statistics
#'
#' @param returns Numeric vector or xts of returns
#' @param risk_free Risk-free rate (annualized)
#' @param periods_per_year Number of periods per year (252 for daily)
#' @return List with all return metrics
calculate_return_metrics <- function(returns, risk_free = 0.02, periods_per_year = 252) {
  # Handle both numeric and xts
  if (!is.xts(returns)) {
    returns <- as.numeric(returns)
  }
  
  n <- length(returns)
  
  # Basic statistics
  mean_return <- mean(returns) * periods_per_year
  total_return <- prod(1 + returns) - 1
  
  # CAGR (Compound Annual Growth Rate)
  years <- n / periods_per_year
  cagr <- (1 + total_return)^(1/years) - 1
  
  # Volatility
  volatility <- sd(returns) * sqrt(periods_per_year)
  downside_vol <- sqrt(mean(pmin(returns, 0)^2)) * sqrt(periods_per_year)
  
  # Higher moments
  skewness <- mean((returns - mean(returns))^3) / sd(returns)^3
  kurtosis <- mean((returns - mean(returns))^4) / sd(returns)^4 - 3  # Excess
  
  list(
    # Return metrics
    mean_return = mean_return,
    total_return = total_return,
    cagr = cagr,
    
    # Risk metrics
    volatility = volatility,
    downside_volatility = downside_vol,
    
    # Distribution
    skewness = skewness,
    kurtosis = kurtosis,
    
    # Basic stats
    n_observations = n,
    years = years,
    positive_periods = sum(returns > 0) / n,
    negative_periods = sum(returns < 0) / n
  )
}


# =============================================================================
# RISK-ADJUSTED RETURN RATIOS
# =============================================================================

#' Calculate Sharpe Ratio
#'
#' Sharpe Ratio = (R_p - R_f) / σ_p
#'
#' Interpretation:
#' - < 1: Suboptimal
#' - 1-2: Acceptable
#' - 2-3: Very Good
#' - > 3: Excellent
#'
#' @param returns Numeric vector of returns
#' @param risk_free Annualized risk-free rate
#' @param periods_per_year Periods per year
#' @return Annualized Sharpe ratio
sharpe_ratio <- function(returns, risk_free = 0.02, periods_per_year = 252) {
  excess_returns <- returns - risk_free/periods_per_year
  
  mean_excess <- mean(excess_returns) * periods_per_year
  vol <- sd(returns) * sqrt(periods_per_year)
  
  mean_excess / vol
}


#' Calculate Sortino Ratio
#'
#' Sortino Ratio = (R_p - R_f) / σ_downside
#'
#' Only penalizes downside volatility, not upside
#' Better for asymmetric return distributions
#'
#' @param returns Numeric vector of returns
#' @param mar Minimum Acceptable Return (default = risk_free)
#' @param risk_free Annualized risk-free rate
#' @param periods_per_year Periods per year
#' @return Annualized Sortino ratio
sortino_ratio <- function(returns, mar = NULL, risk_free = 0.02, periods_per_year = 252) {
  if (is.null(mar)) {
    mar <- risk_free / periods_per_year
  }
  
  excess_returns <- returns - mar
  mean_excess <- mean(excess_returns) * periods_per_year
  
  # Downside deviation (only negative deviations)
  downside <- pmin(returns - mar, 0)
  downside_dev <- sqrt(mean(downside^2)) * sqrt(periods_per_year)
  
  mean_excess / downside_dev
}


#' Calculate Calmar Ratio
#'
#' Calmar Ratio = CAGR / Max Drawdown
#'
#' Measures return per unit of maximum drawdown risk
#'
#' @param returns Numeric vector of returns
#' @param periods_per_year Periods per year
#' @return Calmar ratio
calmar_ratio <- function(returns, periods_per_year = 252) {
  # CAGR
  total_return <- prod(1 + returns) - 1
  years <- length(returns) / periods_per_year
  cagr <- (1 + total_return)^(1/years) - 1
  
  # Max drawdown
  cum_returns <- cumprod(1 + returns)
  running_max <- cummax(cum_returns)
  drawdowns <- (cum_returns - running_max) / running_max
  max_dd <- abs(min(drawdowns))
  
  cagr / max_dd
}


#' Calculate Information Ratio
#'
#' Information Ratio = (R_p - R_b) / σ_tracking
#'
#' Measures excess return over benchmark per unit of tracking error
#'
#' @param returns Portfolio returns
#' @param benchmark Benchmark returns
#' @param periods_per_year Periods per year
#' @return Annualized Information ratio
information_ratio <- function(returns, benchmark, periods_per_year = 252) {
  active_returns <- returns - benchmark
  
  mean_active <- mean(active_returns) * periods_per_year
  tracking_error <- sd(active_returns) * sqrt(periods_per_year)
  
  mean_active / tracking_error
}


#' Calculate Omega Ratio
#'
#' Omega Ratio = Σ(R - threshold)⁺ / Σ(threshold - R)⁺
#'
#' Ratio of gains above threshold to losses below threshold
#' Captures entire distribution, not just first two moments
#'
#' @param returns Numeric vector of returns
#' @param threshold Threshold return (default 0)
#' @return Omega ratio
omega_ratio <- function(returns, threshold = 0) {
  gains <- sum(pmax(returns - threshold, 0))
  losses <- sum(pmax(threshold - returns, 0))
  
  if (losses == 0) return(Inf)
  gains / losses
}


#' Calculate Treynor Ratio
#'
#' Treynor Ratio = (R_p - R_f) / β
#'
#' Excess return per unit of systematic risk (beta)
#'
#' @param returns Portfolio returns
#' @param market_returns Market/benchmark returns
#' @param risk_free Annualized risk-free rate
#' @param periods_per_year Periods per year
#' @return Treynor ratio
treynor_ratio <- function(returns, market_returns, risk_free = 0.02, periods_per_year = 252) {
  # Calculate beta
  beta <- cov(returns, market_returns) / var(market_returns)
  
  # Excess return
  excess_return <- mean(returns - risk_free/periods_per_year) * periods_per_year
  
  excess_return / beta
}


# =============================================================================
# DRAWDOWN ANALYSIS
# =============================================================================

#' Comprehensive drawdown analysis
#'
#' @param returns Numeric vector of returns
#' @param top_n Number of top drawdowns to report
#' @return List with drawdown statistics
drawdown_analysis <- function(returns, top_n = 5) {
  # Calculate cumulative returns
  cum_returns <- cumprod(1 + returns)
  
  # Running maximum
  running_max <- cummax(cum_returns)
  
  # Drawdown series
  drawdowns <- (cum_returns - running_max) / running_max
  
  # Find drawdown periods
  in_drawdown <- drawdowns < 0
  
  # Find drawdown start/end points
  dd_start <- which(diff(c(FALSE, in_drawdown)) == 1)
  dd_end <- which(diff(c(in_drawdown, FALSE)) == -1)
  
  # Handle edge cases
  if (length(dd_start) == 0) {
    return(list(
      max_drawdown = 0,
      avg_drawdown = 0,
      drawdown_periods = 0,
      message = "No drawdowns detected"
    ))
  }
  
  # Calculate drawdown statistics for each period
  dd_stats <- data.frame(
    start = dd_start,
    end = dd_end,
    length = dd_end - dd_start + 1,
    depth = sapply(1:length(dd_start), function(i) {
      min(drawdowns[dd_start[i]:dd_end[i]])
    })
  )
  
  # Recovery time (to peak)
  dd_stats$recovery <- sapply(1:nrow(dd_stats), function(i) {
    if (dd_stats$end[i] < length(drawdowns)) {
      future <- which(cum_returns[(dd_stats$end[i]+1):length(cum_returns)] >= 
                        running_max[dd_stats$start[i]])
      if (length(future) > 0) future[1] else NA
    } else NA
  })
  
  # Sort by depth
  dd_stats <- dd_stats[order(dd_stats$depth), ]
  
  list(
    max_drawdown = abs(min(drawdowns)),
    avg_drawdown = abs(mean(drawdowns[drawdowns < 0])),
    max_drawdown_length = max(dd_stats$length),
    avg_drawdown_length = mean(dd_stats$length),
    n_drawdowns = nrow(dd_stats),
    time_in_drawdown = sum(in_drawdown) / length(in_drawdown),
    drawdown_series = as.numeric(drawdowns),
    top_drawdowns = head(dd_stats, top_n),
    current_drawdown = if(tail(in_drawdown, 1)) abs(tail(drawdowns, 1)) else 0
  )
}


# =============================================================================
# ROLLING ANALYSIS
# =============================================================================

#' Calculate rolling performance metrics
#'
#' @param returns Numeric vector of returns
#' @param window Rolling window size (default 252 = 1 year)
#' @param risk_free Annualized risk-free rate
#' @return List with rolling metrics
rolling_performance <- function(returns, window = 252, risk_free = 0.02) {
  n <- length(returns)
  
  if (n < window) {
    return(list(error = "Insufficient data for rolling window"))
  }
  
  # Pre-allocate
  n_rolls <- n - window + 1
  rolling_return <- numeric(n_rolls)
  rolling_vol <- numeric(n_rolls)
  rolling_sharpe <- numeric(n_rolls)
  rolling_sortino <- numeric(n_rolls)
  rolling_max_dd <- numeric(n_rolls)
  
  for (i in 1:n_rolls) {
    window_returns <- returns[i:(i + window - 1)]
    
    # Annualized return
    rolling_return[i] <- mean(window_returns) * 252
    
    # Annualized volatility
    rolling_vol[i] <- sd(window_returns) * sqrt(252)
    
    # Sharpe
    rolling_sharpe[i] <- (rolling_return[i] - risk_free) / rolling_vol[i]
    
    # Sortino
    downside <- pmin(window_returns - risk_free/252, 0)
    downside_dev <- sqrt(mean(downside^2)) * sqrt(252)
    rolling_sortino[i] <- (rolling_return[i] - risk_free) / downside_dev
    
    # Max drawdown
    cum <- cumprod(1 + window_returns)
    dd <- (cum - cummax(cum)) / cummax(cum)
    rolling_max_dd[i] <- abs(min(dd))
  }
  
  list(
    rolling_return = rolling_return,
    rolling_volatility = rolling_vol,
    rolling_sharpe = rolling_sharpe,
    rolling_sortino = rolling_sortino,
    rolling_max_drawdown = rolling_max_dd,
    window = window,
    dates = (window:n)
  )
}


#' Calculate rolling beta
#'
#' @param returns Portfolio returns
#' @param market_returns Market/benchmark returns
#' @param window Rolling window size
#' @return Numeric vector of rolling betas
rolling_beta <- function(returns, market_returns, window = 252) {
  n <- length(returns)
  n_rolls <- n - window + 1
  
  betas <- numeric(n_rolls)
  
  for (i in 1:n_rolls) {
    port_window <- returns[i:(i + window - 1)]
    mkt_window <- market_returns[i:(i + window - 1)]
    
    betas[i] <- cov(port_window, mkt_window) / var(mkt_window)
  }
  
  betas
}


# =============================================================================
# PERFORMANCE ATTRIBUTION
# =============================================================================

#' Brinson-Fachler Performance Attribution
#'
#' Decomposes portfolio return vs benchmark into:
#' - Allocation effect: Over/underweighting sectors
#' - Selection effect: Stock picking within sectors
#' - Interaction effect: Combined effect
#'
#' @param portfolio_weights Portfolio sector weights
#' @param benchmark_weights Benchmark sector weights
#' @param portfolio_returns Portfolio sector returns
#' @param benchmark_returns Benchmark sector returns
#' @return List with attribution effects
brinson_attribution <- function(portfolio_weights, benchmark_weights,
                                portfolio_returns, benchmark_returns) {
  n_sectors <- length(portfolio_weights)
  
  # Total returns
  port_total <- sum(portfolio_weights * portfolio_returns)
  bench_total <- sum(benchmark_weights * benchmark_returns)
  
  # Allocation effect: (w_p - w_b) * (r_b - R_b)
  allocation <- (portfolio_weights - benchmark_weights) * 
    (benchmark_returns - bench_total)
  
  # Selection effect: w_b * (r_p - r_b)
  selection <- benchmark_weights * (portfolio_returns - benchmark_returns)
  
  # Interaction effect: (w_p - w_b) * (r_p - r_b)
  interaction <- (portfolio_weights - benchmark_weights) * 
    (portfolio_returns - benchmark_returns)
  
  list(
    portfolio_return = port_total,
    benchmark_return = bench_total,
    active_return = port_total - bench_total,
    allocation_effect = sum(allocation),
    selection_effect = sum(selection),
    interaction_effect = sum(interaction),
    sector_allocation = as.numeric(allocation),
    sector_selection = as.numeric(selection),
    sector_interaction = as.numeric(interaction)
  )
}


#' Factor Attribution using regression
#'
#' Decomposes returns into factor exposures
#'
#' @param returns Portfolio returns
#' @param factors Matrix of factor returns (each column is a factor)
#' @return List with factor attribution results
factor_attribution <- function(returns, factors) {
  # Ensure factors is a matrix
  if (!is.matrix(factors)) {
    factors <- as.matrix(factors)
  }
  
  # Regression: R = α + Σ β_i * F_i + ε
  model <- lm(returns ~ factors)
  
  # Extract results
  coefs <- coef(model)
  alpha <- coefs[1]  # Intercept
  betas <- coefs[-1]  # Factor loadings
  
  # Factor contributions
  factor_returns <- colMeans(factors)
  contributions <- betas * factor_returns
  
  # Residual (idiosyncratic) return
  residual <- mean(returns) - alpha - sum(contributions)
  
  # R-squared (explanatory power)
  r_squared <- summary(model)$r.squared
  
  list(
    alpha = alpha * 252,  # Annualized
    factor_loadings = as.numeric(betas),
    factor_contributions = as.numeric(contributions) * 252,
    residual_return = residual * 252,
    r_squared = r_squared,
    explained_variance = r_squared,
    factor_names = colnames(factors),
    model_summary = capture.output(summary(model))
  )
}


# =============================================================================
# TAIL RISK METRICS
# =============================================================================

#' Calculate tail risk metrics
#'
#' @param returns Numeric vector of returns
#' @param confidence_level Confidence level (default 0.95)
#' @return List with tail risk metrics
tail_risk_metrics <- function(returns, confidence_level = 0.95) {
  # VaR
  var_hist <- quantile(returns, 1 - confidence_level)
  
  # CVaR (Expected Shortfall)
  cvar <- mean(returns[returns <= var_hist])
  
  # Tail ratio (gain/loss beyond VaR)
  upper_var <- quantile(returns, confidence_level)
  gains_beyond <- returns[returns >= upper_var]
  losses_beyond <- returns[returns <= var_hist]
  
  tail_ratio <- if(length(losses_beyond) > 0 && length(gains_beyond) > 0) {
    mean(gains_beyond) / abs(mean(losses_beyond))
  } else NA
  
  # Pain index (average drawdown)
  cum <- cumprod(1 + returns)
  dd <- (cum - cummax(cum)) / cummax(cum)
  pain_index <- mean(abs(dd))
  
  # Ulcer index (RMS of drawdown)
  ulcer_index <- sqrt(mean(dd^2))
  
  list(
    var = as.numeric(var_hist),
    cvar = as.numeric(cvar),
    tail_ratio = tail_ratio,
    pain_index = pain_index,
    ulcer_index = ulcer_index,
    confidence_level = confidence_level,
    worst_return = min(returns),
    best_return = max(returns)
  )
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if (FALSE) {  # Set to TRUE to run examples
  set.seed(42)
  
  # Generate sample returns
  n <- 500
  returns <- rnorm(n, mean = 0.0004, sd = 0.015)
  benchmark <- rnorm(n, mean = 0.0003, sd = 0.012)
  
  # Return metrics
  cat("Return Metrics:\n")
  metrics <- calculate_return_metrics(returns)
  cat("Annualized Return:", round(metrics$mean_return * 100, 2), "%\n")
  cat("Volatility:", round(metrics$volatility * 100, 2), "%\n")
  cat("Skewness:", round(metrics$skewness, 2), "\n")
  
  # Risk-adjusted ratios
  cat("\nRisk-Adjusted Ratios:\n")
  cat("Sharpe Ratio:", round(sharpe_ratio(returns), 2), "\n")
  cat("Sortino Ratio:", round(sortino_ratio(returns), 2), "\n")
  cat("Calmar Ratio:", round(calmar_ratio(returns), 2), "\n")
  cat("Information Ratio:", round(information_ratio(returns, benchmark), 2), "\n")
  cat("Omega Ratio:", round(omega_ratio(returns), 2), "\n")
  
  # Drawdown analysis
  cat("\nDrawdown Analysis:\n")
  dd <- drawdown_analysis(returns)
  cat("Max Drawdown:", round(dd$max_drawdown * 100, 2), "%\n")
  cat("Avg Drawdown:", round(dd$avg_drawdown * 100, 2), "%\n")
  cat("Time in Drawdown:", round(dd$time_in_drawdown * 100, 1), "%\n")
  
  # Tail risk
  cat("\nTail Risk (95% confidence):\n")
  tail <- tail_risk_metrics(returns)
  cat("VaR:", round(tail$var * 100, 2), "%\n")
  cat("CVaR:", round(tail$cvar * 100, 2), "%\n")
  cat("Pain Index:", round(tail$pain_index * 100, 2), "%\n")
}
