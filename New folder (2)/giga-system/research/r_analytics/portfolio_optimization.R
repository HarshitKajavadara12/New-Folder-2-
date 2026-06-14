# =============================================================================
# GIGA SYSTEM - Portfolio Optimization (R)
# Mean-Variance, Black-Litterman, Risk Parity
# =============================================================================

library(PortfolioAnalytics)
library(quadprog)
library(MASS)

# =============================================================================
# MARKOWITZ MEAN-VARIANCE OPTIMIZATION
# =============================================================================

#' Classical Mean-Variance Optimization
#'
#' Markowitz (1952) Portfolio Theory:
#' 
#' Minimize: w' Σ w  (portfolio variance)
#' Subject to: w' μ = μ_target  (target return)
#'            Σ w_i = 1  (fully invested)
#'            w_i ≥ 0  (no short selling, optional)
#'
#' Efficient Frontier: Set of portfolios with maximum return for given risk
#'
#' @param returns Matrix of asset returns (T x N)
#' @param target_return Target portfolio return (if NULL, optimize Sharpe)
#' @param risk_free Risk-free rate for Sharpe ratio
#' @param short_selling Allow short selling (default FALSE)
#' @return List with optimal weights and portfolio statistics
mean_variance_optimize <- function(returns, target_return = NULL, 
                                   risk_free = 0.02/252, short_selling = FALSE) {
  # Calculate expected returns and covariance
  n_assets <- ncol(returns)
  mu <- colMeans(returns)  # Expected returns
  Sigma <- cov(returns)    # Covariance matrix
  
  if (is.null(target_return)) {
    # Maximize Sharpe ratio
    # This is equivalent to minimizing (w'Σw) / (w'μ - r_f)²
    # Transform to QP: minimize w'Σw subject to w'(μ - r_f) = 1
    
    excess_returns <- mu - risk_free
    
    # Quadratic programming setup
    Dmat <- 2 * Sigma
    dvec <- rep(0, n_assets)
    
    # Constraints: sum(w) = 1, w'excess = 1 (scaled)
    if (short_selling) {
      Amat <- cbind(rep(1, n_assets), excess_returns)
      bvec <- c(1, 1)
      meq <- 2
    } else {
      Amat <- cbind(rep(1, n_assets), excess_returns, diag(n_assets))
      bvec <- c(1, 1, rep(0, n_assets))
      meq <- 2
    }
    
    # Solve
    tryCatch({
      sol <- solve.QP(Dmat, dvec, Amat, bvec, meq = meq)
      w <- sol$solution
      # Normalize to sum to 1
      w <- w / sum(w)
    }, error = function(e) {
      # Fallback to equal weight
      w <- rep(1/n_assets, n_assets)
    })
    
  } else {
    # Target return optimization
    Dmat <- 2 * Sigma
    dvec <- rep(0, n_assets)
    
    if (short_selling) {
      Amat <- cbind(rep(1, n_assets), mu)
      bvec <- c(1, target_return)
      meq <- 2
    } else {
      Amat <- cbind(rep(1, n_assets), mu, diag(n_assets))
      bvec <- c(1, target_return, rep(0, n_assets))
      meq <- 2
    }
    
    sol <- solve.QP(Dmat, dvec, Amat, bvec, meq = meq)
    w <- sol$solution
  }
  
  # Portfolio statistics
  port_return <- sum(w * mu) * 252  # Annualized
  port_volatility <- sqrt(t(w) %*% Sigma %*% w * 252)
  sharpe_ratio <- (port_return - risk_free * 252) / port_volatility
  
  list(
    weights = as.numeric(w),
    expected_return = as.numeric(port_return),
    volatility = as.numeric(port_volatility),
    sharpe_ratio = as.numeric(sharpe_ratio),
    asset_names = colnames(returns),
    covariance_matrix = Sigma,
    expected_returns = mu * 252
  )
}


#' Generate efficient frontier
#'
#' @param returns Matrix of asset returns
#' @param n_portfolios Number of portfolios on frontier
#' @param short_selling Allow short selling
#' @return Data frame with frontier portfolios
efficient_frontier <- function(returns, n_portfolios = 50, short_selling = FALSE) {
  mu <- colMeans(returns)
  
  # Range of target returns
  min_return <- min(mu)
  max_return <- max(mu)
  target_returns <- seq(min_return * 1.1, max_return * 0.9, length.out = n_portfolios)
  
  frontier <- data.frame(
    target_return = numeric(n_portfolios),
    portfolio_return = numeric(n_portfolios),
    portfolio_volatility = numeric(n_portfolios),
    sharpe_ratio = numeric(n_portfolios)
  )
  
  weights_matrix <- matrix(0, nrow = n_portfolios, ncol = ncol(returns))
  
  for (i in 1:n_portfolios) {
    tryCatch({
      opt <- mean_variance_optimize(returns, target_return = target_returns[i], 
                                    short_selling = short_selling)
      frontier$target_return[i] <- target_returns[i] * 252
      frontier$portfolio_return[i] <- opt$expected_return
      frontier$portfolio_volatility[i] <- opt$volatility
      frontier$sharpe_ratio[i] <- opt$sharpe_ratio
      weights_matrix[i, ] <- opt$weights
    }, error = function(e) {
      frontier$target_return[i] <- NA
    })
  }
  
  # Remove failed optimizations
  valid <- !is.na(frontier$target_return)
  
  list(
    frontier = frontier[valid, ],
    weights = weights_matrix[valid, ],
    asset_names = colnames(returns)
  )
}


# =============================================================================
# BLACK-LITTERMAN MODEL
# =============================================================================

#' Black-Litterman portfolio optimization
#'
#' Combines market equilibrium with investor views:
#' 
#' Prior: π = δ * Σ * w_mkt  (equilibrium excess returns)
#' Views: Q = P * μ + ε, where ε ~ N(0, Ω)
#' 
#' Posterior: E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]
#'
#' @param returns Matrix of asset returns
#' @param market_caps Market capitalizations (for equilibrium weights)
#' @param views_P Pick matrix (K x N) - which assets the views are about
#' @param views_Q View values (K x 1) - expected returns from views
#' @param views_confidence Confidence in views (0-1)
#' @param tau Scaling parameter (default 0.05)
#' @param delta Risk aversion coefficient
#' @return List with Black-Litterman optimal weights
black_litterman <- function(returns, market_caps = NULL, views_P = NULL, 
                            views_Q = NULL, views_confidence = 0.5,
                            tau = 0.05, delta = 2.5) {
  
  n_assets <- ncol(returns)
  Sigma <- cov(returns)
  
  # Market equilibrium weights (if not provided, use equal weight)
  if (is.null(market_caps)) {
    w_mkt <- rep(1/n_assets, n_assets)
  } else {
    w_mkt <- market_caps / sum(market_caps)
  }
  
  # Implied equilibrium returns (reverse optimization)
  pi_eq <- delta * Sigma %*% w_mkt
  
  if (is.null(views_P) || is.null(views_Q)) {
    # No views - return equilibrium portfolio
    return(list(
      weights = as.numeric(w_mkt),
      expected_returns = as.numeric(pi_eq) * 252,
      volatility = sqrt(t(w_mkt) %*% Sigma %*% w_mkt * 252),
      type = "Equilibrium (No Views)"
    ))
  }
  
  # View uncertainty matrix (Ω)
  # Higher confidence = lower uncertainty
  k <- nrow(views_P)  # Number of views
  omega <- diag(k) * (1 - views_confidence) / views_confidence
  
  # Black-Litterman posterior
  # M = τΣ
  M <- tau * Sigma
  M_inv <- solve(M)
  omega_inv <- solve(omega)
  
  # Posterior precision
  posterior_precision <- M_inv + t(views_P) %*% omega_inv %*% views_P
  
  # Posterior mean
  posterior_mean <- solve(posterior_precision) %*% 
    (M_inv %*% pi_eq + t(views_P) %*% omega_inv %*% views_Q)
  
  # Posterior covariance
  posterior_cov <- solve(posterior_precision) + Sigma
  
  # Optimize using posterior estimates
  # Maximize: w'μ_post - (δ/2) * w'Σ_post * w
  w_bl <- solve(delta * posterior_cov) %*% posterior_mean
  w_bl <- as.numeric(w_bl)
  
  # Normalize weights
  w_bl <- w_bl / sum(w_bl)
  
  # Ensure non-negative (simple projection)
  w_bl[w_bl < 0] <- 0
  w_bl <- w_bl / sum(w_bl)
  
  list(
    weights = w_bl,
    expected_returns = as.numeric(posterior_mean) * 252,
    posterior_covariance = posterior_cov,
    equilibrium_returns = as.numeric(pi_eq) * 252,
    type = "Black-Litterman",
    n_views = k
  )
}


# =============================================================================
# RISK PARITY
# =============================================================================

#' Risk Parity (Equal Risk Contribution) Portfolio
#'
#' Each asset contributes equally to total portfolio risk:
#' RC_i = w_i * (Σw)_i / σ_p = σ_p / N
#'
#' Risk Contribution: RC_i = w_i * ∂σ_p/∂w_i = w_i * (Σw)_i / σ_p
#'
#' @param returns Matrix of asset returns
#' @param target_risk Target risk contributions (default equal)
#' @return List with risk parity weights
risk_parity <- function(returns, target_risk = NULL) {
  n_assets <- ncol(returns)
  Sigma <- cov(returns)
  
  if (is.null(target_risk)) {
    target_risk <- rep(1/n_assets, n_assets)
  }
  
  # Objective: minimize sum of (RC_i/RC_total - target_i)^2
  objective <- function(w) {
    w <- abs(w)  # Ensure positive
    w <- w / sum(w)  # Normalize
    
    port_var <- t(w) %*% Sigma %*% w
    port_vol <- sqrt(port_var)
    
    # Marginal risk contribution
    mrc <- Sigma %*% w / as.numeric(port_vol)
    
    # Risk contribution
    rc <- w * mrc
    rc_pct <- rc / sum(rc)
    
    # SSE from target
    sum((rc_pct - target_risk)^2)
  }
  
  # Optimize
  init_w <- rep(1/n_assets, n_assets)
  opt <- optim(init_w, objective, method = "L-BFGS-B",
               lower = rep(0.001, n_assets), upper = rep(1, n_assets))
  
  # Final weights
  w <- abs(opt$par)
  w <- w / sum(w)
  
  # Calculate risk contributions
  port_vol <- sqrt(t(w) %*% Sigma %*% w)
  mrc <- Sigma %*% w / as.numeric(port_vol)
  rc <- w * mrc
  rc_pct <- rc / sum(rc)
  
  list(
    weights = as.numeric(w),
    risk_contributions = as.numeric(rc),
    risk_contribution_pct = as.numeric(rc_pct),
    portfolio_volatility = as.numeric(port_vol) * sqrt(252),
    target_risk = target_risk,
    convergence = opt$convergence == 0
  )
}


# =============================================================================
# MINIMUM VARIANCE PORTFOLIO
# =============================================================================

#' Global Minimum Variance Portfolio
#'
#' Minimizes portfolio variance without return constraint
#' Useful when return estimates are unreliable
#'
#' @param returns Matrix of asset returns
#' @param short_selling Allow short selling
#' @return List with minimum variance weights
minimum_variance <- function(returns, short_selling = FALSE) {
  n_assets <- ncol(returns)
  Sigma <- cov(returns)
  
  # QP: minimize w'Σw subject to sum(w) = 1
  Dmat <- 2 * Sigma
  dvec <- rep(0, n_assets)
  
  if (short_selling) {
    Amat <- matrix(1, nrow = n_assets, ncol = 1)
    bvec <- 1
    meq <- 1
  } else {
    Amat <- cbind(rep(1, n_assets), diag(n_assets))
    bvec <- c(1, rep(0, n_assets))
    meq <- 1
  }
  
  sol <- solve.QP(Dmat, dvec, Amat, bvec, meq = meq)
  w <- sol$solution
  
  # Portfolio statistics
  mu <- colMeans(returns)
  port_return <- sum(w * mu) * 252
  port_volatility <- sqrt(t(w) %*% Sigma %*% w * 252)
  
  list(
    weights = as.numeric(w),
    expected_return = as.numeric(port_return),
    volatility = as.numeric(port_volatility),
    asset_names = colnames(returns)
  )
}


# =============================================================================
# MAXIMUM DIVERSIFICATION PORTFOLIO
# =============================================================================

#' Maximum Diversification Portfolio
#'
#' Maximizes Diversification Ratio: DR = (w'σ) / σ_p
#' 
#' Where w'σ is weighted average volatility and σ_p is portfolio volatility
#' DR ≥ 1, with equality only for single asset portfolio
#'
#' @param returns Matrix of asset returns
#' @return List with maximum diversification weights
maximum_diversification <- function(returns) {
  n_assets <- ncol(returns)
  Sigma <- cov(returns)
  asset_vols <- sqrt(diag(Sigma))
  
  # Objective: minimize -DR = -(w'σ) / sqrt(w'Σw)
  # Equivalent to: minimize w'Σw / (w'σ)² 
  objective <- function(w) {
    w <- abs(w) / sum(abs(w))  # Normalize
    port_var <- t(w) %*% Sigma %*% w
    weighted_vol <- sum(w * asset_vols)
    as.numeric(port_var / weighted_vol^2)
  }
  
  init_w <- rep(1/n_assets, n_assets)
  opt <- optim(init_w, objective, method = "L-BFGS-B",
               lower = rep(0.001, n_assets), upper = rep(1, n_assets))
  
  w <- abs(opt$par)
  w <- w / sum(w)
  
  # Calculate diversification ratio
  port_vol <- sqrt(t(w) %*% Sigma %*% w)
  weighted_avg_vol <- sum(w * asset_vols)
  div_ratio <- weighted_avg_vol / port_vol
  
  list(
    weights = as.numeric(w),
    diversification_ratio = as.numeric(div_ratio),
    portfolio_volatility = as.numeric(port_vol) * sqrt(252),
    weighted_avg_volatility = as.numeric(weighted_avg_vol) * sqrt(252)
  )
}


# =============================================================================
# PORTFOLIO CONSTRAINTS
# =============================================================================

#' Apply portfolio constraints and reoptimize
#'
#' @param returns Matrix of asset returns
#' @param min_weights Minimum weight per asset
#' @param max_weights Maximum weight per asset
#' @param sector_constraints List of sector constraints
#' @return List with constrained optimal weights
constrained_optimization <- function(returns, min_weights = 0, max_weights = 1,
                                     sector_constraints = NULL) {
  n_assets <- ncol(returns)
  Sigma <- cov(returns)
  mu <- colMeans(returns)
  
  # Expand scalar constraints to vectors
  if (length(min_weights) == 1) {
    min_weights <- rep(min_weights, n_assets)
  }
  if (length(max_weights) == 1) {
    max_weights <- rep(max_weights, n_assets)
  }
  
  # Objective: maximize Sharpe (equivalent to min variance at optimal return)
  objective <- function(w) {
    port_return <- sum(w * mu)
    port_vol <- sqrt(t(w) %*% Sigma %*% w)
    -(port_return / port_vol)  # Negative for minimization
  }
  
  # Equality constraint: sum to 1
  eq_constraint <- function(w) {
    sum(w) - 1
  }
  
  # Initial weights (equal)
  init_w <- rep(1/n_assets, n_assets)
  
  # Optimize with constraints
  opt <- constrOptim(init_w, objective, grad = NULL,
                     ui = rbind(diag(n_assets), -diag(n_assets)),
                     ci = c(min_weights, -max_weights),
                     method = "Nelder-Mead")
  
  # Normalize to sum to 1
  w <- opt$par / sum(opt$par)
  
  # Clip to bounds
  w <- pmax(pmin(w, max_weights), min_weights)
  w <- w / sum(w)
  
  # Portfolio statistics
  port_return <- sum(w * mu) * 252
  port_volatility <- sqrt(t(w) %*% Sigma %*% w * 252)
  
  list(
    weights = as.numeric(w),
    expected_return = as.numeric(port_return),
    volatility = as.numeric(port_volatility),
    sharpe_ratio = port_return / port_volatility,
    min_weights = min_weights,
    max_weights = max_weights
  )
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if (FALSE) {  # Set to TRUE to run examples
  set.seed(42)
  
  # Generate sample returns for 5 assets
  n <- 500
  n_assets <- 5
  returns <- matrix(rnorm(n * n_assets, mean = 0.0004, sd = 0.02), 
                    nrow = n, ncol = n_assets)
  colnames(returns) <- c("Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E")
  
  # Add some correlation structure
  corr <- matrix(0.3, n_assets, n_assets)
  diag(corr) <- 1
  L <- chol(corr)
  returns <- returns %*% L
  
  # Mean-Variance Optimization
  cat("Mean-Variance Portfolio:\n")
  mv <- mean_variance_optimize(returns)
  cat("Weights:", round(mv$weights, 3), "\n")
  cat("Expected Return:", round(mv$expected_return * 100, 2), "%\n")
  cat("Volatility:", round(mv$volatility * 100, 2), "%\n")
  cat("Sharpe Ratio:", round(mv$sharpe_ratio, 2), "\n")
  
  # Risk Parity
  cat("\nRisk Parity Portfolio:\n")
  rp <- risk_parity(returns)
  cat("Weights:", round(rp$weights, 3), "\n")
  cat("Risk Contributions:", round(rp$risk_contribution_pct * 100, 2), "%\n")
  
  # Minimum Variance
  cat("\nMinimum Variance Portfolio:\n")
  minvar <- minimum_variance(returns)
  cat("Weights:", round(minvar$weights, 3), "\n")
  cat("Volatility:", round(minvar$volatility * 100, 2), "%\n")
  
  # Black-Litterman with view
  cat("\nBlack-Litterman Portfolio:\n")
  P <- matrix(0, nrow = 1, ncol = n_assets)
  P[1, 1] <- 1  # View on first asset
  Q <- matrix(0.001, nrow = 1)  # Expect 10bps daily return
  bl <- black_litterman(returns, views_P = P, views_Q = Q, views_confidence = 0.7)
  cat("Weights:", round(bl$weights, 3), "\n")
  
  # Efficient Frontier
  cat("\nEfficient Frontier (first 5 portfolios):\n")
  ef <- efficient_frontier(returns, n_portfolios = 20)
  print(head(ef$frontier, 5))
}
