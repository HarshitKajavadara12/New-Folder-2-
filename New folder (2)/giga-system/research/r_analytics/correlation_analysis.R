# =============================================================================
# GIGA SYSTEM - Correlation Analysis (R)
# DCC-GARCH, Rolling Correlation, Tail Dependence
# =============================================================================

library(rugarch)
library(rmgarch)
library(stats)

# =============================================================================
# DYNAMIC CONDITIONAL CORRELATION (DCC-GARCH)
# =============================================================================

#' Fit DCC-GARCH model for time-varying correlations
#'
#' DCC Model (Engle, 2002):
#' 
#' Step 1: Fit univariate GARCH to each series
#' Step 2: Estimate dynamic correlations:
#'   Q_t = (1-a-b)*Q̄ + a*ε_{t-1}*ε'_{t-1} + b*Q_{t-1}
#'   R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
#'
#' @param returns Matrix of returns (T x N)
#' @param garch_order GARCH order c(p, q)
#' @return List with DCC model and time-varying correlations
fit_dcc <- function(returns, garch_order = c(1, 1)) {
  n_assets <- ncol(returns)
  
  # Step 1: Univariate GARCH specifications
  uspec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = garch_order),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"
  )
  
  # Multispec for all assets
  multispec <- multispec(replicate(n_assets, uspec))
  
  # DCC specification
  dcc_spec <- dccspec(
    uspec = multispec,
    dccOrder = c(1, 1),
    distribution = "mvt"
  )
  
  # Fit DCC model
  dcc_fit <- dccfit(dcc_spec, data = returns)
  
  # Extract time-varying correlations
  R <- rcor(dcc_fit)  # Array: n_assets x n_assets x T
  
  # Extract time-varying covariances
  H <- rcov(dcc_fit)  # Array: n_assets x n_assets x T
  
  # DCC parameters
  dcc_params <- coef(dcc_fit)
  
  # Average correlation matrix
  avg_corr <- apply(R, c(1, 2), mean)
  
  # Correlation between first two assets over time (for pairs)
  if (n_assets >= 2) {
    corr_12 <- R[1, 2, ]
  } else {
    corr_12 <- NULL
  }
  
  list(
    model = dcc_fit,
    correlations = R,  # Time-varying correlation matrices
    covariances = H,   # Time-varying covariance matrices
    average_correlation = avg_corr,
    pairwise_correlation = as.numeric(corr_12),  # Time series for first pair
    dcc_parameters = as.list(dcc_params),
    n_assets = n_assets,
    n_observations = nrow(returns),
    asset_names = colnames(returns),
    log_likelihood = likelihood(dcc_fit)
  )
}


#' Extract correlation time series for specific pair
#'
#' @param dcc_model Fitted DCC model from fit_dcc
#' @param asset1 Index or name of first asset
#' @param asset2 Index or name of second asset
#' @return Numeric vector of pairwise correlations over time
get_pairwise_correlation <- function(dcc_model, asset1 = 1, asset2 = 2) {
  R <- dcc_model$correlations
  as.numeric(R[asset1, asset2, ])
}


#' Forecast correlations using DCC model
#'
#' @param dcc_model Fitted DCC model from fit_dcc
#' @param horizon Forecast horizon
#' @return List with forecasted correlation matrices
forecast_dcc <- function(dcc_model, horizon = 10) {
  fc <- dccforecast(dcc_model$model, n.ahead = horizon)
  
  # Extract forecasted correlations
  R_fc <- rcor(fc)
  H_fc <- rcov(fc)
  
  # Get correlation for first pair
  if (dcc_model$n_assets >= 2) {
    corr_12_fc <- sapply(1:horizon, function(h) R_fc[1, 2, h, 1])
  } else {
    corr_12_fc <- NULL
  }
  
  list(
    correlation_forecast = R_fc,
    covariance_forecast = H_fc,
    pairwise_forecast = corr_12_fc,
    horizon = horizon
  )
}


# =============================================================================
# ROLLING CORRELATION
# =============================================================================

#' Calculate rolling correlation between two series
#'
#' @param x First return series
#' @param y Second return series
#' @param window Rolling window size
#' @return List with rolling correlations and statistics
rolling_correlation <- function(x, y, window = 60) {
  n <- length(x)
  
  if (n != length(y)) {
    stop("Series must have same length")
  }
  
  if (n < window) {
    stop("Series too short for specified window")
  }
  
  # Calculate rolling correlation
  n_rolls <- n - window + 1
  roll_corr <- numeric(n_rolls)
  
  for (i in 1:n_rolls) {
    roll_corr[i] <- cor(x[i:(i+window-1)], y[i:(i+window-1)])
  }
  
  # Statistics
  avg_corr <- mean(roll_corr)
  sd_corr <- sd(roll_corr)
  min_corr <- min(roll_corr)
  max_corr <- max(roll_corr)
  
  # Correlation of correlations (autocorrelation)
  acf_corr <- acf(roll_corr, lag.max = 10, plot = FALSE)$acf[-1]
  
  # Detect correlation breakdowns (large negative deviations)
  threshold <- avg_corr - 2 * sd_corr
  breakdown_periods <- which(roll_corr < threshold)
  
  list(
    rolling_correlation = roll_corr,
    average = avg_corr,
    std_deviation = sd_corr,
    minimum = min_corr,
    maximum = max_corr,
    current_correlation = tail(roll_corr, 1),
    autocorrelation = as.numeric(acf_corr),
    breakdown_periods = breakdown_periods,
    window = window,
    dates = (window:n)
  )
}


#' Calculate rolling correlation matrix for multiple assets
#'
#' @param returns Matrix of returns
#' @param window Rolling window size
#' @return Array of rolling correlation matrices
rolling_correlation_matrix <- function(returns, window = 60) {
  n <- nrow(returns)
  n_assets <- ncol(returns)
  
  n_rolls <- n - window + 1
  
  # Array to store correlation matrices
  corr_array <- array(0, dim = c(n_assets, n_assets, n_rolls))
  
  for (i in 1:n_rolls) {
    window_returns <- returns[i:(i+window-1), ]
    corr_array[, , i] <- cor(window_returns)
  }
  
  # Average correlation (excluding diagonal)
  avg_corr_series <- sapply(1:n_rolls, function(i) {
    corr_mat <- corr_array[, , i]
    mean(corr_mat[upper.tri(corr_mat)])
  })
  
  list(
    correlation_matrices = corr_array,
    average_correlation = avg_corr_series,
    current_matrix = corr_array[, , n_rolls],
    asset_names = colnames(returns)
  )
}


# =============================================================================
# TAIL DEPENDENCE
# =============================================================================

#' Estimate tail dependence coefficient
#'
#' Tail dependence measures probability of joint extreme events:
#' λ_L = lim_{q→0} P(Y ≤ F_Y^{-1}(q) | X ≤ F_X^{-1}(q))
#' λ_U = lim_{q→1} P(Y > F_Y^{-1}(q) | X > F_X^{-1}(q))
#'
#' @param x First return series
#' @param y Second return series
#' @param quantile Tail quantile (default 0.05 for 5% tail)
#' @return List with tail dependence estimates
tail_dependence <- function(x, y, quantile = 0.05) {
  n <- length(x)
  
  # Convert to ranks (pseudo-observations)
  u <- rank(x) / (n + 1)
  v <- rank(y) / (n + 1)
  
  # Lower tail dependence (left tail, both small)
  left_threshold <- quantile
  left_joint <- sum(u <= left_threshold & v <= left_threshold)
  left_marginal <- sum(u <= left_threshold)
  lambda_lower <- left_joint / left_marginal
  
  # Upper tail dependence (right tail, both large)
  right_threshold <- 1 - quantile
  right_joint <- sum(u >= right_threshold & v >= right_threshold)
  right_marginal <- sum(u >= right_threshold)
  lambda_upper <- right_joint / right_marginal
  
  # Calculate across multiple thresholds
  thresholds <- c(0.01, 0.025, 0.05, 0.10)
  lower_at_thresholds <- sapply(thresholds, function(q) {
    joint <- sum(u <= q & v <= q)
    marginal <- sum(u <= q)
    if (marginal > 0) joint / marginal else NA
  })
  
  upper_at_thresholds <- sapply(thresholds, function(q) {
    q_upper <- 1 - q
    joint <- sum(u >= q_upper & v >= q_upper)
    marginal <- sum(u >= q_upper)
    if (marginal > 0) joint / marginal else NA
  })
  
  # Asymmetry: difference between upper and lower
  asymmetry <- lambda_upper - lambda_lower
  
  list(
    lower_tail_dependence = lambda_lower,
    upper_tail_dependence = lambda_upper,
    asymmetry = asymmetry,
    is_asymmetric = abs(asymmetry) > 0.1,
    lower_by_threshold = setNames(lower_at_thresholds, paste0("q", thresholds*100, "%")),
    upper_by_threshold = setNames(upper_at_thresholds, paste0("q", (1-thresholds)*100, "%")),
    linear_correlation = cor(x, y),
    interpretation = if(lambda_lower > 0.3) {
      "Strong lower tail dependence - crash together"
    } else if(lambda_upper > 0.3) {
      "Strong upper tail dependence - rally together"
    } else {
      "Weak tail dependence - independent in extremes"
    }
  )
}


#' Rolling tail dependence
#'
#' @param x First return series
#' @param y Second return series
#' @param window Rolling window
#' @param quantile Tail quantile
#' @return List with rolling tail dependence
rolling_tail_dependence <- function(x, y, window = 250, quantile = 0.05) {
  n <- length(x)
  n_rolls <- n - window + 1
  
  lower_td <- numeric(n_rolls)
  upper_td <- numeric(n_rolls)
  
  for (i in 1:n_rolls) {
    x_window <- x[i:(i+window-1)]
    y_window <- y[i:(i+window-1)]
    
    td <- tail_dependence(x_window, y_window, quantile)
    lower_td[i] <- td$lower_tail_dependence
    upper_td[i] <- td$upper_tail_dependence
  }
  
  list(
    rolling_lower_tail = lower_td,
    rolling_upper_tail = upper_td,
    avg_lower = mean(lower_td),
    avg_upper = mean(upper_td),
    current_lower = tail(lower_td, 1),
    current_upper = tail(upper_td, 1)
  )
}


# =============================================================================
# CORRELATION ANALYSIS UTILITIES
# =============================================================================

#' Compute correlation matrix with significance tests
#'
#' @param returns Matrix of returns
#' @param alpha Significance level
#' @return List with correlation matrix and p-values
correlation_significance <- function(returns, alpha = 0.05) {
  n_assets <- ncol(returns)
  n_obs <- nrow(returns)
  
  # Correlation matrix
  corr_mat <- cor(returns)
  
  # P-value matrix
  pval_mat <- matrix(NA, n_assets, n_assets)
  
  for (i in 1:(n_assets-1)) {
    for (j in (i+1):n_assets) {
      test <- cor.test(returns[, i], returns[, j])
      pval_mat[i, j] <- test$p.value
      pval_mat[j, i] <- test$p.value
    }
  }
  diag(pval_mat) <- 0
  
  # Significance matrix
  sig_mat <- pval_mat < alpha
  
  # Bonferroni-corrected alpha
  n_tests <- n_assets * (n_assets - 1) / 2
  bonferroni_alpha <- alpha / n_tests
  sig_mat_bonferroni <- pval_mat < bonferroni_alpha
  
  list(
    correlation_matrix = corr_mat,
    pvalue_matrix = pval_mat,
    significant_at_alpha = sig_mat,
    significant_bonferroni = sig_mat_bonferroni,
    bonferroni_alpha = bonferroni_alpha,
    n_significant_pairs = sum(sig_mat[upper.tri(sig_mat)]),
    n_significant_bonferroni = sum(sig_mat_bonferroni[upper.tri(sig_mat_bonferroni)])
  )
}


#' Detect correlation regime changes
#'
#' @param x First return series
#' @param y Second return series
#' @param window Rolling window
#' @param threshold Z-score threshold for regime change
#' @return List with correlation regime analysis
correlation_regime_detection <- function(x, y, window = 60, threshold = 2) {
  roll <- rolling_correlation(x, y, window)
  roll_corr <- roll$rolling_correlation
  
  # Z-score of correlation
  avg <- roll$average
  std <- roll$std_deviation
  zscore <- (roll_corr - avg) / std
  
  # Classify regimes
  regime <- rep("Normal", length(roll_corr))
  regime[zscore > threshold] <- "High Correlation"
  regime[zscore < -threshold] <- "Low Correlation"
  
  # Find regime change points
  regime_changes <- which(diff(as.numeric(factor(regime))) != 0) + 1
  
  list(
    regime = regime,
    zscore = zscore,
    regime_changes = regime_changes,
    n_regime_changes = length(regime_changes),
    current_regime = tail(regime, 1),
    proportion_high = sum(regime == "High Correlation") / length(regime),
    proportion_low = sum(regime == "Low Correlation") / length(regime)
  )
}


#' Calculate portfolio diversification metrics based on correlations
#'
#' @param returns Matrix of returns
#' @return List with diversification metrics
correlation_diversification <- function(returns) {
  n_assets <- ncol(returns)
  corr_mat <- cor(returns)
  
  # Average correlation (excluding diagonal)
  avg_corr <- mean(corr_mat[upper.tri(corr_mat)])
  
  # Diversification ratio potential
  # DR = weighted_avg_vol / portfolio_vol
  # Max DR occurs when correlations are low
  
  # Effective number of assets (based on correlation structure)
  # If all perfectly correlated, N_eff = 1
  # If all uncorrelated, N_eff = n_assets
  eigenvalues <- eigen(corr_mat)$values
  eigenvalues <- eigenvalues[eigenvalues > 0]
  
  # Effective N using eigenvalue approach
  n_effective_eigen <- sum(eigenvalues)^2 / sum(eigenvalues^2)
  
  # Effective N using average correlation
  # N_eff ≈ 1 + (N-1)*(1-ρ_avg) / (1 + (N-1)*ρ_avg)
  n_effective_corr <- 1 / (1 + (n_assets - 1) * avg_corr)
  n_effective_corr <- n_effective_corr * n_assets
  
  # Maximum pairwise correlation
  max_corr <- max(corr_mat[upper.tri(corr_mat)])
  
  # Minimum pairwise correlation
  min_corr <- min(corr_mat[upper.tri(corr_mat)])
  
  list(
    correlation_matrix = corr_mat,
    average_correlation = avg_corr,
    max_correlation = max_corr,
    min_correlation = min_corr,
    n_assets = n_assets,
    effective_n_eigenvalue = n_effective_eigen,
    effective_n_correlation = n_effective_corr,
    diversification_potential = n_effective_eigen / n_assets,
    eigenvalues = eigenvalues
  )
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if (FALSE) {  # Set to TRUE to run examples
  set.seed(42)
  
  # Generate correlated returns
  n <- 500
  n_assets <- 3
  
  # Base returns
  base <- rnorm(n, 0, 0.01)
  
  returns <- matrix(0, n, n_assets)
  returns[, 1] <- base + rnorm(n, 0.0003, 0.005)
  returns[, 2] <- 0.7 * base + rnorm(n, 0.0002, 0.008)
  returns[, 3] <- -0.3 * base + rnorm(n, 0.0001, 0.012)
  colnames(returns) <- c("Asset_A", "Asset_B", "Asset_C")
  
  # Rolling correlation
  cat("Rolling Correlation (A vs B):\n")
  roll <- rolling_correlation(returns[,1], returns[,2], window = 60)
  cat("Average:", round(roll$average, 3), "\n")
  cat("Current:", round(roll$current_correlation, 3), "\n")
  
  # Tail dependence
  cat("\nTail Dependence (A vs B):\n")
  td <- tail_dependence(returns[,1], returns[,2])
  cat("Lower tail:", round(td$lower_tail_dependence, 3), "\n")
  cat("Upper tail:", round(td$upper_tail_dependence, 3), "\n")
  cat("Linear correlation:", round(td$linear_correlation, 3), "\n")
  
  # Diversification analysis
  cat("\nDiversification Analysis:\n")
  div <- correlation_diversification(returns)
  cat("Average correlation:", round(div$average_correlation, 3), "\n")
  cat("Effective N:", round(div$effective_n_eigenvalue, 2), "out of", div$n_assets, "\n")
  
  # DCC-GARCH (takes longer to fit)
  cat("\nDCC-GARCH Model:\n")
  dcc <- fit_dcc(returns[, 1:2])  # Fit on first two assets
  cat("Current correlation:", round(tail(dcc$pairwise_correlation, 1), 3), "\n")
  cat("Average correlation:", round(mean(dcc$pairwise_correlation), 3), "\n")
}
