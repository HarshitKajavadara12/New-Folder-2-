# =============================================================================
# GIGA SYSTEM - Econometrics (R)
# Cointegration, Granger Causality, Vector Models
# =============================================================================

library(tseries)
library(vars)
library(urca)
library(stats)

# =============================================================================
# COINTEGRATION ANALYSIS
# =============================================================================

#' Test for cointegration using Engle-Granger two-step method
#'
#' Cointegration: Two non-stationary series that have a stationary linear combination
#'
#' Step 1: Regress Y on X to get residuals
#' Step 2: Test residuals for stationarity (ADF test)
#'
#' Trading Application: Pairs trading - if cointegrated, spread is mean-reverting
#'
#' @param y Dependent variable (prices or log-prices)
#' @param x Independent variable (prices or log-prices)
#' @return List with test results, hedge ratio, and half-life
test_cointegration <- function(y, x) {
  # Step 1: OLS regression
  model <- lm(y ~ x)
  residuals <- model$residuals
  hedge_ratio <- coef(model)["x"]
  
  # Step 2: ADF test on residuals (critical values are different for cointegration)
  adf_result <- adf.test(residuals, alternative = "stationary")
  
  # Engle-Granger critical values (5% level, 2 variables)
  # These are stricter than standard ADF critical values
  eg_critical_5pct <- -3.37  # Approximate for n > 250
  
  is_cointegrated <- adf_result$statistic < eg_critical_5pct
  
  # Calculate half-life of mean reversion
  # OU process: dx = θ(μ - x)dt + σdW
  # Half-life = ln(2) / θ
  spread <- y - hedge_ratio * x
  spread_lag <- c(NA, spread[-length(spread)])
  spread_diff <- c(NA, diff(spread))
  
  # Regression: Δspread_t = θ * spread_{t-1} + ε
  hl_model <- lm(spread_diff[-1] ~ spread_lag[-1] - 1)
  theta <- -coef(hl_model)[1]
  half_life <- log(2) / theta
  
  list(
    is_cointegrated = is_cointegrated,
    adf_statistic = as.numeric(adf_result$statistic),
    adf_pvalue = adf_result$p.value,
    critical_value_5pct = eg_critical_5pct,
    hedge_ratio = as.numeric(hedge_ratio),
    intercept = as.numeric(coef(model)["(Intercept)"]),
    spread = as.numeric(spread),
    spread_mean = mean(spread),
    spread_std = sd(spread),
    half_life = as.numeric(half_life),
    r_squared = summary(model)$r.squared
  )
}


#' Johansen cointegration test for multiple series
#'
#' More powerful than Engle-Granger for multivariate analysis
#'
#' Tests for number of cointegrating relationships (rank r)
#' - r = 0: No cointegration
#' - 0 < r < n: r cointegrating vectors
#' - r = n: All series are stationary
#'
#' @param data Matrix or data frame with each column being a time series
#' @param type Test type: "trace" or "eigen"
#' @param K VAR lag order
#' @return List with test statistics and cointegrating vectors
johansen_test <- function(data, type = "trace", K = 2) {
  # Convert to matrix
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  
  n_vars <- ncol(data)
  
  # Johansen test
  jtest <- ca.jo(data, type = type, K = K, ecdet = "const", spec = "transitory")
  
  # Extract results
  test_stats <- jtest@teststat
  crit_values <- jtest@cval
  
  # Determine cointegration rank
  # Compare test statistic with critical values (5% level)
  rank <- 0
  for (i in 1:length(test_stats)) {
    if (test_stats[i] > crit_values[i, "5pct"]) {
      rank <- rank + 1
    }
  }
  
  # Cointegrating vectors (β matrix)
  beta <- jtest@V
  
  # Adjustment coefficients (α matrix)
  alpha <- jtest@W
  
  list(
    cointegration_rank = rank,
    test_statistics = as.numeric(test_stats),
    critical_values_5pct = crit_values[, "5pct"],
    cointegrating_vectors = beta,
    adjustment_coefficients = alpha,
    n_variables = n_vars,
    lags = K,
    summary = capture.output(summary(jtest))
  )
}


# =============================================================================
# GRANGER CAUSALITY
# =============================================================================

#' Test Granger causality between two series
#'
#' "X Granger-causes Y" means X's past values help predict Y,
#' beyond what Y's own past values can predict.
#'
#' Null hypothesis: X does NOT Granger-cause Y
#' p < 0.05: Reject null → X Granger-causes Y
#'
#' @param y Target series (effect)
#' @param x Potential cause series
#' @param max_lag Maximum lag to test
#' @return List with causality test results
granger_causality <- function(y, x, max_lag = 5) {
  results <- list()
  
  for (lag in 1:max_lag) {
    # Create lagged variables
    data <- data.frame(y = y, x = x)
    for (i in 1:lag) {
      data[[paste0("y_lag", i)]] <- c(rep(NA, i), y[1:(length(y)-i)])
      data[[paste0("x_lag", i)]] <- c(rep(NA, i), x[1:(length(x)-i)])
    }
    data <- na.omit(data)
    
    # Restricted model: Y ~ Y_lags only
    y_lags_formula <- paste("y ~", paste(paste0("y_lag", 1:lag), collapse = " + "))
    model_restricted <- lm(as.formula(y_lags_formula), data = data)
    
    # Unrestricted model: Y ~ Y_lags + X_lags
    full_formula <- paste(y_lags_formula, "+", paste(paste0("x_lag", 1:lag), collapse = " + "))
    model_unrestricted <- lm(as.formula(full_formula), data = data)
    
    # F-test
    anova_result <- anova(model_restricted, model_unrestricted)
    f_stat <- anova_result$F[2]
    p_value <- anova_result$`Pr(>F)`[2]
    
    results[[paste0("lag_", lag)]] <- list(
      lag = lag,
      f_statistic = f_stat,
      p_value = p_value,
      x_causes_y = p_value < 0.05
    )
  }
  
  # Find optimal lag using AIC
  aic_values <- sapply(1:max_lag, function(lag) {
    data <- data.frame(y = y, x = x)
    for (i in 1:lag) {
      data[[paste0("y_lag", i)]] <- c(rep(NA, i), y[1:(length(y)-i)])
      data[[paste0("x_lag", i)]] <- c(rep(NA, i), x[1:(length(x)-i)])
    }
    data <- na.omit(data)
    full_formula <- paste("y ~", paste(c(paste0("y_lag", 1:lag), paste0("x_lag", 1:lag)), collapse = " + "))
    AIC(lm(as.formula(full_formula), data = data))
  })
  
  optimal_lag <- which.min(aic_values)
  
  list(
    results_by_lag = results,
    optimal_lag = optimal_lag,
    aic_by_lag = aic_values,
    summary = list(
      x_causes_y_at_optimal_lag = results[[paste0("lag_", optimal_lag)]]$x_causes_y,
      p_value_at_optimal_lag = results[[paste0("lag_", optimal_lag)]]$p_value
    )
  )
}


#' Bidirectional Granger causality test
#'
#' @param x First series
#' @param y Second series
#' @param max_lag Maximum lag to test
#' @return List with bidirectional causality results
bidirectional_granger <- function(x, y, max_lag = 5) {
  x_to_y <- granger_causality(y, x, max_lag)
  y_to_x <- granger_causality(x, y, max_lag)
  
  list(
    x_causes_y = x_to_y,
    y_causes_x = y_to_x,
    relationship = ifelse(
      x_to_y$summary$x_causes_y_at_optimal_lag & y_to_x$summary$x_causes_y_at_optimal_lag,
      "Bidirectional causality",
      ifelse(
        x_to_y$summary$x_causes_y_at_optimal_lag,
        "X causes Y (unidirectional)",
        ifelse(
          y_to_x$summary$x_causes_y_at_optimal_lag,
          "Y causes X (unidirectional)",
          "No Granger causality detected"
        )
      )
    )
  )
}


# =============================================================================
# VECTOR AUTOREGRESSION (VAR)
# =============================================================================

#' Fit Vector Autoregression model
#'
#' VAR(p) model:
#' Y_t = A_1 * Y_{t-1} + A_2 * Y_{t-2} + ... + A_p * Y_{t-p} + ε_t
#'
#' Applications:
#' - Forecasting multiple related series
#' - Impulse response analysis (how shocks propagate)
#' - Variance decomposition
#'
#' @param data Matrix or data frame of time series
#' @param p VAR lag order (or "auto" for automatic selection)
#' @param max_p Maximum lag order for automatic selection
#' @return List with VAR model and diagnostics
fit_var <- function(data, p = "auto", max_p = 10) {
  # Convert to matrix
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  
  # Automatic lag selection
  if (p == "auto") {
    selection <- VARselect(data, lag.max = max_p)
    p <- selection$selection["AIC(n)"]
  }
  
  # Fit VAR
  var_model <- VAR(data, p = p)
  
  # Stability check (all eigenvalues should be < 1 in modulus)
  roots <- roots(var_model)
  is_stable <- all(Mod(roots) < 1)
  
  # Serial correlation test (Portmanteau)
  serial_test <- serial.test(var_model, lags.pt = 10)
  
  list(
    model = var_model,
    lag_order = p,
    coefficients = coef(var_model),
    roots = roots,
    is_stable = is_stable,
    serial_correlation_pvalue = serial_test$serial$p.value,
    has_serial_correlation = serial_test$serial$p.value < 0.05,
    aic = AIC(var_model),
    bic = BIC(var_model)
  )
}


#' Impulse Response Functions from VAR
#'
#' Shows how a shock to one variable propagates to others over time
#'
#' @param var_model Fitted VAR model from fit_var
#' @param impulse Variable receiving the shock
#' @param response Variable's response to measure
#' @param horizon Forecast horizon
#' @param ortho Use orthogonalized impulses (Cholesky decomposition)
#' @return List with IRF values
compute_irf <- function(var_model, impulse = NULL, response = NULL, 
                        horizon = 20, ortho = TRUE) {
  
  irf_result <- irf(var_model$model, impulse = impulse, response = response,
                    n.ahead = horizon, ortho = ortho, boot = TRUE, ci = 0.95)
  
  # Extract IRF values
  irfs <- list()
  for (imp in names(irf_result$irf)) {
    irfs[[imp]] <- list(
      point = as.matrix(irf_result$irf[[imp]]),
      lower = as.matrix(irf_result$Lower[[imp]]),
      upper = as.matrix(irf_result$Upper[[imp]])
    )
  }
  
  list(
    irfs = irfs,
    horizon = horizon,
    orthogonalized = ortho
  )
}


#' Forecast Error Variance Decomposition
#'
#' Shows proportion of forecast variance attributable to each variable
#'
#' @param var_model Fitted VAR model
#' @param horizon Forecast horizon
#' @return List with variance decomposition
variance_decomposition <- function(var_model, horizon = 20) {
  fevd_result <- fevd(var_model$model, n.ahead = horizon)
  
  # Convert to list
  decomp <- list()
  for (var_name in names(fevd_result)) {
    decomp[[var_name]] <- as.matrix(fevd_result[[var_name]])
  }
  
  list(
    decomposition = decomp,
    horizon = horizon
  )
}


#' Forecast using VAR model
#'
#' @param var_model Fitted VAR model from fit_var
#' @param horizon Forecast horizon
#' @param confidence_level Confidence level for intervals
#' @return List with forecasts
forecast_var <- function(var_model, horizon = 10, confidence_level = 0.95) {
  fc <- predict(var_model$model, n.ahead = horizon, ci = confidence_level)
  
  forecasts <- list()
  for (var_name in names(fc$fcst)) {
    forecasts[[var_name]] <- list(
      point = fc$fcst[[var_name]][, "fcst"],
      lower = fc$fcst[[var_name]][, "lower"],
      upper = fc$fcst[[var_name]][, "upper"]
    )
  }
  
  list(
    forecasts = forecasts,
    horizon = horizon,
    confidence_level = confidence_level
  )
}


# =============================================================================
# ERROR CORRECTION MODEL (ECM/VECM)
# =============================================================================

#' Fit Vector Error Correction Model
#'
#' VECM: For cointegrated series
#' ΔY_t = αβ'Y_{t-1} + Γ_1ΔY_{t-1} + ... + Γ_{k-1}ΔY_{t-k+1} + ε_t
#'
#' - β: Cointegrating vectors (long-run equilibrium)
#' - α: Adjustment coefficients (speed of adjustment to equilibrium)
#'
#' @param data Matrix of cointegrated time series
#' @param r Cointegration rank (number of cointegrating relationships)
#' @param K VAR lag order
#' @return List with VECM model
fit_vecm <- function(data, r = 1, K = 2) {
  # First, run Johansen test to verify cointegration
  jtest <- ca.jo(data, type = "trace", K = K, ecdet = "const", spec = "transitory")
  
  # Convert to VECM
  vecm <- cajorls(jtest, r = r)
  
  # Extract parameters
  beta <- vecm$beta  # Cointegrating vectors
  
  list(
    vecm = vecm,
    cointegrating_vectors = beta,
    cointegration_rank = r,
    johansen_test = jtest
  )
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if (FALSE) {  # Set to TRUE to run examples
  set.seed(42)
  n <- 500
  
  # Generate cointegrated series (for pairs trading)
  x <- cumsum(rnorm(n, 0, 1))  # Random walk
  y <- 0.8 * x + rnorm(n, 0, 0.5)  # Cointegrated with x
  
  # Test cointegration
  cat("Cointegration Test:\n")
  coint <- test_cointegration(y, x)
  cat("Cointegrated:", coint$is_cointegrated, "\n")
  cat("Hedge ratio:", coint$hedge_ratio, "\n")
  cat("Half-life:", coint$half_life, "periods\n")
  
  # Granger causality
  cat("\nGranger Causality:\n")
  returns_x <- diff(x)
  returns_y <- diff(y)
  gc <- bidirectional_granger(returns_x, returns_y, max_lag = 5)
  cat("Relationship:", gc$relationship, "\n")
  
  # VAR model
  cat("\nVAR Model:\n")
  var_data <- cbind(returns_x, returns_y)
  colnames(var_data) <- c("X", "Y")
  var_fit <- fit_var(var_data, p = "auto")
  cat("Optimal lag:", var_fit$lag_order, "\n")
  cat("Stable:", var_fit$is_stable, "\n")
  
  # Forecast
  fc <- forecast_var(var_fit, horizon = 5)
  cat("5-step ahead forecast for X:", fc$forecasts$X$point, "\n")
}
