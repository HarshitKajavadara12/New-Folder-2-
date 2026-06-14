# =============================================================================
# GIGA SYSTEM - Time Series Models (R)
# ARIMA, Exponential Smoothing, VAR Models
# =============================================================================

library(forecast)
library(tseries)
library(stats)

# =============================================================================
# ARIMA MODELING
# =============================================================================

#' Fit ARIMA model with automatic order selection
#'
#' Uses AIC/BIC criterion to select optimal (p,d,q) order.
#' 
#' @param returns Numeric vector of returns
#' @param max_p Maximum AR order to try
#' @param max_q Maximum MA order to try
#' @param criterion Information criterion ("aic" or "bic")
#' @return List with fitted model and diagnostics
fit_arima <- function(returns, max_p = 5, max_q = 5, criterion = "aic") {
  # Convert to time series
  ts_data <- ts(returns)
  
  # Auto ARIMA selection
  model <- auto.arima(
    ts_data,
    max.p = max_p,
    max.q = max_q,
    seasonal = FALSE,
    ic = criterion,
    stepwise = FALSE,  # Exhaustive search for best model
    approximation = FALSE
  )
  
  # Residual diagnostics
  residuals <- residuals(model)
  
  # Ljung-Box test for autocorrelation in residuals
  lb_test <- Box.test(residuals, lag = 10, type = "Ljung-Box")
  
  # ADF test for stationarity
  adf_test <- adf.test(returns, alternative = "stationary")
  
  list(
    model = model,
    order = arimaorder(model),
    aic = AIC(model),
    bic = BIC(model),
    residuals = as.numeric(residuals),
    ljung_box_pvalue = lb_test$p.value,
    adf_pvalue = adf_test$p.value,
    is_stationary = adf_test$p.value < 0.05,
    summary = capture.output(summary(model))
  )
}


#' Forecast returns using fitted ARIMA model
#'
#' @param model Fitted ARIMA model from fit_arima
#' @param horizon Forecast horizon (number of periods)
#' @param confidence_level Confidence interval level (default 0.95)
#' @return List with point forecasts and intervals
forecast_arima <- function(model, horizon = 10, confidence_level = 0.95) {
  # Generate forecasts
  fc <- forecast(model$model, h = horizon, level = confidence_level * 100)
  
  list(
    point_forecast = as.numeric(fc$mean),
    lower_bound = as.numeric(fc$lower),
    upper_bound = as.numeric(fc$upper),
    horizon = horizon,
    confidence_level = confidence_level
  )
}


# =============================================================================
# EXPONENTIAL SMOOTHING
# =============================================================================

#' Fit exponential smoothing state space model (ETS)
#'
#' ETS models: Error, Trend, Seasonality
#' - Error: Additive (A) or Multiplicative (M)
#' - Trend: None (N), Additive (A), Additive Damped (Ad)
#' - Seasonal: None (N), Additive (A), Multiplicative (M)
#'
#' @param data Numeric vector of time series values
#' @param model_type ETS model type or "auto" for automatic selection
#' @return Fitted ETS model
fit_ets <- function(data, model_type = "auto") {
  ts_data <- ts(data)
  
  if (model_type == "auto") {
    model <- ets(ts_data)
  } else {
    model <- ets(ts_data, model = model_type)
  }
  
  list(
    model = model,
    model_type = model$method,
    aic = model$aic,
    bic = model$bic,
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model)),
    smoothing_params = model$par
  )
}


#' Forecast using ETS model
forecast_ets <- function(model, horizon = 10, confidence_level = 0.95) {
  fc <- forecast(model$model, h = horizon, level = confidence_level * 100)
  
  list(
    point_forecast = as.numeric(fc$mean),
    lower_bound = as.numeric(fc$lower),
    upper_bound = as.numeric(fc$upper)
  )
}


# =============================================================================
# UNIT ROOT TESTS
# =============================================================================

#' Test for stationarity using multiple tests
#'
#' Tests performed:
#' - ADF (Augmented Dickey-Fuller)
#' - PP (Phillips-Perron)
#' - KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
#'
#' Interpretation:
#' - ADF/PP: p < 0.05 -> Reject null of unit root -> Stationary
#' - KPSS: p > 0.05 -> Cannot reject null of stationarity -> Stationary
#'
#' @param data Numeric vector
#' @return List with test results and recommendation
test_stationarity <- function(data) {
  # ADF test (null: unit root exists)
  adf <- adf.test(data, alternative = "stationary")
  
  # Phillips-Perron test (null: unit root exists)
  pp <- pp.test(data, alternative = "stationary")
  
  # KPSS test (null: series is stationary)
  kpss <- kpss.test(data)
  
  # Decision logic
  adf_stationary <- adf$p.value < 0.05
  pp_stationary <- pp$p.value < 0.05
  kpss_stationary <- kpss$p.value > 0.05
  
  # Consensus
  votes <- sum(c(adf_stationary, pp_stationary, kpss_stationary))
  is_stationary <- votes >= 2
  
  list(
    adf_pvalue = adf$p.value,
    adf_stationary = adf_stationary,
    pp_pvalue = pp$p.value,
    pp_stationary = pp_stationary,
    kpss_pvalue = kpss$p.value,
    kpss_stationary = kpss_stationary,
    consensus_stationary = is_stationary,
    recommendation = ifelse(
      is_stationary,
      "Series appears stationary, can model directly",
      "Series appears non-stationary, consider differencing"
    )
  )
}


#' Determine optimal differencing order
#'
#' @param data Numeric vector
#' @param max_d Maximum differencing order to test
#' @return Recommended differencing order (0, 1, or 2)
find_differencing_order <- function(data, max_d = 2) {
  for (d in 0:max_d) {
    if (d == 0) {
      test_data <- data
    } else {
      test_data <- diff(data, differences = d)
    }
    
    result <- test_stationarity(test_data)
    
    if (result$consensus_stationary) {
      return(list(
        optimal_d = d,
        differenced_data = test_data,
        test_results = result
      ))
    }
  }
  
  # If still not stationary, return max_d
  list(
    optimal_d = max_d,
    differenced_data = diff(data, differences = max_d),
    test_results = test_stationarity(diff(data, differences = max_d)),
    warning = "Series may require transformation beyond differencing"
  )
}


# =============================================================================
# RETURN DECOMPOSITION
# =============================================================================

#' Decompose returns into trend, seasonal, and residual components
#'
#' @param data Numeric vector of returns
#' @param frequency Seasonal frequency (252 for daily financial data)
#' @param method "stl" or "classical"
decompose_returns <- function(data, frequency = 252, method = "stl") {
  # Need at least 2 periods for seasonal decomposition
  if (length(data) < 2 * frequency) {
    return(list(
      error = "Insufficient data for seasonal decomposition",
      min_required = 2 * frequency,
      provided = length(data)
    ))
  }
  
  ts_data <- ts(data, frequency = frequency)
  
  if (method == "stl") {
    # STL: Seasonal and Trend decomposition using Loess
    decomp <- stl(ts_data, s.window = "periodic")
    
    list(
      trend = as.numeric(decomp$time.series[, "trend"]),
      seasonal = as.numeric(decomp$time.series[, "seasonal"]),
      remainder = as.numeric(decomp$time.series[, "remainder"]),
      method = "STL"
    )
  } else {
    # Classical decomposition
    decomp <- decompose(ts_data)
    
    list(
      trend = as.numeric(decomp$trend),
      seasonal = as.numeric(decomp$seasonal),
      remainder = as.numeric(decomp$random),
      method = "Classical"
    )
  }
}


# =============================================================================
# AUTOCORRELATION ANALYSIS
# =============================================================================

#' Compute ACF and PACF for model identification
#'
#' Interpretation for ARIMA:
#' - ACF cuts off at lag q -> MA(q)
#' - PACF cuts off at lag p -> AR(p)
#' - Both tail off -> ARMA
#'
#' @param data Numeric vector
#' @param max_lag Maximum lag to compute
#' @return List with ACF, PACF values and significant lags
compute_correlations <- function(data, max_lag = 20) {
  acf_result <- acf(data, lag.max = max_lag, plot = FALSE)
  pacf_result <- pacf(data, lag.max = max_lag, plot = FALSE)
  
  # Significance threshold (approximate)
  n <- length(data)
  threshold <- 1.96 / sqrt(n)
  
  acf_values <- as.numeric(acf_result$acf)[-1]  # Remove lag 0
  pacf_values <- as.numeric(pacf_result$acf)
  
  # Find significant lags
  sig_acf_lags <- which(abs(acf_values) > threshold)
  sig_pacf_lags <- which(abs(pacf_values) > threshold)
  
  list(
    acf = acf_values,
    pacf = pacf_values,
    lags = 1:max_lag,
    significance_threshold = threshold,
    significant_acf_lags = sig_acf_lags,
    significant_pacf_lags = sig_pacf_lags,
    suggested_ar_order = ifelse(length(sig_pacf_lags) > 0, max(sig_pacf_lags), 0),
    suggested_ma_order = ifelse(length(sig_acf_lags) > 0, max(sig_acf_lags), 0)
  )
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if (FALSE) {  # Set to TRUE to run examples
  # Generate sample data
  set.seed(42)
  n <- 500
  returns <- arima.sim(model = list(ar = c(0.5, -0.2), ma = 0.3), n = n)
  returns <- as.numeric(returns) + rnorm(n, 0, 0.01)
  
  # Test stationarity
  cat("Stationarity Test:\n")
  stat_test <- test_stationarity(returns)
  print(stat_test$recommendation)
  
  # Fit ARIMA
  cat("\nFitting ARIMA:\n")
  arima_fit <- fit_arima(returns)
  cat("Selected order:", arima_fit$order, "\n")
  cat("AIC:", arima_fit$aic, "\n")
  
  # Forecast
  cat("\nForecast:\n")
  fc <- forecast_arima(arima_fit, horizon = 5)
  print(fc$point_forecast)
  
  # ACF/PACF analysis
  cat("\nCorrelation Analysis:\n")
  corr <- compute_correlations(returns)
  cat("Suggested AR order:", corr$suggested_ar_order, "\n")
  cat("Suggested MA order:", corr$suggested_ma_order, "\n")
}
