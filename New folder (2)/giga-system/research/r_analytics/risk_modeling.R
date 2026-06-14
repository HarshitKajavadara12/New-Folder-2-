# =============================================================================
# GIGA SYSTEM - Risk Modeling (R)
# GARCH, Extreme Value Theory, Copulas
# =============================================================================

library(rugarch)
library(copula)
library(evd)

# =============================================================================
# GARCH FAMILY MODELS
# =============================================================================

#' Fit GARCH(1,1) model for volatility
#'
#' GARCH(1,1) Equation:
#' σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
#'
#' Persistence: α + β (< 1 for stationary)
#' Half-life: log(0.5) / log(α + β)
#'
#' @param returns Numeric vector of returns
#' @param p ARCH order (default 1)
#' @param q GARCH order (default 1)
#' @param dist Innovation distribution: "norm", "std" (Student-t), "ged"
#' @return List with model, parameters, volatility forecast
fit_garch <- function(returns, p = 1, q = 1, dist = "std") {
  # Model specification
  spec <- ugarchspec(
    variance.model = list(
      model = "sGARCH",
      garchOrder = c(p, q)
    ),
    mean.model = list(
      armaOrder = c(0, 0),
      include.mean = TRUE
    ),
    distribution.model = dist
  )
  
  # Fit model
  fit <- ugarchfit(spec, data = returns, solver = "hybrid")
  
  # Extract parameters
  coef_vals <- coef(fit)
  alpha <- coef_vals["alpha1"]
  beta <- coef_vals["beta1"]
  omega <- coef_vals["omega"]
  
  # Persistence and half-life
  persistence <- alpha + beta
  half_life <- ifelse(persistence < 1, log(0.5) / log(persistence), Inf)
  
  # Long-run variance
  long_run_var <- omega / (1 - persistence)
  
  list(
    model = fit,
    coefficients = as.list(coef_vals),
    alpha = as.numeric(alpha),
    beta = as.numeric(beta),
    omega = as.numeric(omega),
    persistence = as.numeric(persistence),
    half_life = as.numeric(half_life),
    long_run_volatility = as.numeric(sqrt(long_run_var) * sqrt(252)),  # Annualized
    conditional_volatility = as.numeric(sigma(fit)),
    residuals = as.numeric(residuals(fit)),
    standardized_residuals = as.numeric(residuals(fit, standardize = TRUE)),
    aic = infocriteria(fit)[1],
    bic = infocriteria(fit)[2],
    log_likelihood = likelihood(fit)
  )
}


#' Fit EGARCH model (asymmetric)
#'
#' EGARCH captures leverage effect (negative returns → higher volatility)
#'
#' log(σ²_t) = ω + α * |z_{t-1}| + γ * z_{t-1} + β * log(σ²_{t-1})
#'
#' γ < 0: Leverage effect present (asymmetric response)
#'
#' @param returns Numeric vector of returns
#' @param dist Innovation distribution
#' @return List with model and asymmetry coefficient
fit_egarch <- function(returns, dist = "std") {
  spec <- ugarchspec(
    variance.model = list(
      model = "eGARCH",
      garchOrder = c(1, 1)
    ),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = dist
  )
  
  fit <- ugarchfit(spec, data = returns, solver = "hybrid")
  coef_vals <- coef(fit)
  
  # Gamma is the asymmetry parameter
  gamma <- coef_vals["gamma1"]
  
  list(
    model = fit,
    coefficients = as.list(coef_vals),
    asymmetry_gamma = as.numeric(gamma),
    has_leverage_effect = gamma < 0,
    conditional_volatility = as.numeric(sigma(fit)),
    aic = infocriteria(fit)[1]
  )
}


#' Fit GJR-GARCH model (Threshold GARCH)
#'
#' GJR-GARCH: σ²_t = ω + (α + γ * I_{t-1}) * ε²_{t-1} + β * σ²_{t-1}
#'
#' I_{t-1} = 1 if ε_{t-1} < 0, else 0
#' γ > 0: Negative shocks have larger impact
#'
#' @param returns Numeric vector of returns
#' @param dist Innovation distribution
fit_gjr_garch <- function(returns, dist = "std") {
  spec <- ugarchspec(
    variance.model = list(
      model = "gjrGARCH",
      garchOrder = c(1, 1)
    ),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = dist
  )
  
  fit <- ugarchfit(spec, data = returns, solver = "hybrid")
  coef_vals <- coef(fit)
  
  list(
    model = fit,
    coefficients = as.list(coef_vals),
    asymmetry_gamma = as.numeric(coef_vals["gamma1"]),
    conditional_volatility = as.numeric(sigma(fit)),
    aic = infocriteria(fit)[1]
  )
}


#' Forecast volatility using GARCH model
#'
#' @param garch_model Fitted GARCH model from fit_garch
#' @param horizon Forecast horizon in days
#' @return List with volatility forecasts
forecast_garch <- function(garch_model, horizon = 10) {
  fc <- ugarchforecast(garch_model$model, n.ahead = horizon)
  
  list(
    volatility_forecast = as.numeric(sigma(fc)),
    variance_forecast = as.numeric(sigma(fc))^2,
    mean_forecast = as.numeric(fitted(fc)),
    horizon = horizon
  )
}


# =============================================================================
# EXTREME VALUE THEORY
# =============================================================================

#' Fit Generalized Extreme Value (GEV) distribution to block maxima
#'
#' GEV Distribution Parameters:
#' - μ (location): Central tendency
#' - σ (scale): Spread
#' - ξ (shape): Tail behavior
#'   - ξ > 0: Fréchet (heavy tail, unbounded)
#'   - ξ = 0: Gumbel (light tail, exponential decay)
#'   - ξ < 0: Weibull (bounded upper tail)
#'
#' @param losses Numeric vector of losses (positive = loss)
#' @param block_size Size of blocks for maxima extraction
#' @return List with GEV parameters and VaR estimates
fit_gev <- function(losses, block_size = 21) {  # 21 ~ monthly trading days
  # Extract block maxima
  n_blocks <- floor(length(losses) / block_size)
  block_maxima <- sapply(1:n_blocks, function(i) {
    start <- (i - 1) * block_size + 1
    end <- i * block_size
    max(losses[start:end])
  })
  
  # Fit GEV
  fit <- fgev(block_maxima)
  params <- fit$estimate
  
  # VaR calculation using GEV quantile
  var_95 <- qgev(0.95, loc = params["loc"], scale = params["scale"], 
                 shape = params["shape"])
  var_99 <- qgev(0.99, loc = params["loc"], scale = params["scale"], 
                 shape = params["shape"])
  
  # Return level (e.g., 100-year return level)
  m <- 100  # Return period
  return_level <- params["loc"] + (params["scale"] / params["shape"]) * 
    ((-log(1 - 1/m))^(-params["shape"]) - 1)
  
  list(
    location = as.numeric(params["loc"]),
    scale = as.numeric(params["scale"]),
    shape = as.numeric(params["shape"]),
    tail_type = ifelse(params["shape"] > 0.1, "Heavy (Fréchet)",
                       ifelse(params["shape"] < -0.1, "Bounded (Weibull)", "Light (Gumbel)")),
    var_95 = as.numeric(var_95),
    var_99 = as.numeric(var_99),
    return_level_100 = as.numeric(return_level),
    block_maxima = block_maxima,
    n_blocks = n_blocks
  )
}


#' Fit Generalized Pareto Distribution (GPD) using Peaks Over Threshold
#'
#' GPD for exceedances over threshold u:
#' F(x) = 1 - (1 + ξ(x-u)/σ)^(-1/ξ) for ξ ≠ 0
#'
#' @param losses Numeric vector of losses
#' @param threshold Threshold for exceedances (or quantile if < 1)
#' @return List with GPD parameters and tail risk estimates
fit_gpd <- function(losses, threshold = 0.95) {
  # If threshold < 1, treat as quantile
  if (threshold < 1) {
    threshold <- quantile(losses, threshold)
  }
  
  # Extract exceedances
  exceedances <- losses[losses > threshold] - threshold
  n_exceedances <- length(exceedances)
  n_total <- length(losses)
  
  if (n_exceedances < 30) {
    warning("Less than 30 exceedances, GPD estimates may be unreliable")
  }
  
  # Fit GPD
  fit <- fpot(losses, threshold = threshold)
  params <- fit$estimate
  
  # Exceedance probability
  prob_exceed <- n_exceedances / n_total
  
  # VaR and CVaR using GPD
  # VaR_p = u + (σ/ξ) * ((n/n_u * (1-p))^(-ξ) - 1)
  calculate_var <- function(p) {
    if (params["shape"] != 0) {
      threshold + (params["scale"] / params["shape"]) * 
        ((prob_exceed / (1 - p))^params["shape"] - 1)
    } else {
      threshold - params["scale"] * log((1 - p) / prob_exceed)
    }
  }
  
  var_99 <- calculate_var(0.99)
  var_999 <- calculate_var(0.999)
  
  # Expected Shortfall
  es_99 <- var_99 / (1 - params["shape"]) + 
    (params["scale"] - params["shape"] * threshold) / (1 - params["shape"])
  
  list(
    threshold = as.numeric(threshold),
    scale = as.numeric(params["scale"]),
    shape = as.numeric(params["shape"]),
    n_exceedances = n_exceedances,
    exceedance_rate = prob_exceed,
    var_99 = as.numeric(var_99),
    var_999 = as.numeric(var_999),
    cvar_99 = as.numeric(es_99),
    tail_index = 1 / as.numeric(params["shape"])  # Reciprocal of shape
  )
}


# =============================================================================
# COPULA MODELING
# =============================================================================

#' Fit copula to model dependence structure
#'
#' Copulas separate:
#' - Marginal distributions (individual asset behavior)
#' - Dependence structure (how assets move together)
#'
#' Sklar's Theorem: H(x,y) = C(F(x), G(y))
#'
#' @param returns1 Returns of first asset
#' @param returns2 Returns of second asset
#' @param family Copula family: "gaussian", "t", "clayton", "gumbel", "frank"
#' @return List with copula parameters and tail dependence
fit_copula <- function(returns1, returns2, family = "t") {
  # Convert to pseudo-observations (uniform margins)
  u1 <- pobs(returns1)
  u2 <- pobs(returns2)
  u <- cbind(u1, u2)
  
  # Fit copula based on family
  if (family == "gaussian") {
    copula_obj <- normalCopula(dim = 2)
    fit <- fitCopula(copula_obj, u, method = "ml")
    rho <- coef(fit)[1]
    
    list(
      family = "Gaussian",
      correlation = rho,
      parameters = list(rho = rho),
      upper_tail_dependence = 0,  # Gaussian has no tail dependence
      lower_tail_dependence = 0,
      copula = fit@copula
    )
    
  } else if (family == "t") {
    # t-copula captures tail dependence
    copula_obj <- tCopula(dim = 2)
    fit <- fitCopula(copula_obj, u, method = "ml")
    rho <- coef(fit)[1]
    df <- coef(fit)[2]
    
    # Tail dependence for t-copula
    # λ = 2 * t_{df+1}(-√((df+1)(1-ρ)/(1+ρ)))
    t_val <- -sqrt((df + 1) * (1 - rho) / (1 + rho))
    tail_dep <- 2 * pt(t_val, df + 1)
    
    list(
      family = "Student-t",
      correlation = rho,
      degrees_of_freedom = df,
      parameters = list(rho = rho, df = df),
      upper_tail_dependence = tail_dep,
      lower_tail_dependence = tail_dep,  # Symmetric
      copula = fit@copula
    )
    
  } else if (family == "clayton") {
    # Clayton: Lower tail dependence
    copula_obj <- claytonCopula(dim = 2)
    fit <- fitCopula(copula_obj, u, method = "ml")
    theta <- coef(fit)[1]
    
    list(
      family = "Clayton",
      theta = theta,
      parameters = list(theta = theta),
      upper_tail_dependence = 0,
      lower_tail_dependence = 2^(-1/theta),
      copula = fit@copula
    )
    
  } else if (family == "gumbel") {
    # Gumbel: Upper tail dependence
    copula_obj <- gumbelCopula(dim = 2)
    fit <- fitCopula(copula_obj, u, method = "ml")
    theta <- coef(fit)[1]
    
    list(
      family = "Gumbel",
      theta = theta,
      parameters = list(theta = theta),
      upper_tail_dependence = 2 - 2^(1/theta),
      lower_tail_dependence = 0,
      copula = fit@copula
    )
    
  } else if (family == "frank") {
    # Frank: No tail dependence, symmetric
    copula_obj <- frankCopula(dim = 2)
    fit <- fitCopula(copula_obj, u, method = "ml")
    theta <- coef(fit)[1]
    
    list(
      family = "Frank",
      theta = theta,
      parameters = list(theta = theta),
      upper_tail_dependence = 0,
      lower_tail_dependence = 0,
      copula = fit@copula
    )
  }
}


#' Compare multiple copula families and select best fit
#'
#' @param returns1 Returns of first asset
#' @param returns2 Returns of second asset
#' @return List with all fits and best model
select_copula <- function(returns1, returns2) {
  families <- c("gaussian", "t", "clayton", "gumbel", "frank")
  
  u1 <- pobs(returns1)
  u2 <- pobs(returns2)
  u <- cbind(u1, u2)
  
  results <- list()
  aics <- c()
  
  for (fam in families) {
    tryCatch({
      fit <- fit_copula(returns1, returns2, family = fam)
      
      # Calculate AIC
      copula_obj <- fit$copula
      ll <- loglikCopula(coef(copula_obj), u, copula_obj)
      k <- length(coef(copula_obj))
      aic <- -2 * ll + 2 * k
      
      results[[fam]] <- fit
      results[[fam]]$aic <- aic
      aics <- c(aics, aic)
      names(aics)[length(aics)] <- fam
    }, error = function(e) {
      message(paste("Failed to fit", fam, "copula:", e$message))
    })
  }
  
  # Select best by AIC
  best_family <- names(which.min(aics))
  
  list(
    all_fits = results,
    aic_values = aics,
    best_family = best_family,
    best_model = results[[best_family]]
  )
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if (FALSE) {  # Set to TRUE to run examples
  set.seed(42)
  
  # Generate sample returns with volatility clustering
  n <- 1000
  returns <- numeric(n)
  sigma <- numeric(n)
  sigma[1] <- 0.02
  
  for (i in 2:n) {
    sigma[i] <- sqrt(0.00001 + 0.1 * returns[i-1]^2 + 0.85 * sigma[i-1]^2)
    returns[i] <- sigma[i] * rnorm(1)
  }
  
  # Fit GARCH
  cat("GARCH(1,1) Fit:\n")
  garch_fit <- fit_garch(returns)
  cat("Persistence:", garch_fit$persistence, "\n")
  cat("Half-life:", garch_fit$half_life, "days\n")
  
  # Fit EGARCH
  cat("\nEGARCH Fit:\n")
  egarch_fit <- fit_egarch(returns)
  cat("Asymmetry:", egarch_fit$asymmetry_gamma, "\n")
  cat("Has leverage effect:", egarch_fit$has_leverage_effect, "\n")
  
  # EVT - GPD
  cat("\nGPD Tail Risk:\n")
  losses <- -returns  # Convert to losses
  gpd_fit <- fit_gpd(losses[losses > 0], threshold = 0.95)
  cat("99% VaR:", gpd_fit$var_99, "\n")
  cat("Tail index:", gpd_fit$tail_index, "\n")
  
  # Copula
  cat("\nCopula Analysis:\n")
  returns2 <- returns + rnorm(n, 0, 0.01)  # Correlated series
  copula_result <- select_copula(returns, returns2)
  cat("Best copula:", copula_result$best_family, "\n")
  cat("Tail dependence:", copula_result$best_model$upper_tail_dependence, "\n")
}
