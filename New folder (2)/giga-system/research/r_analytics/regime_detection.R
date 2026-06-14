# =============================================================================
# GIGA SYSTEM - Regime Detection (R)
# Hidden Markov Models, Change Point Detection
# =============================================================================

library(depmixS4)
library(stats)

# =============================================================================
# HIDDEN MARKOV MODELS
# =============================================================================

#' Fit Hidden Markov Model for regime detection
#'
#' HMM assumes market is in one of K unobservable states (regimes)
#' Each regime has its own return distribution
#'
#' Common regimes:
#' - Bull market (high mean, low volatility)
#' - Bear market (negative mean, high volatility)
#' - High volatility (normal mean, very high volatility)
#'
#' @param returns Numeric vector of returns
#' @param n_states Number of hidden states (regimes)
#' @param distribution Distribution family ("gaussian" or "student")
#' @return List with fitted HMM and regime probabilities
fit_hmm <- function(returns, n_states = 2, distribution = "gaussian") {
  # Create data frame
  data <- data.frame(returns = returns)
  
  # Specify HMM model
  if (distribution == "gaussian") {
    # Gaussian HMM
    model <- depmix(
      returns ~ 1,  # Response formula (mean depends on state)
      data = data,
      nstates = n_states,
      family = gaussian()
    )
  } else {
    # For student-t, we use gaussian as approximation
    model <- depmix(
      returns ~ 1,
      data = data,
      nstates = n_states,
      family = gaussian()
    )
  }
  
  # Fit model using EM algorithm
  fit <- fit(model, verbose = FALSE)
  
  # Extract parameters
  params <- getpars(fit)
  
  # Get posterior probabilities (most likely state at each time)
  posteriors <- posterior(fit)
  
  # State assignments
  states <- posteriors$state
  
  # State probabilities over time
  state_probs <- posteriors[, -1]  # Remove state column
  
  # Calculate state-specific statistics
  state_stats <- list()
  for (s in 1:n_states) {
    state_returns <- returns[states == s]
    if (length(state_returns) > 1) {
      state_stats[[paste0("state_", s)]] <- list(
        mean = mean(state_returns) * 252,  # Annualized
        volatility = sd(state_returns) * sqrt(252),
        sharpe = mean(state_returns) / sd(state_returns) * sqrt(252),
        n_observations = length(state_returns),
        proportion = length(state_returns) / length(returns)
      )
    }
  }
  
  # Transition matrix
  # Extract from parameters
  n_trans <- n_states * n_states
  trans_params <- params[(length(params) - n_trans + 1):length(params)]
  
  # Build transition matrix (need to handle multinomial logit transformation)
  # For simplicity, estimate from state sequence
  trans_matrix <- matrix(0, n_states, n_states)
  for (i in 1:(length(states)-1)) {
    trans_matrix[states[i], states[i+1]] <- trans_matrix[states[i], states[i+1]] + 1
  }
  trans_matrix <- trans_matrix / rowSums(trans_matrix)
  
  # Expected duration in each state
  # E[duration_i] = 1 / (1 - p_ii)
  expected_durations <- 1 / (1 - diag(trans_matrix))
  
  # Log-likelihood and information criteria
  ll <- logLik(fit)
  n_params <- length(params)
  n_obs <- length(returns)
  aic <- -2 * as.numeric(ll) + 2 * n_params
  bic <- -2 * as.numeric(ll) + n_params * log(n_obs)
  
  list(
    model = fit,
    n_states = n_states,
    states = as.numeric(states),
    state_probabilities = as.matrix(state_probs),
    transition_matrix = trans_matrix,
    state_statistics = state_stats,
    expected_durations = as.numeric(expected_durations),
    log_likelihood = as.numeric(ll),
    aic = aic,
    bic = bic,
    current_state = tail(states, 1),
    current_probabilities = as.numeric(tail(state_probs, 1))
  )
}


#' Predict next period regime probabilities
#'
#' @param hmm_model Fitted HMM from fit_hmm
#' @param horizon Prediction horizon (number of periods)
#' @return Matrix of regime probabilities over horizon
predict_regime <- function(hmm_model, horizon = 10) {
  # Current state probabilities
  current_probs <- hmm_model$current_probabilities
  
  # Transition matrix
  trans <- hmm_model$transition_matrix
  
  # Predict forward
  predictions <- matrix(0, nrow = horizon, ncol = hmm_model$n_states)
  predictions[1, ] <- as.numeric(current_probs %*% trans)
  
  for (h in 2:horizon) {
    predictions[h, ] <- predictions[h-1, ] %*% trans
  }
  
  colnames(predictions) <- paste0("state_", 1:hmm_model$n_states)
  
  list(
    probabilities = predictions,
    most_likely_state = apply(predictions, 1, which.max),
    horizon = horizon
  )
}


#' Select optimal number of states using information criteria
#'
#' @param returns Numeric vector of returns
#' @param max_states Maximum number of states to test
#' @return List with model comparison results
select_n_states <- function(returns, max_states = 5) {
  results <- data.frame(
    n_states = 2:max_states,
    aic = NA,
    bic = NA,
    log_likelihood = NA
  )
  
  models <- list()
  
  for (k in 2:max_states) {
    tryCatch({
      fit <- fit_hmm(returns, n_states = k)
      results$aic[k-1] <- fit$aic
      results$bic[k-1] <- fit$bic
      results$log_likelihood[k-1] <- fit$log_likelihood
      models[[k]] <- fit
    }, error = function(e) {
      message(paste("Failed to fit", k, "state model:", e$message))
    })
  }
  
  # Remove failed fits
  valid <- !is.na(results$aic)
  results <- results[valid, ]
  
  # Optimal by AIC and BIC
  optimal_aic <- results$n_states[which.min(results$aic)]
  optimal_bic <- results$n_states[which.min(results$bic)]
  
  list(
    comparison = results,
    optimal_by_aic = optimal_aic,
    optimal_by_bic = optimal_bic,
    recommendation = optimal_bic,  # BIC tends to be more parsimonious
    all_models = models
  )
}


# =============================================================================
# CHANGE POINT DETECTION
# =============================================================================

#' Detect structural breaks in time series
#'
#' Uses CUSUM-type test for parameter stability
#'
#' @param returns Numeric vector of returns
#' @param min_segment Minimum segment length
#' @return List with detected change points
detect_changepoints <- function(returns, min_segment = 20) {
  n <- length(returns)
  
  # Simple CUSUM approach
  mean_return <- mean(returns)
  cum_sum <- cumsum(returns - mean_return)
  
  # Standardize
  sigma <- sd(returns)
  cusum <- cum_sum / (sigma * sqrt(n))
  
  # Find potential change points (local extrema in CUSUM)
  potential_cp <- c()
  
  for (i in (min_segment+1):(n-min_segment)) {
    # Check if local maximum or minimum
    window <- cusum[(i-5):(i+5)]
    if (cusum[i] == max(window) || cusum[i] == min(window)) {
      # Check significance (simple threshold)
      if (abs(cusum[i]) > 1.5) {
        potential_cp <- c(potential_cp, i)
      }
    }
  }
  
  # Merge nearby changepoints
  if (length(potential_cp) > 1) {
    merged <- potential_cp[1]
    for (i in 2:length(potential_cp)) {
      if (potential_cp[i] - tail(merged, 1) > min_segment) {
        merged <- c(merged, potential_cp[i])
      }
    }
    potential_cp <- merged
  }
  
  # Calculate segment statistics
  segments <- list()
  breaks <- c(1, potential_cp, n)
  
  for (i in 1:(length(breaks)-1)) {
    seg_returns <- returns[breaks[i]:breaks[i+1]]
    segments[[i]] <- list(
      start = breaks[i],
      end = breaks[i+1],
      mean = mean(seg_returns) * 252,
      volatility = sd(seg_returns) * sqrt(252),
      n_obs = length(seg_returns)
    )
  }
  
  list(
    changepoints = potential_cp,
    n_changepoints = length(potential_cp),
    cusum_statistic = as.numeric(cusum),
    segments = segments,
    n_segments = length(segments)
  )
}


#' Detect volatility regime changes
#'
#' @param returns Numeric vector of returns
#' @param window Rolling window for volatility estimation
#' @param threshold Z-score threshold for regime change
#' @return List with volatility regime changes
detect_volatility_regimes <- function(returns, window = 21, threshold = 2) {
  n <- length(returns)
  
  # Rolling volatility
  roll_vol <- numeric(n)
  roll_vol[1:(window-1)] <- NA
  
  for (i in window:n) {
    roll_vol[i] <- sd(returns[(i-window+1):i]) * sqrt(252)
  }
  
  # Long-term average volatility
  long_term_vol <- mean(roll_vol, na.rm = TRUE)
  vol_of_vol <- sd(roll_vol, na.rm = TRUE)
  
  # Z-score of volatility
  vol_zscore <- (roll_vol - long_term_vol) / vol_of_vol
  
  # Classify regimes
  regimes <- rep("Normal", n)
  regimes[vol_zscore > threshold] <- "High Volatility"
  regimes[vol_zscore < -threshold] <- "Low Volatility"
  regimes[1:(window-1)] <- NA
  
  # Find regime changes
  regime_changes <- which(diff(as.numeric(factor(regimes))) != 0) + 1
  
  # Calculate regime statistics
  regime_stats <- list(
    low_vol = list(
      n_periods = sum(regimes == "Low Volatility", na.rm = TRUE),
      avg_vol = mean(roll_vol[regimes == "Low Volatility"], na.rm = TRUE),
      avg_return = mean(returns[regimes == "Low Volatility"], na.rm = TRUE) * 252
    ),
    normal = list(
      n_periods = sum(regimes == "Normal", na.rm = TRUE),
      avg_vol = mean(roll_vol[regimes == "Normal"], na.rm = TRUE),
      avg_return = mean(returns[regimes == "Normal"], na.rm = TRUE) * 252
    ),
    high_vol = list(
      n_periods = sum(regimes == "High Volatility", na.rm = TRUE),
      avg_vol = mean(roll_vol[regimes == "High Volatility"], na.rm = TRUE),
      avg_return = mean(returns[regimes == "High Volatility"], na.rm = TRUE) * 252
    )
  )
  
  list(
    regimes = regimes,
    rolling_volatility = roll_vol,
    volatility_zscore = vol_zscore,
    regime_changes = regime_changes,
    n_regime_changes = length(regime_changes),
    long_term_volatility = long_term_vol,
    current_regime = tail(regimes[!is.na(regimes)], 1),
    regime_statistics = regime_stats
  )
}


# =============================================================================
# MARKOV REGIME SWITCHING
# =============================================================================

#' Simple two-state Markov regime switching model
#'
#' State 1: Bull market (positive drift, low vol)
#' State 2: Bear market (negative drift, high vol)
#'
#' @param returns Numeric vector of returns
#' @return List with regime switching results
markov_regime_switching <- function(returns) {
  # Fit 2-state HMM
  hmm <- fit_hmm(returns, n_states = 2)
  
  # Identify bull/bear based on mean
  state_means <- sapply(hmm$state_statistics, function(x) x$mean)
  
  if (state_means[1] > state_means[2]) {
    bull_state <- 1
    bear_state <- 2
  } else {
    bull_state <- 2
    bear_state <- 1
  }
  
  # Relabel states
  bull_bear <- ifelse(hmm$states == bull_state, "Bull", "Bear")
  
  # Calculate transition probabilities
  p_bull_to_bear <- hmm$transition_matrix[bull_state, bear_state]
  p_bear_to_bull <- hmm$transition_matrix[bear_state, bull_state]
  
  # Expected duration
  bull_duration <- 1 / p_bull_to_bear
  bear_duration <- 1 / p_bear_to_bull
  
  # Smoothed probabilities
  bull_prob <- if (bull_state == 1) {
    hmm$state_probabilities[, 1]
  } else {
    hmm$state_probabilities[, 2]
  }
  
  list(
    regime = bull_bear,
    bull_probability = as.numeric(bull_prob),
    bull_state_stats = hmm$state_statistics[[paste0("state_", bull_state)]],
    bear_state_stats = hmm$state_statistics[[paste0("state_", bear_state)]],
    p_bull_to_bear = p_bull_to_bear,
    p_bear_to_bull = p_bear_to_bull,
    expected_bull_duration = bull_duration,
    expected_bear_duration = bear_duration,
    current_regime = tail(bull_bear, 1),
    current_bull_probability = tail(bull_prob, 1),
    hmm_model = hmm
  )
}


#' Generate regime-conditional forecasts
#'
#' @param regime_model Fitted regime model from markov_regime_switching
#' @param horizon Forecast horizon
#' @param n_simulations Number of Monte Carlo simulations
#' @return List with regime-conditional forecasts
regime_forecast <- function(regime_model, horizon = 21, n_simulations = 1000) {
  # Get parameters
  bull_stats <- regime_model$bull_state_stats
  bear_stats <- regime_model$bear_state_stats
  
  p_bb <- regime_model$p_bull_to_bear
  p_rb <- regime_model$p_bear_to_bull
  
  # Current state probability
  current_bull_prob <- regime_model$current_bull_probability
  
  # Monte Carlo simulation
  simulated_returns <- matrix(0, nrow = n_simulations, ncol = horizon)
  simulated_regimes <- matrix(0, nrow = n_simulations, ncol = horizon)
  
  for (sim in 1:n_simulations) {
    # Initial state based on current probability
    current_state <- ifelse(runif(1) < current_bull_prob, "Bull", "Bear")
    
    for (t in 1:horizon) {
      simulated_regimes[sim, t] <- ifelse(current_state == "Bull", 1, 0)
      
      # Generate return based on current state
      if (current_state == "Bull") {
        simulated_returns[sim, t] <- rnorm(1, 
                                           bull_stats$mean / 252,
                                           bull_stats$volatility / sqrt(252))
        # Transition
        if (runif(1) < p_bb) current_state <- "Bear"
      } else {
        simulated_returns[sim, t] <- rnorm(1,
                                           bear_stats$mean / 252,
                                           bear_stats$volatility / sqrt(252))
        # Transition
        if (runif(1) < p_rb) current_state <- "Bull"
      }
    }
  }
  
  # Calculate statistics
  cum_returns <- t(apply(simulated_returns, 1, function(x) cumprod(1 + x) - 1))
  
  list(
    expected_return = colMeans(cum_returns) * 100,
    return_std = apply(cum_returns, 2, sd) * 100,
    var_95 = apply(cum_returns, 2, quantile, 0.05) * 100,
    var_99 = apply(cum_returns, 2, quantile, 0.01) * 100,
    bull_probability = colMeans(simulated_regimes),
    horizon = 1:horizon
  )
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if (FALSE) {  # Set to TRUE to run examples
  set.seed(42)
  
  # Generate regime-switching data
  n <- 500
  regime <- rep(1, n)
  # Simulate regime changes
  for (i in 2:n) {
    if (regime[i-1] == 1) {
      regime[i] <- ifelse(runif(1) < 0.02, 2, 1)  # 2% chance of switching to bear
    } else {
      regime[i] <- ifelse(runif(1) < 0.05, 1, 2)  # 5% chance of switching to bull
    }
  }
  
  # Generate returns based on regime
  returns <- numeric(n)
  for (i in 1:n) {
    if (regime[i] == 1) {
      returns[i] <- rnorm(1, 0.0005, 0.01)  # Bull: positive drift, low vol
    } else {
      returns[i] <- rnorm(1, -0.001, 0.025)  # Bear: negative drift, high vol
    }
  }
  
  # Fit HMM
  cat("Hidden Markov Model:\n")
  hmm <- fit_hmm(returns, n_states = 2)
  cat("Current state:", hmm$current_state, "\n")
  cat("Transition matrix:\n")
  print(round(hmm$transition_matrix, 3))
  
  # Regime switching
  cat("\nMarkov Regime Switching:\n")
  mrs <- markov_regime_switching(returns)
  cat("Current regime:", mrs$current_regime, "\n")
  cat("Bull probability:", round(mrs$current_bull_probability, 3), "\n")
  cat("Expected bull duration:", round(mrs$expected_bull_duration, 1), "days\n")
  
  # Change point detection
  cat("\nChange Point Detection:\n")
  cp <- detect_changepoints(returns)
  cat("Number of changepoints:", cp$n_changepoints, "\n")
  
  # Volatility regimes
  cat("\nVolatility Regimes:\n")
  vr <- detect_volatility_regimes(returns)
  cat("Current vol regime:", vr$current_regime, "\n")
  cat("Regime changes:", vr$n_regime_changes, "\n")
  
  # Regime forecast
  cat("\nRegime Forecast (21 days):\n")
  fc <- regime_forecast(mrs, horizon = 21)
  cat("Expected return:", round(tail(fc$expected_return, 1), 2), "%\n")
  cat("Bull probability:", round(tail(fc$bull_probability, 1), 3), "\n")
}
