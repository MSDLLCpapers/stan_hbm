/* 
 * Weibull survival model including clustering by tumor types
*/

// Utility functions to define the sampling distribution accounting for censoring
functions {
  // Defines the log hazard
  vector log_h (vector t, real shape, vector scale) {
    vector[num_elements(t)] log_h;
    log_h = log(shape)+(shape-1)*log(t ./ scale)-log(scale);
    return log_h;
  }
  
  // Defines the log survival
  vector log_S (vector t, real shape, vector scale) {
    vector[num_elements(t)] log_S;
    for (i in 1:num_elements(t)) {
      log_S[i] = -pow((t[i]/scale[i]),shape);
    }
    return log_S;
  }
  
  // Defines the sampling distribution
  real surv_weibullAF_lpdf (vector t, vector d, real shape, vector scale) {
    vector[num_elements(t)] log_lik;
    real prob;
    log_lik = d .* log_h(t,shape,scale) + log_S(t,shape,scale);
    prob = sum(log_lik);
    return prob;
  }
}

// Model code
data {
  int n;                            // number of observations
  vector[n] t;                      // observed times
  vector[n] d;                      // event indicator (1= fully observed; 0=censored)
  int tumor_type[n];                // vector of indicators for the tumor type (1,2,...,J)
  int J;                            // number of tumor types included in the analysis
  int H;                            // number of covariates
  matrix[n,H] X;                    // matrix of covariates (with n rows and H columns)
  vector[H] mu_beta;	              // mean of the covariates coefficients
  vector<lower=0> [H] sigma_beta;   // sd of the covariates coefficients
  real<lower=0> a_alpha;            // first parameter for the shape distribution
  real<lower=0> b_alpha;            // second parameter for the shape distribution
}

parameters {
  vector[H] beta;                   // Coefficients for the *fixed effects* part of the linear predictor (including intercept)
  real<lower=0> alpha;              // shape parameter
  vector[J] gamma;                  // vector of *random effects* associated with various tumours (account for clustering)
  real<lower=0> sigma_gamma;        // sd for the random effects
}

transformed parameters {
  vector[n] linpred;                // "linear predictor". This defines the location parameter as a function of the covariates (on the log scale)
  vector[n] mu;                     // rescaled linear predictor (= location parameter)
  linpred = X*beta;                 // regression for the *fixed effect* part of the model
  for (i in 1:n) {                  // This adds the *random effects* part to the linear predictor before rescaling to the location
      mu[i] = exp(linpred[i] + gamma[tumor_type[i]]);
  }
}

model {
  alpha ~ gamma(a_alpha,b_alpha);       // prior for the shape parameter
  beta ~ normal(mu_beta,sigma_beta);    // prior for the regression coefficients in the linear predictor (fixed effects)
  gamma ~ normal(0,sigma_gamma);        // prior for the random effects
  sigma_gamma ~ normal(0,10);           // **half**-normal prior on the sd of the random effects
  t ~ surv_weibullAF(d,alpha,mu);
}

generated quantities {
  real log_lik;
  log_lik = surv_weibullAF_lpdf(t | d, alpha, mu);
}
