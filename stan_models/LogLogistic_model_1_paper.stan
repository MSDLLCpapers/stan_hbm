// log-Logistic survival model

functions {
  // Defines the log hazard
  vector log_h (vector t, real shape, vector scale) {
    vector[num_elements(t)] log_h;
    for (i in 1:num_elements(t)) {
      log_h[i] = log(shape)-log(scale[i])+(shape-1)*(log(t[i])-log(scale[i]))-log(1+pow((t[i]/scale[i]),shape));
    }
    return log_h;
  }
  
  // Defines the log survival
  vector log_S (vector t, real shape, vector scale) {
    vector[num_elements(t)] log_S;
    for (i in 1:num_elements(t)) {
      log_S[i] = -log(1+pow((t[i]/scale[i]),shape));
    }
    return log_S;
  }
  
  // Defines the sampling distribution
  real surv_loglogistic_lpdf (vector t, vector d, real shape, vector scale) {
    vector[num_elements(t)] log_lik;
    real prob;
    log_lik = d .* log_h(t,shape,scale) + log_S(t,shape,scale);
    prob = sum(log_lik);
    return prob;
  }
}

data {
  int n;                  // number of observations
  vector[n] t;            // observed times
  vector[n] d;            // censoring indicator (1=observed, 0=censored)
  int tumor_type[n];      // vector of indicators for the tumor type (1,2,...,J)
  int J;                            // number of tumor types included in the analysis
  int H;                  // number of covariates
  matrix[n,H] X;          // matrix of covariates (with n rows and H columns)
  vector[H] mu_beta;	  // mean of the covariates coefficients
  vector<lower=0> [H] sigma_beta;   // sd of the covariates coefficients
  real<lower=0> a_alpha;
  real<lower=0> b_alpha;
  
}

parameters {
  vector[H] beta;         // Coefficients in the linear predictor (including intercept)
  real<lower=0> alpha;    // shape parameter
  vector[J] gamma;                  // vector of *random effects* associated with various tumours (account for clustering)
  real<lower=0> sigma_gamma;        // sd for the random effects
}

transformed parameters {
  vector[n] linpred;
  vector[n] mu;
  linpred = X*beta;
  for (i in 1:n) {
    mu[i] = exp(linpred[i] + gamma[tumor_type[i]]);// NB: this adds the *random effects* part to the linear predictor before rescaling to the location
  }
}

model {
  alpha ~ gamma(a_alpha,b_alpha);       // prior for the shape parameter
  beta ~ normal(mu_beta,sigma_beta);    // prior for the regression coefficients in the linear predictor (fixed effects)
  gamma ~ normal(0,sigma_gamma);        // prior for the random effects
  sigma_gamma ~ normal(0,10);           // **half**-normal prior on the sd of the random effects (NB the variable is bounded by 0!)
  
  t ~ surv_loglogistic(d,alpha,mu);
}

generated quantities {
  real log_lik;
  log_lik = surv_loglogistic_lpdf(t | d, alpha, mu);
}
