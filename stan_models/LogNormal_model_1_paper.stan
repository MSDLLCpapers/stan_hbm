// log-Normal survival model

functions {
  // Defines the log survival
  vector log_S (vector t, vector mean, real sd) {
    vector[num_elements(t)] log_S;
    for (i in 1:num_elements(t)) {
      log_S[i] = log(1-Phi((log(t[i])-mean[i])/sd));
    }
    return log_S;
  }
  
  // Defines the log hazard
  vector log_h (vector t, vector mean, real sd) {
    vector[num_elements(t)] log_h;
    vector[num_elements(t)] ls;
    ls = log_S(t,mean,sd);
    for (i in 1:num_elements(t)) {
      log_h[i] = lognormal_lpdf(t[i]|mean[i],sd) - ls[i];
    }
    return log_h;
  }
  
  // Defines the sampling distribution
  real surv_lognormal_lpdf (vector t, vector d, vector mean, real sd) {
    vector[num_elements(t)] log_lik;
    real prob;
    log_lik = d .* log_h(t,mean,sd) + log_S(t,mean,sd);
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
  vector[H] mu_beta;	    // mean of the covariates coefficients
  vector<lower=0> [H] sigma_beta;   // sd of the covariates coefficients
  real a_alpha;			      // lower bound for the sd of the data			  
  real b_alpha;			      // upper bound for the sd of the data
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
  for (i in 1:n) {
    mu[i] = exp(linpred[i] + gamma[tumor_type[i]]);// NB: this adds the *random effects* part to the linear predictor before rescaling to the location
  }
}



model {
  alpha ~ gamma(a_alpha,b_alpha);       // prior for the shape parameter
  beta ~ normal(mu_beta,sigma_beta);    // prior for the regression coefficients in the linear predictor (fixed effects)
  gamma ~ normal(0,sigma_gamma);        // prior for the random effects
  sigma_gamma ~ normal(0,10);           // **half**-normal prior on the sd of the random effects (NB the variable is bounded by 0!)
  
  //alpha ~ uniform(a_alpha,b_alpha); alternative prior
  t ~ surv_lognormal(d,mu,alpha);
}

generated quantities {
  real log_lik;
  log_lik = surv_lognormal_lpdf(t | d, mu, alpha);
}
