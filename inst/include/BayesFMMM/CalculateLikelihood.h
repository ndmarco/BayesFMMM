#ifndef BayesFMMM_CALCULATE__LIKELIHOOD_H
#define BayesFMMM_CALCULATE__LIKELIHOOD_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFMMM {
// Calculates the log likelihood of the model
//
// @name calcLikelihood
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param sigma Double containing current sigma parameter
// @return log_lik Double containing the log likelihood of the model
inline double calcLikelihood(const arma::field<arma::vec>& y_obs,
                             const arma::field<arma::mat>& B_obs,
                             const arma::mat& nu,
                             const arma::cube& Phi,
                             const arma::mat& Z,
                             const arma::mat& chi,
                             const double& sigma){
  double log_lik = 0;
  double mean = 0;
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      mean = 0;
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          mean = mean + Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            mean = mean + Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k),
                            B_obs(i,0).row(l));
          }
        }
      }
      log_lik = log_lik + R::dnorm(y_obs(i,0)(l), mean, std::sqrt(sigma), true);
    }
  }
  return log_lik;
}

// Calculates the second term of the DIC expression
//
// @name calcDIC2
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param i Int containing function number of interest
// @param j Int containing the time point of the ith function of interest
// @param sigma Double containing current sigma parameter
// @return lik Double containing likelihood
inline double calcDIC2(const arma::vec& y_obs,
                       const arma::mat& B_obs,
                       const arma::mat& nu,
                       const arma::cube& Phi,
                       const arma::mat& Z,
                       const arma::mat& chi,
                       const int i,
                       const int j,
                       const double& sigma){
  double mean = 0;
  for(int k = 0; k < Z.n_cols; k++){
    if(Z(i,k) != 0){
      mean = mean + Z(i,k) * arma::dot(nu.row(k), B_obs.row(j));
      for(int n = 0; n < Phi.n_slices; n++){
        mean = mean + Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k),
                        B_obs.row(j));
      }
    }
  }
  double lik = R::dnorm(y_obs(j), mean, std::sqrt(sigma), false);
  return lik;
}


// Calculates the second term of the DIC expression for covariate adjusted model
//
// @name calcDIC2CovariateAdj
// @param y_obs Field of vectors containing observed time points
// @param X Matrix containing the covariates of interest
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing xi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter Int containing the MCMC iteration of interest for xi parameters
// @param i Int containing function number of interest
// @param j Int containing the time point of the ith function of interest
// @param sigma Double containing current sigma parameter
// @return lik Double containing likelihood
inline double calcDIC2CovariateAdj(const arma::vec& y_obs,
                                   const arma::mat& X,
                                   const arma::mat& B_obs,
                                   const arma::mat& nu,
                                   const arma::cube& eta,
                                   const arma::cube& Phi,
                                   const arma::field<arma::cube>& xi,
                                   const arma::mat& Z,
                                   const arma::mat& chi,
                                   const int iter,
                                   const int i,
                                   const int j,
                                   const double& sigma){
  double mean = 0;
  for(int k = 0; k < Z.n_cols; k++){
    if(Z(i,k) != 0){
      mean = mean + Z(i,k) * arma::dot(nu.row(k) + (eta.slice(k) * X.row(i).t()).t(), B_obs.row(j));
      for(int n = 0; n < Phi.n_slices; n++){
        mean = mean + Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k) + (xi(iter,k).slice(n) * X.row(i).t()).t(),
                        B_obs.row(j));
      }
    }
  }
  double lik = R::dnorm(y_obs(j), mean, std::sqrt(sigma), false);
  return lik;
}

// Calculates the log likelihood of the multivariate model
//
// @name calcLikelihoodMV
// @param y_obs Matrix of observations containing observed vectors
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param sigma Double containing current sigma parameter
// @return log_lik Double containing the log likelihood of the model
inline double calcLikelihoodMV(const arma::mat& y_obs,
                               const arma::mat& nu,
                               const arma::cube& Phi,
                               const arma::mat& Z,
                               const arma::mat& chi,
                               const double& sigma){
  double log_lik = 0;
  arma::vec mean = arma::zeros(y_obs.n_cols);
  for(int i = 0; i < chi.n_rows; i++){
    mean = arma::zeros(y_obs.n_cols);
    for(int k = 0; k < Z.n_cols; k++){
      if(Z(i,k) != 0){
        mean = mean + Z(i,k) * nu.row(k).t();
        for(int n = 0; n < Phi.n_slices; n++){
          mean = mean + Z(i,k) * chi(i,n) * Phi.slice(n).row(k).t();
        }
      }
    }
    log_lik = log_lik - ((y_obs.n_cols / 2) * std::log(2 * arma::datum::pi * sigma)) - (
      (1 / (sigma * 2)) * arma::dot(y_obs.row(i).t() - mean, y_obs.row(i).t() - mean));
  }
  return log_lik;
}

// Calculates the second term of the DIC expression for multivariate model
//
// @name calcDIC2MV
// @param y_obs Field of vectors containing observed time points
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param i Int containing vector number of interest
// @param sigma Double containing current sigma parameter
// @return lik Double containing likelihood
inline double calcDIC2MV(const arma::rowvec& y_obs,
                         const arma::mat& nu,
                         const arma::cube& Phi,
                         const arma::mat& Z,
                         const arma::mat& chi,
                         const int i,
                         const double& sigma){
  arma::vec mean = arma::zeros(y_obs.n_elem);
  for(int k = 0; k < Z.n_cols; k++){
    if(Z(i,k) != 0){
      mean = mean + Z(i,k) * nu.row(k).t();
      for(int n = 0; n < Phi.n_slices; n++){
        mean = mean + Z(i,k) * chi(i,n) * Phi.slice(n).row(k).t();
      }
    }
  }
  double lik = 1;
  for(int j = 0; j < y_obs.n_elem; j++){
    lik = lik * R::dnorm(y_obs(j), mean(j), std::sqrt(sigma), false);
  }

  return lik;
}


// Calculates the second term of the DIC expression for covariate adjusted multivariate model
//
// @name calcDIC2MVCovariateAdj
// @param y_obs Field of vectors containing observed time points
// @param X Matrix containing the covariates of interest
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing xi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter Int containing the MCMC iteration of interest for xi parameters
// @param i Int containing vector number of interest
// @param sigma Double containing current sigma parameter
// @return lik Double containing likelihood
inline double calcDIC2MVCovariateAdj(const arma::rowvec& y_obs,
                                     const arma::mat& X,
                                     const arma::mat& nu,
                                     const arma::cube& eta,
                                     const arma::cube& Phi,
                                     const arma::field<arma::cube>& xi,
                                     const arma::mat& Z,
                                     const arma::mat& chi,
                                     const int iter,
                                     const int i,
                                     const double& sigma){
  arma::vec mean = arma::zeros(y_obs.n_elem);
  for(int k = 0; k < Z.n_cols; k++){
    if(Z(i,k) != 0){
      mean = mean + Z(i,k) * (nu.row(k).t() + eta.slice(k) * X.row(i).t());
      for(int n = 0; n < Phi.n_slices; n++){
        mean = mean + Z(i,k) * chi(i,n) * (Phi.slice(n).row(k).t() + (xi(iter,k).slice(n) * X.row(i).t()));
      }
    }
  }
  double lik = 1;
  for(int j = 0; j < y_obs.n_elem; j++){
    lik = lik * R::dnorm(y_obs(j), mean(j), std::sqrt(sigma), false);
  }

  return lik;
}
// Calculates the log likelihood of the covariate adjusted partial membership model for functional data
//
// @name calcLikelihoodCovariateAdj
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing current xi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter int containing current iteration number
// @param X Matrix containing covariates
// @param sigma Double containing current sigma parameter
// @return log_lik Double containing the log likelihood of the model
inline double calcLikelihoodCovariateAdj(const arma::field<arma::vec>& y_obs,
                                         const arma::field<arma::mat>& B_obs,
                                         const arma::mat& nu,
                                         const arma::cube& eta,
                                         const arma::cube& Phi,
                                         const arma::field<arma::cube>& xi,
                                         const arma::mat& Z,
                                         const arma::mat& chi,
                                         const int& iter,
                                         const arma::mat& X,
                                         const double& sigma){
  double log_lik = 0;
  double mean = 0;
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      mean = 0;
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          mean = mean + Z(i,k) * arma::dot(nu.row(k).t() +
            (eta.slice(k) * X.row(i).t()), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            mean = mean + Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k).t() +
              (xi(iter,k).slice(n) * X.row(i).t()), B_obs(i,0).row(l));
          }
        }
      }
      log_lik = log_lik + R::dnorm(y_obs(i,0)(l), mean, std::sqrt(sigma), true);
    }
  }
  return log_lik;
}

// Calculates the log likelihood of the covariate adjusted multivariate model
//
// @name calcLikelihoodMVCovariateAdj
// @param y_obs Matrix of observations containing observed vectors
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing current xi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter int containing current iteration number
// @param X Matrix containing covariates
// @param sigma Double containing current sigma parameter
// @return log_lik Double containing the log likelihood of the model
inline double calcLikelihoodMVCovariateAdj(const arma::mat& y_obs,
                                           const arma::mat& nu,
                                           const arma::cube& eta,
                                           const arma::cube& Phi,
                                           const arma::field<arma::cube>& xi,
                                           const arma::mat& Z,
                                           const arma::mat& chi,
                                           const int& iter,
                                           const arma::mat& X,
                                           const double& sigma){
  double log_lik = 0;
  arma::vec mean = arma::zeros(y_obs.n_cols);
  for(int i = 0; i < chi.n_rows; i++){
    mean = arma::zeros(y_obs.n_cols);
    for(int k = 0; k < Z.n_cols; k++){
      if(Z(i,k) != 0){
        mean = mean + Z(i,k) * (nu.row(k).t() + (eta.slice(k) * X.row(i).t()));
        for(int n = 0; n < Phi.n_slices; n++){
          mean = mean + Z(i,k) * chi(i,n) * (Phi.slice(n).row(k).t() +
            (xi(iter,k).slice(n) * X.row(i).t()));
        }
      }
    }
    log_lik = log_lik - ((y_obs.n_cols / 2) * std::log(2 * arma::datum::pi * sigma)) - (
      (1 / (sigma * 2)) * arma::dot(y_obs.row(i).t() - mean, y_obs.row(i).t() - mean));
  }
  return log_lik;
}

// Calculates the CPO of the covariate adjusted partial membership model for functional data
//
// @name calcLikelihoodCPO
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing current xi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter int containing current iteration number
// @param X Matrix containing covariates
// @param sigma Double containing current sigma parameter
// @return log_lik Double containing the log likelihood of the model
inline arma::vec calcLikelihoodCPO(const arma::field<arma::vec>& y_obs,
                                   const arma::field<arma::mat>& B_obs,
                                   const arma::cube& nu,
                                   const arma::field<arma::cube>& eta,
                                   const arma::field<arma::cube>& Phi,
                                   const arma::field<arma::cube>& xi,
                                   const arma::cube& Z,
                                   const arma::cube& chi,
                                   const arma::mat& X,
                                   const arma::vec& sigma,
                                   const int& mcmc_iters,
                                   const double& burnin_prop){
  double N = Z.n_rows;
  arma::vec CPO = arma::zeros(N);
  for(int i = 0; i < N; i++){
    arma::vec logl_i = arma::zeros(std::ceil((1-burnin_prop) * mcmc_iters));
    for(int l = std::floor(burnin_prop * mcmc_iters); l < mcmc_iters; l++){
      arma::vec mean = arma::zeros(y_obs(i,0).n_elem);
      arma::mat cov = arma::zeros(y_obs(i,0).n_elem, y_obs(i,0).n_elem);
      for(int k = 0; k < Z.n_cols; k++){
        mean = mean + Z(i,k,l) * B_obs(i,0) * (nu.slice(l).row(k).t() + (eta(l,0).slice(k) * X.row(i).t()));
        for(int k1 = 0; k1< Z.n_cols; k1++){
          for(int m = 0; m < chi.n_cols; m++){
            cov = cov + Z(i,k,l) * Z(i,k1,l) * (B_obs(i,0) * (Phi(l,0).slice(m).row(k).t() + xi(l,k).slice(m) * X.row(i).t()) *
              (Phi(l,0).slice(m).row(k1).t() + xi(l,k1).slice(m) * X.row(i).t()).t() * B_obs(i,0).t());
          }
        }
      }
      cov.diag() = cov.diag() + sigma(l);

      logl_i(l- std::floor(burnin_prop * mcmc_iters)) = -(0.5*y_obs(i,0).n_elem)*std::log(2 * arma::datum::pi) -
        0.5*arma::log_det_sympd(cov) - 0.5*arma::dot((y_obs(i,0) - mean).t() * arma::inv_sympd(cov), (y_obs(i,0) - mean));
    }
    double logl_i_min = logl_i.min();
    double ph = 0;
    for(int l = 0; l < logl_i.n_elem; l++){
      ph = ph + std::exp(logl_i_min - logl_i(l));
    }
    CPO(i) = std::log(logl_i.n_elem) + logl_i_min - std::log(ph);
    Rcpp::Rcout << "CPO calculated for observation " << i + 1 << "\n";
  }
  return CPO;
}

}
#endif
