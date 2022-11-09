#ifndef BayesFMMM_UPDATE_ETA_H
#define BayesFMMM_UPDATE_ETA_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFMMM{

// Updates the eta parameters
//
// @name updateEta
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param tau_eta matrix containing current tau_eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing xi paramaters
// @param nu Matrix containing current nu paramaters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param sigma Double containing current sigma parameter
// @param iter Int containing MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param P Matrix containing tridiagonal P matrix
// @param X Matrix containing covariates
// @param b_1 Vector acting as a placeholder for mean vector
// @param B_1 Matrix acting as placeholder for covariance matrix
// @param nu Cube containing MCMC samples for nu
inline void updateEta(const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& B_obs,
                      const arma::mat& tau_eta,
                      const arma::cube& Phi,
                      const arma::field<arma::cube>& xi,
                      const arma::mat& nu,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const double& sigma,
                      const int& iter,
                      const int& tot_mcmc_iters,
                      const arma::mat& P,
                      const arma::mat& X,
                      arma::vec& b_1,
                      arma::mat& B_1,
                      arma::field<arma::cube>& eta){

  double ph = 0;
  // initialize P matrix
  for(int d = 0; d < eta(iter,0).n_cols; d++){
    for(int j = 0; j < eta(iter,0).n_slices; j++){
      b_1.zeros();
      B_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          for(int l = 0; l < y_obs(i,0).n_elem; l++){
            ph = 0;
            ph = y_obs(i,0)(l);
            B_1 = B_1 + Z(i,j) * Z(i,j) * X(i,d) * X(i,d) * (B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
            for(int r = 0; r < eta(iter,0).n_cols; r++){
              if(r != d){
                ph = ph - Z(i,j) * X(i,r) * arma::dot(eta(iter,0).slice(j).col(r), B_obs(i,0).row(l));
              }
            }
            for(int k = 0; k < Z.n_cols; k++){
              if(Z(i,k) != 0){
                if(k != j){
                  ph = ph - Z(i,k) * (arma::dot(eta(iter,0).slice(k) * X.row(i).t(),
                                      B_obs(i,0).row(l)));
                }
                ph = ph -  Z(i,k) * (arma::dot(nu.row(k),
                                     B_obs(i,0).row(l)));

                for(int n = 0; n < Phi.n_slices; n++){
                  ph = ph - Z(i,k) * chi(i,n) * (arma::dot(Phi.slice(n).row(k),
                                         B_obs(i,0).row(l)) + arma::dot(xi(iter,k).slice(n) * X.row(i).t(),
                                         B_obs(i,0).row(l)));
                }
              }
            }
            b_1 = b_1 + Z(i,j) * B_obs(i,0).row(l).t() * X(i,d) *  ph;
          }
        }
      }
      b_1 = b_1 / sigma;
      B_1 = B_1 / sigma;
      B_1 = B_1 + tau_eta(j,d) * P;
      B_1 = arma::pinv(B_1);
      B_1 = (B_1 + B_1.t())/2;
      eta(iter,0).slice(j).col(d) = arma::mvnrnd(B_1 * b_1, B_1);
    }
  }

  if(iter < (tot_mcmc_iters - 1)){
    eta(iter + 1,0) = eta(iter,0);
  }
}

// Updates the eta parameters using tempered transitions
//
// @name updateEtaTempered
// @param beta_i temperature at current step
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param tau_eta matrix containing current tau_eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing xi paramaters
// @param nu Matrix containing current nu paramaters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param sigma Double containing current sigma parameter
// @param iter Int containing MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param P Matrix containing tridiagonal P matrix
// @param X Matrix containing covariates
// @param b_1 Vector acting as a placeholder for mean vector
// @param B_1 Matrix acting as placeholder for covariance matrix
// @param nu Cube containing MCMC samples for nu
inline void updateEtaTempered(const double& beta_i,
                              const arma::field<arma::vec>& y_obs,
                              const arma::field<arma::mat>& B_obs,
                              const arma::mat& tau_eta,
                              const arma::cube& Phi,
                              const arma::field<arma::cube>& xi,
                              const arma::mat& nu,
                              const arma::mat& Z,
                              const arma::mat& chi,
                              const double& sigma,
                              const int& iter,
                              const int& tot_mcmc_iters,
                              const arma::mat& P,
                              const arma::mat& X,
                              arma::vec& b_1,
                              arma::mat& B_1,
                              arma::field<arma::cube>& eta){

  double ph = 0;
  // initialize P matrix
  for(int d = 0; d < eta(iter,0).n_cols; d++){
    for(int j = 0; j < eta(iter,0).n_slices; j++){
      b_1.zeros();
      B_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          for(int l = 0; l < y_obs(i,0).n_elem; l++){
            ph = 0;
            ph = y_obs(i,0)(l);
            B_1 = B_1 + Z(i,j) * Z(i,j) * X(i,d) * X(i,d) * (B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
            for(int r = 0; r < eta(iter,0).n_cols; r++){
              if(r != d){
                ph = ph - Z(i,j) * X(i,r) * arma::dot(eta(iter,0).slice(j).col(r), B_obs(i,0).row(l));
              }
            }
            for(int k = 0; k < Z.n_cols; k++){
              if(Z(i,k) != 0){
                if(k != j){
                  ph = ph - Z(i,k) * (arma::dot(eta(iter,0).slice(k) * X.row(i).t(),
                                      B_obs(i,0).row(l)));
                }
                ph = ph -  Z(i,k) * (arma::dot(nu.row(k),
                                     B_obs(i,0).row(l)));

                for(int n = 0; n < Phi.n_slices; n++){
                  ph = ph - Z(i,k) * chi(i,n) * (arma::dot(Phi.slice(n).row(k),
                                         B_obs(i,0).row(l)) + arma::dot(xi(iter,k).slice(n) * X.row(i).t(),
                                         B_obs(i,0).row(l)));
                }
              }
            }
            b_1 = b_1 + Z(i,j) * B_obs(i,0).row(l).t() * X(i,d) *  ph;
          }
        }
      }
      b_1 = b_1 * (beta_i / sigma);
      B_1 = B_1 * (beta_i / sigma);
      B_1 = B_1 + tau_eta(j,d) * P;
      B_1 = arma::pinv(B_1);
      B_1 = (B_1 + B_1.t())/2;
      eta(iter,0).slice(j).col(d) = arma::mvnrnd(B_1 * b_1, B_1);
    }
  }

  if(iter < (tot_mcmc_iters - 1)){
    eta(iter + 1,0) = eta(iter,0);
  }
}

// Updates the eta parameters for multivariate model
//
// @name updateEtaMV
// @param y_obs Matrix containing observed vectors
// @param tau_eta matrix containing current tau_eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing xi paramaters
// @param nu Matrix containing current nu paramaters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param sigma Double containing current sigma parameter
// @param iter Int containing MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param P Matrix containing tridiagonal P matrix
// @param X Matrix containing covariates
// @param b_1 Vector acting as a placeholder for mean vector
// @param B_1 Matrix acting as placeholder for covariance matrix
// @param nu Cube containing MCMC samples for nu
inline void updateEtaMV(const arma::mat& y_obs,
                        const arma::mat& tau_eta,
                        const arma::cube& Phi,
                        const arma::field<arma::cube>& xi,
                        const arma::mat& nu,
                        const arma::mat& Z,
                        const arma::mat& chi,
                        const double& sigma,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        const arma::mat& X,
                        arma::vec& b_1,
                        arma::mat& B_1,
                        arma::field<arma::cube>& eta){

  // initialize P matrix
  for(int d = 0; d < eta(iter,0).n_cols; d++){
    for(int j = 0; j < eta(iter,0).n_slices; j++){
      b_1.zeros();
      B_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          arma::vec ph = arma::zeros(y_obs.n_cols);
          ph = y_obs.row(i).t();
          B_1 = B_1 + Z(i,j) * Z(i,j) * X(i,d) * X(i,d) * arma::eye(y_obs.n_cols, y_obs.n_cols);
          for(int r = 0; r < eta(iter,0).n_cols; r++){
            if(r != d){
              ph = ph - Z(i,j) * X(i,r) * eta(iter,0).slice(j).col(r);
            }
          }
          for(int k = 0; k < Z.n_cols; k++){
            if(Z(i,k) != 0){
              if(k != j){
                ph = ph - Z(i,k) * eta(iter,0).slice(k) * X.row(i).t();
              }
              ph = ph - Z(i,k) * nu.row(k).t();

              for(int n = 0; n < Phi.n_slices; n++){
                ph = ph - Z(i,k) * chi(i,n) * (Phi.slice(n).row(k).t() +
                  xi(iter,k).slice(n) * X.row(i).t());
              }
            }
          }
          b_1 = b_1 + Z(i,j) * X(i,d) * ph;
        }
      }
      b_1 = b_1 / sigma;
      B_1 = B_1 / sigma;
      arma::mat D = arma::diagmat((1 / tau_eta(j,d)) * arma::ones(b_1.n_elem));
      B_1 = B_1 + D;
      B_1 = arma::pinv(B_1);
      B_1 = (B_1 + B_1.t())/2;
      eta(iter,0).slice(j).col(d) = arma::mvnrnd(B_1 * b_1, B_1);
    }
  }

  if(iter < (tot_mcmc_iters - 1)){
    eta(iter + 1,0) = eta(iter,0);
  }
}

// Updates the eta parameters for tempered multivariate model
//
// @name updateEtaTemperedMV
// @param beta_i temperature at current step
// @param y_obs Matrix containing observed vectors
// @param tau_eta matrix containing current tau_eta parameters
// @param Phi Cube containing current Phi parameters
// @param xi Field of cubes containing xi paramaters
// @param nu Matrix containing current nu paramaters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param sigma Double containing current sigma parameter
// @param iter Int containing MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param P Matrix containing tridiagonal P matrix
// @param X Matrix containing covariates
// @param b_1 Vector acting as a placeholder for mean vector
// @param B_1 Matrix acting as placeholder for covariance matrix
// @param eta Cube containing MCMC samples for eta
inline void updateEtaTemperedMV(const double& beta_i,
                                const arma::mat& y_obs,
                                const arma::mat& tau_eta,
                                const arma::cube& Phi,
                                const arma::field<arma::cube>& xi,
                                const arma::mat& nu,
                                const arma::mat& Z,
                                const arma::mat& chi,
                                const double& sigma,
                                const int& iter,
                                const int& tot_mcmc_iters,
                                const arma::mat& X,
                                arma::vec& b_1,
                                arma::mat& B_1,
                                arma::field<arma::cube>& eta){

  // initialize P matrix
  for(int d = 0; d < eta(iter,0).n_cols; d++){
    for(int j = 0; j < eta(iter,0).n_slices; j++){
      b_1.zeros();
      B_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          arma::vec ph = arma::zeros(y_obs.n_cols);
          ph = y_obs.row(i).t();
          B_1 = B_1 + Z(i,j) * Z(i,j) * X(i,d) * X(i,d) * arma::eye(y_obs.n_cols, y_obs.n_cols);
          for(int r = 0; r < eta(iter,0).n_cols; r++){
            if(r != d){
              ph = ph - Z(i,j) * X(i,r) * eta(iter,0).slice(j).col(r);
            }
          }
          for(int k = 0; k < Z.n_cols; k++){
            if(Z(i,k) != 0){
              if(k != j){
                ph = ph - Z(i,k) * eta(iter,0).slice(k) * X.row(i).t();
              }
              ph = ph - Z(i,k) * nu.row(k).t();

              for(int n = 0; n < Phi.n_slices; n++){
                ph = ph - Z(i,k) * chi(i,n) * (Phi.slice(n).row(k).t() +
                  xi(iter,k).slice(n) * X.row(i).t());
              }
            }
          }
          b_1 = b_1 + Z(i,j) * X(i,d) * ph;
        }
      }
      b_1 = b_1 * (beta_i / sigma);
      B_1 = B_1 * (beta_i / sigma);
      arma::mat D = arma::diagmat((1 / tau_eta(j,d)) * arma::ones(b_1.n_elem));
      B_1 = B_1 + D;
      B_1 = arma::pinv(B_1);
      B_1 = (B_1 + B_1.t())/2;
      eta(iter,0).slice(j).col(d) = arma::mvnrnd(B_1 * b_1, B_1);
    }
  }

  if(iter < (tot_mcmc_iters - 1)){
    eta(iter + 1,0) = eta(iter,0);
  }
}

}
#endif
