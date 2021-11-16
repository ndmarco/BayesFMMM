#include <RcppArmadillo.h>
#include <cmath>
#include "Distributions.H"

//' Updates the nu parameters
//'
//' @name updateNu
//' @param y_obs Field of vectors containing observed time points
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param tau Vector containing current tau parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma Double containing current sigma parameter
//' @param iter Int containing MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param P Matrix containing tridiagonal P matrix
//' @param b_1 Vector acting as a placeholder for mean vector
//' @param B_1 Matrix acting as placeholder for covariance matrix
//' @param nu Cube containing MCMC samples for nu
void updateNu(const arma::field<arma::vec>& y_obs,
              const arma::field<arma::mat>& B_obs,
              const arma::vec& tau,
              const arma::cube& Phi,
              const arma::mat& Z,
              const arma::mat& chi,
              const double& sigma,
              const int& iter,
              const int& tot_mcmc_iters,
              const arma::mat& P,
              arma::vec& b_1,
              arma::mat& B_1,
              arma::cube& nu){
  double ph = 0;
  // initialize P matrix
  for(int j = 0; j < nu.n_rows; j++){
    b_1.zeros();
    B_1.zeros();
    for(int i = 0; i < Z.n_rows; i++){
      if(Z(i,j) != 0){
        for(int l = 0; l < y_obs(i,0).n_elem; l++){
          ph = 0;
          ph = y_obs(i,0)(l);
          B_1 = B_1 + Z(i,j) * Z(i,j) * (B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
          for(int k = 0; k < nu.n_rows; k++){
            if(Z(i,k) != 0){
              if(k != j){
                ph = ph -  Z(i,k) * (arma::dot(nu.slice(iter).row(k),
                             B_obs(i,0).row(l)));
              }
              for(int n = 0; n < Phi.n_slices; n++){
                ph = ph - Z(i,k) * (chi(i,n) * arma::dot(Phi.slice(n).row(k),
                               B_obs(i,0).row(l)));
              }
            }
          }
          b_1 = b_1 + Z(i,j) * B_obs(i,0).row(l).t() * ph;
        }
      }
    }
    b_1 = b_1 / sigma;
    B_1 = B_1 / sigma;
    B_1 = B_1 + tau(j) * P;
    B_1 = arma::pinv(B_1);
    B_1 = (B_1 + B_1.t())/2;
    nu.slice(iter).row(j) = arma::mvnrnd(B_1 * b_1, B_1).t();
  }
  if(iter < (tot_mcmc_iters - 1)){
    nu.slice(iter + 1) = nu.slice(iter);
  }
}

//' Updates the nu parameters using tempered transitions
//'
//' @name updateNuTempered
//' @param beta_i temperature at current step
//' @param y_obs Field of vectors containing observed time points
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param tau Vector containing current tau parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma Double containing current sigma parameter
//' @param iter Int containing MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param P Matrix containing tridiagonal P matrix
//' @param b_1 Vector acting as a placeholder for mean vector
//' @param B_1 Matrix acting as placeholder for covariance matrix
//' @param nu Cube containing MCMC samples for nu
void updateNuTempered(const double& beta_i,
                      const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& B_obs,
                      const arma::vec& tau,
                      const arma::cube& Phi,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const double& sigma,
                      const int& iter,
                      const int& tot_mcmc_iters,
                      const arma::mat& P,
                      arma::vec& b_1,
                      arma::mat& B_1,
                      arma::cube& nu){
  double ph = 0;
  // initialize P matrix
  for(int j = 0; j < nu.n_rows; j++){
    b_1.zeros();
    B_1.zeros();
    for(int i = 0; i < Z.n_rows; i++){
      if(Z(i,j) != 0){
        for(int l = 0; l < y_obs(i,0).n_elem; l++){
          ph = 0;
          ph = y_obs(i,0)(l);
          B_1 = B_1 + Z(i,j) * Z(i,j) * (B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
          for(int k = 0; k < nu.n_rows; k++){
            if(Z(i,k) != 0){
              if(k != j){
                ph = ph - Z(i,k) * (arma::dot(nu.slice(iter).row(k),
                                      B_obs(i,0).row(l)));
              }
              for(int n = 0; n < Phi.n_slices; n++){
                ph = ph - Z(i,k) * (chi(i,n) * arma::dot(Phi.slice(n).row(k),
                               B_obs(i,0).row(l)));
              }
            }
          }
          b_1 = b_1 + Z(i,j) * B_obs(i,0).row(l).t() * ph;
        }
      }
    }
    b_1 = b_1 * (beta_i / sigma);
    B_1 = B_1 * (beta_i / sigma);
    B_1 = B_1 + tau(j) * P;
    B_1 = arma::pinv(B_1);
    B_1 = (B_1 + B_1.t())/2;
    nu.slice(iter).row(j) = arma::mvnrnd(B_1 * b_1, B_1).t();
  }
  if(iter < (tot_mcmc_iters - 1)){
    nu.slice(iter + 1) = nu.slice(iter);
  }
}

//' Updates the nu parameters for the multivariate model
//'
//' @name updateNuMV
//' @param y_obs Matrix containing observed vectors
//' @param tau Vector containing current tau parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma Double containing current sigma parameter
//' @param iter Int containing MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param P Matrix containing tridiagonal P matrix
//' @param b_1 Vector acting as a placeholder for mean vector
//' @param B_1 Matrix acting as placeholder for covariance matrix
//' @param nu Cube containing MCMC samples for nu
void updateNuMV(const arma::mat& y_obs,
                const arma::vec& tau,
                const arma::cube& Phi,
                const arma::mat& Z,
                const arma::mat& chi,
                const double& sigma,
                const int& iter,
                const int& tot_mcmc_iters,
                const arma::mat& P,
                arma::vec& b_1,
                arma::mat& B_1,
                arma::cube& nu){
  double ph = 0;
  // initialize P matrix
  for(int j = 0; j < nu.n_rows; j++){
    b_1.zeros();
    B_1.zeros();
    for(int i = 0; i < Z.n_rows; i++){
      if(Z(i,j) != 0){
        arma::vec ph = arma::zeros(y_obs.n_cols);
        ph = y_obs.row(i).t();
        B_1 = B_1 + Z(i,j) * Z(i,j) * arma::eye(y_obs.n_cols, y_obs.n_cols);
        for(int k = 0; k < nu.n_rows; k++){
          if(Z(i,k) != 0){
            if(k != j){
              ph = ph - Z(i,k) * nu.slice(iter).row(k).t();
            }
            for(int n = 0; n < Phi.n_slices; n++){
              ph = ph - Z(i,k) * chi(i,n) * Phi.slice(n).row(k).t();
            }
          }
        }
        b_1 = b_1 + Z(i,j) * ph;
      }
    }
    b_1 = b_1 / sigma;
    B_1 = B_1 / sigma;
    arma::mat D = arma::diagmat(tau);
    B_1 = B_1 + arma::pinv(D);
    B_1 = arma::pinv(B_1);
    B_1 = (B_1 + B_1.t())/2;
    nu.slice(iter).row(j) = arma::mvnrnd(B_1 * b_1, B_1).t();
  }
  if(iter < (tot_mcmc_iters - 1)){
    nu.slice(iter + 1) = nu.slice(iter);
  }
}

//' Updates the nu parameters for the tempered multivariate model
//'
//' @name updateNuMVTempered
//' @param beta_i temperature at current step
//' @param y_obs Matrix containing observed vectors
//' @param tau Vector containing current tau parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma Double containing current sigma parameter
//' @param iter Int containing MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param P Matrix containing tridiagonal P matrix
//' @param b_1 Vector acting as a placeholder for mean vector
//' @param B_1 Matrix acting as placeholder for covariance matrix
//' @param nu Cube containing MCMC samples for nu
void updateNuMVTempered(const double& beta_i,
                        const arma::mat& y_obs,
                        const arma::vec& tau,
                        const arma::cube& Phi,
                        const arma::mat& Z,
                        const arma::mat& chi,
                        const double& sigma,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        const arma::mat& P,
                        arma::vec& b_1,
                        arma::mat& B_1,
                        arma::cube& nu){
  double ph = 0;
  // initialize P matrix
  for(int j = 0; j < nu.n_rows; j++){
    b_1.zeros();
    B_1.zeros();
    for(int i = 0; i < Z.n_rows; i++){
      if(Z(i,j) != 0){
        arma::vec ph = arma::zeros(y_obs.n_cols);
        ph = y_obs.row(i).t();
        B_1 = B_1 + Z(i,j) * Z(i,j) * arma::eye(y_obs.n_cols, y_obs.n_cols);
        for(int k = 0; k < nu.n_rows; k++){
          if(Z(i,k) != 0){
            if(k != j){
              ph = ph - Z(i,k) * nu.slice(iter).row(k).t();
            }
            for(int n = 0; n < Phi.n_slices; n++){
              ph = ph - Z(i,k) * chi(i,n) * Phi.slice(n).row(k).t();
            }
          }
        }
        b_1 = b_1 + Z(i,j) * ph;
      }
    }
    b_1 = b_1 * (beta_i / sigma);
    B_1 = B_1 * (beta_i / sigma);
    arma::mat D = arma::diagmat(tau);
    B_1 = B_1 + arma::pinv(D);
    B_1 = arma::pinv(B_1);
    B_1 = (B_1 + B_1.t())/2;
    nu.slice(iter).row(j) = arma::mvnrnd(B_1 * b_1, B_1).t();
  }
  if(iter < (tot_mcmc_iters - 1)){
    nu.slice(iter + 1) = nu.slice(iter);
  }
}
