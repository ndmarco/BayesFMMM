#include <RcppArmadillo.h>
#include <cmath>

//' Updates the tau parameters
//'
//' @name updateTau
//' @param alpha double containing hyperparamter alpha
//' @param beta double containing hyperparameter beta
//' @param nu Matrix containing nu parameters
//' @param iter int containing MCMC sample
//' @param P matrix acting as a placeholder for penalization-smoothing matrix
//' @param tau matrix acting as placeholder for MCMC samples
void updateTau(const double& alpha,
               const double& beta,
               const arma::mat& nu,
               const int& iter,
               arma::mat& P,
               arma::mat& tau){
  P.zeros();
  for(int j = 0; j < nu.n_rows; j++){
    P(0,0) = 1;
    P(nu.n_rows - 1, nu.n_rows - 1) = 1;
    if(j > 0){
      P(j,j) = 2;
      P(j-1,j) = -1;
      P(j,j-1) = -1;
    }
  }
  for(int j = 0; j < nu.n_rows; j++){
    tau(iter,j) = R::rgamma( alpha + (nu.n_cols / 2),
        1 /(beta + 0.5 * arma::dot( P * nu.row(j), nu.row(j))));
  }
}
