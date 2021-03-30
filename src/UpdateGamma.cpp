#include <RcppArmadillo.h>
#include <cmath>

//' Updates the gamma parameters
//'
//' @name updateGamma
//' @param nu double containing hyperparameter
//' @param iter int containing MCMC iteration
//' @param delta Matrix containing current values of delta
//' @param phi Cube containing current values of phi
//' @param Z matrix containing current values of class inclusion
//' @param gamma Field of cubes contianing MCMC samples for gamma
void updateGamma(const double& nu,
                 const arma::vec& delta,
                 const arma::cube& phi,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 arma::field<arma::cube>& gamma){
  double placeholder = 1;
  for(int i = 0; i < phi.n_rows; i++){
    for(int l = 0; l < phi.n_cols; l++){
      placeholder = 1;
      for(int j = 0; j < phi.n_slices; j++){
        placeholder = placeholder * delta(j);
        gamma(iter,0)(i,l,j) = R::rgamma((nu + 1)/2, 2/(nu + placeholder *
          (phi(i,l,j) * phi(i,l,j))));
      }
    }
  }
  if(iter < (tot_mcmc_iters) - 1){
    gamma(iter + 1, 0) = gamma(iter, 0);
  }
}

