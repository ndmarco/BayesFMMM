#include <RcppArmadillo.h>
#include <cmath>

//' Updates pi
//'
//' @param alpha Double that is the hyperparameter
//' @param iter Int containing current MCMC iteration
//' @param l Int  containing element of pi to update
//' @param Z Matrix that contains the current values of the binary matrix
//' @param pi Matrix containg all samples of Pi
void updatePi(const double& alpha,
               const arma::mat& Z,
               const int& iter,
               const int& tot_mcmc_iters,
               arma::mat& pi){
  for(int l = 0; l < Z.n_cols; l++){
    pi(l, iter) = R::rbeta((alpha/ Z.n_cols) + arma::accu(Z.col(l)), Z.n_rows -
      arma::accu(Z.col(l)) + 1);
  }
  if(iter < (tot_mcmc_iters -1)){
    pi.col(iter + 1) = pi.col(iter);
  }
}

