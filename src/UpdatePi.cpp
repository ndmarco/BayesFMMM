#include <RcppArmadillo.h>
#include <cmath>

//' Updates pi
//'
//' @param alpha Double that is the hyperparameter
//' @param iter Int containing current MCMC iteration
//' @param l Int  containing element of pi to update
//' @param Z Matrix that contains the current values of the binary matrix
//' @param pi Matrix containg all samples of Pi
void update_pi(const double& alpha,
                 const int& iter,
                 const arma::mat& Z,
                 arma::mat& pi){
  for(int l = 0; l < Z.n_cols; l++){
    pi(iter, l) = R::rbeta((alpha/ Z.n_cols) + arma::accu(Z.col(l)), Z.n_rows -
      arma::accu(Z.col(l)) + 1);
  }
}

