#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Updates the parameters pi_1 through pi_K
//'
//' @param Z Binary matrix of containing inclusion/exclusion to the various groups
//' @param alpha Hyperparameter for distribution of pi_i
//' @param K Number of clusters
//' @param pi_l Placeholder of new draw
//' @return updated pi_l
//' @export
// [[Rcpp::export]]

void updatePi(const arma::mat& Z, const double alpha, const int K, const int N, const int iter, arma::mat& pi)
{
  for(int i = 0; i < K; i++)
  {
    pi(iter, i) = R::rbeta((alpha/K) + arma::accu(Z.col(i)), Z.n_rows - arma::accu(Z.col(i)) + 1);
  }
}

