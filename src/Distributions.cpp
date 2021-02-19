#include <RcppArmadillo.h>
#include <cmath>

//' Generates (degenerate) multivariate normal random variable
//'
//' @name Rmvnormal
//' @param mu Vector containing mean vector
//' @param sigma Matrix containing covariance matrix
//' @param U Matrix acting as a placeholder for SVD
//' @param S Vector acting as a placeholder for SVD
//' @param V Matrix acting as a placeholder for SVD
arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma,
                    arma::mat U,
                    arma::vec S,
                    arma::mat V,
                    const int rank)
{
  arma::svd(U, S, V, ((sigma + sigma.t())/2));
  for(int i = 0; i < rank - 1; i++){
    if(S(i) < 0)
    {
      S(i) = 0;
    }
  }

  arma::vec Y = arma::randn(rank);
  return mu + U.submat(0, 0, U.n_rows - 1, rank - 1) * arma::sqrt(arma::diagmat(S.subvec(0, rank - 1))) * Y;
}


//' Generates (degenerate) multivariate normal random variable
//'
//' @name Rmvnormal
//' @param mu Vector containing mean vector
//' @param sigma matrix containing covariance matrix
//' @export
// [[Rcpp::export]]

arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma,
                    const int rank)
{
  arma::mat U;
  arma::vec S;
  arma::mat V;
  return Rmvnormal(mu, sigma, U, S, V, rank);
}
