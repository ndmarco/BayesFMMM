#include <RcppArmadillo.h>
#include <cmath>

arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma,
                    arma::mat U,
                    arma::vec S,
                    arma::mat V)
{
  arma::svd(U, S, V, ((sigma + sigma.t())/2));
  int m = 0;
  for(int i = 0; i < S.n_elem; i++){
    if(S(i) > 0)
    {
      m++;
    }else{
      S(i) = 0;
    }
  }
  arma::vec Y = arma::randn(m);
  return mu + U * arma::sqrt(arma::diagmat(S)) * Y;
}


//' Generates (degenerate) multivariate normal random variable
//'
//' @name Rmvnormal
//' @param n Int containing the number of draws
//' @param mu Vector containing mean vector
//' @param sigma matrix containing
//' @export
// [[Rcpp::export]]

arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma)
{
  arma::mat U;
  arma::vec S;
  arma::mat V;
  return Rmvnormal(mu, sigma, U, S, V);
}
