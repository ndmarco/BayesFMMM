#include <RcppArmadillo.h>
#include <cmath>

//' Generates multivariate normal random variable
//'
//' @name Rmvnormal
//' @param mu Vector containing mean vector
//' @param sigma Matrix containing covariance matrix
arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma){
  arma::vec Y = arma::randn(sigma.n_cols);
  return mu + arma::chol(sigma) * Y;
}

//' Calculates log gamma of a double
//'
//' @name logGamma
//' @param x Double
//' @return log(x)
double logGamma(double x){
  return log(tgamma(x));
}


arma::vec rdirichlet(arma::vec alpha){

    arma::vec distribution(alpha.n_elem, arma::fill::zeros);

  double sum_term = 0;

  for (int j = 0; j < alpha.n_elem; ++j) {
    double gam = R::rgamma(alpha[j],1.0);
    distribution(j) = gam;
    sum_term += gam;
  }

  for (int j = 0; j < alpha.n_elem; ++j) {
    distribution(j) = distribution(j) / sum_term;
  }

  return distribution;
}
