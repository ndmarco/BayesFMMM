#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <RcppArmadillo.h>
#include <cmath>

arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma);

double logGamma(double x);

arma::vec rdirichlet(arma::vec alpha);

double calc_lB(const arma::vec& alpha);

#endif
