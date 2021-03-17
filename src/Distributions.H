#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <RcppArmadillo.h>
#include <cmath>

arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma);

#endif
