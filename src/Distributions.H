#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <RcppArmadillo.h>
#include <cmath>

arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma,
                    arma::mat U,
                    arma::vec S,
                    arma::mat V,
                    const int rank);

arma::vec Rmvnormal(arma::vec mu,
                    arma::mat sigma,
                    const int rank);
#endif
