#ifndef B_SPLINES_H
#define B_SPLINES_H

#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>

arma::field<arma::mat> TensorBSpline(const arma::field<arma::mat>& t_obs,
                                     const int n_funct,
                                     const arma::vec& basis_degree,
                                     const arma::mat& boundary_knots,
                                     const arma::field<arma::vec>& internal_knots);

arma::mat GetP(const arma::vec& basis_degree,
               const arma::field<arma::vec>& internal_knots);


#endif
