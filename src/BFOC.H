#ifndef BFOC_H
#define BFOC_H

#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "Distributions.H"
#include "UpdateClassMembership.H"
#include "UpdatePi.H"
#include "UpdatePhi.H"
#include "UpdateDelta.H"
#include "UpdateA.H"
#include "UpdateGamma.H"
#include "UpdateNu.H"
#include "UpdateTau.H"
#include "UpdateSigma.H"
#include "UpdateChi.H"
#include "CalculateLikelihood.H"

Rcpp::List BFOC(const arma::field<arma::vec> y_obs,
                const arma::field<arma::vec> t_obs,
                const double n_funct,
                const int K,
                const int P,
                const int M,
                const int tot_mcmc_iters,
                const arma::field<arma::vec> t_star,
                const double nu_1,
                const double rho,
                const double alpha_3,
                const arma::vec a_12,
                const double alpha1l,
                const double alpha2l,
                const double beta1l,
                const double beta2l,
                const double var_epsilon1,
                const double var_epsilon2,
                const double alpha,
                const double beta,
                const double alpha_0,
                const double beta_0);


#endif
