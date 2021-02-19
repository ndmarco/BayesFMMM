#ifndef UPDATE_CLASS_MEMBERSHIP_H
#define UPDATE_CLASS_MEMBERSHIP_H

#include <RcppArmadillo.h>
#include <cmath>
#include "computeMM.H"
#include "CalculateCov.H"

double g_ldet(const arma::vec& S);

double lpdf_z(const arma::vec& f_obs,
              const arma::mat& S_obs,
              const arma::mat& Cov,
              const arma::mat& nu,
              const arma::vec& pi,
              const arma::mat& Z,
              const int rank,
              const int i,
              arma::field<arma::mat>& mean_UV,
              arma::field<arma::vec>& mean_S,
              arma::vec& mean_ph_obs);


void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::cube& Phi,
             const arma::mat& Rho,
             const arma::mat& nu,
             const int rank,
             arma::field<arma::mat>& mean_UV,
             arma::field<arma::vec>& mean_S,
             arma::mat& Cov,
             arma::field<arma::vec>& mean_ph_obs,
             arma::mat& Z_ph,
             arma::cube& Z);

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::mat& Phi,
             const arma::mat& nu,
             const int rank,
             arma::field<arma::mat>& mean_UV,
             arma::field<arma::vec>& mean_S,
             arma::mat& Cov,
             arma::field<arma::vec>& mean_ph_obs,
             arma::mat& Z_ph,
             arma::cube& Z);

#endif
