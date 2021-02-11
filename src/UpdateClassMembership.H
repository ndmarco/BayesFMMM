#ifndef UPDATE_CLASS_MEMBERSHIP_H
#define UPDATE_CLASS_MEMBERSHIP_H

#include <RcppArmadillo.h>
#include <cmath>
#include "computeMM.H"
#include "CalculateCov.H"

double g_ldet(const arma::mat& M,
              const int rank);

double lpdf_z(const arma::mat& M,
              const arma::vec& m,
              const arma::vec& f_obs,
              const arma::vec& f_star,
              const arma::mat& S_obs,
              const arma::mat& Cov,
              const arma::mat& nu,
              const arma::vec& pi,
              const arma::mat& Z,
              const int i,
              arma::mat& mp_inv,
              arma::vec& mean_ph_obs);

double lpdf_z(const arma::vec& f_obs,
              const arma::mat& S_obs,
              const arma::mat& Cov,
              const arma::mat& nu,
              const arma::vec& pi,
              const arma::mat& Z,
              const int i,
              arma::mat& ph,
              arma::vec& mean_ph_obs);

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::cube& Phi,
             const arma::mat& Rho,
             const arma::mat& nu,
             arma::field<arma::mat>& ph,
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
             arma::field<arma::mat> ph,
             arma::mat& Cov,
             arma::field<arma::vec>& mean_ph_obs,
             arma::mat& Z_ph,
             arma::cube& Z);

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::field<arma::vec>& f_star,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::field<arma::mat>& S_star,
             const arma::cube& Phi,
             const arma::mat& Rho,
             const arma::mat& nu,
             arma::mat& Cov,
             arma::field<arma::mat>& M,
             arma::field<arma::mat>& M_ph,
             arma::field<arma::vec>& m,
             arma::field<arma::vec>& m_ph,
             arma::field<arma::vec>& mean_ph_obs,
             arma::field<arma::vec>& mean_ph_star,
             arma::mat& Z_ph,
             arma::field<arma::mat>& mp_inv,
             arma::cube& Z);

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::field<arma::vec>& f_star,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::field<arma::mat>& S_star,
             const arma::mat& Phi,
             const arma::mat& nu,
             arma::mat& Cov,
             arma::field<arma::mat>& M,
             arma::field<arma::mat>& M_ph,
             arma::field<arma::vec>& m,
             arma::field<arma::vec>& m_ph,
             arma::field<arma::vec>& mean_ph_obs,
             arma::field<arma::vec>& mean_ph_star,
             arma::mat& Z_ph,
             arma::field<arma::mat>& mp_inv,
             arma::cube& Z);

#endif
