#ifndef UPDATE_CLASS_MEMBERSHIP_H
#define UPDATE_CLASS_MEMBERSHIP_H

#include <RcppArmadillo.h>
#include <cmath>

double g_ldet(const arma::mat& M);

double lpdf_z(const arma::mat& M,
              const arma::vec& m,
              const arma::vec& f_obs,
              const arma::vec& f_star,
              const arma::mat& S_obs,
              const arma::mat& phi,
              const arma::mat& nu,
              const double pi_l,
              const arma::mat& Z,
              const int i,
              const int j,
              arma::vec& mean_ph_obs);

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::field<arma::vec>& f_star,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::field<arma::mat>& S_star,
             const arma::cube& phi,
             const arma::mat& nu,
             const std::map<double, int> Map,
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
             const arma::mat& phi,
             const arma::mat& nu,
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
