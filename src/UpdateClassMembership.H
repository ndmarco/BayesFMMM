#ifndef UPDATE_CLASS_MEMBERSHIP_H
#define UPDATE_CLASS_MEMBERSHIP_H

#include <RcppArmadillo.h>
#include <cmath>

void updatePi(const arma::mat& Z,
              const double alpha,
              const int K,
              const int iter,
              arma::mat& pi);

double g_ldet(const arma::mat& M);

double lpdf_z(const arma::mat& M,
              const arma::vec& m,
              const arma::mat& tilde_M,
              const arma::vec& tilde_m,
              const arma::vec& f_obs,
              const arma::vec& f_star,
              const double pi_l,
              const double z_il,
              arma::mat pinv_M,
              arma::mat pinv_tilde_M);

void updateZ(const arma::field<arma::vec>& f_obs,
               const arma::field<arma::vec>& f_star,
               const arma::vec& pi,
               const int iter,
               const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::cube& phi,
               const arma::mat& nu,
               arma::field<arma::mat>& M,
               arma::field<arma::mat>& M_ph,
               arma::field<arma::mat>& pinv_M,
               arma::field<arma::vec>& m,
               arma::field<arma::vec>& m_ph,
               arma::mat& Z_ph,
               arma::field<arma::mat>& tilde_M,
               arma::field<arma::mat>& tilde_M_ph,
               arma::field<arma::mat>& pinv_tilde_M,
               arma::field<arma::vec>& tilde_m,
               arma::field<arma::vec>& tilde_m_ph,
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
             arma::field<arma::mat>& pinv_M,
             arma::field<arma::vec>& m,
             arma::field<arma::vec>& m_ph,
             arma::mat& Z_ph,
             arma::field<arma::mat>& tilde_M,
             arma::field<arma::mat>& tilde_M_ph,
             arma::field<arma::mat>& pinv_tilde_M,
             arma::field<arma::vec>& tilde_m,
             arma::field<arma::vec>& tilde_m_ph,
             arma::field<arma::mat>& mp_inv,
             arma::cube& Z);
#endif
