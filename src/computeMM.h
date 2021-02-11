#ifndef COMPUTEMM_H
#define COMPUTEMM_H

#include <RcppArmadillo.h>
#include <cmath>
#include "CalculateCov.H"

void compute_Mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::mat& Z,
                const arma::cube& Phi,
                const arma::mat& Rho,
                const int i,
                arma::mat& Cov,
                arma::mat& mp_inv,
                arma::mat& M);

void compute_Mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::mat& Z,
                const arma::mat& phi,
                const int i,
                arma::mat& mp_inv,
                arma::mat& M);

void compute_M(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::mat& Z,
               const arma::cube& Phi,
               const arma::mat& Rho,
               arma::mat& Cov,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::mat>& M);

void compute_M(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::mat& Z,
               const arma::mat& phi,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::mat>& M);

void compute_mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::field<arma::vec>& f_obs,
                const arma::mat& Z,
                const arma::cube& Phi,
                const arma::mat& Rho,
                const arma::mat& nu,
                const int i,
                arma::mat& Cov,
                arma::mat& mp_inv,
                arma::vec& mean_ph_obs,
                arma::vec& mean_ph_star,
                arma::vec& m);

void compute_mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::field<arma::vec>& f_obs,
                const arma::mat& Z,
                const arma::mat& phi,
                const arma::mat& nu,
                const int i,
                arma::mat& mp_inv,
                arma::vec& mean_ph_obs,
                arma::vec& mean_ph_star,
                arma::vec& m);

void compute_m(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::field<arma::vec>& f_obs,
               const arma::mat& Z,
               const arma::cube& Phi,
               const arma::mat& Rho,
               const arma::mat& nu,
               arma::mat& Cov,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::vec>& mean_ph_obs,
               arma::field<arma::vec>& mean_ph_star,
               arma::field<arma::vec>& m);

void compute_m(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::field<arma::vec>& f_obs,
               const arma::mat& Z,
               const arma::mat& phi,
               const arma::mat& nu,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::vec>& mean_ph_obs,
               arma::field<arma::vec>& mean_ph_star,
               arma::field<arma::vec>& m);

void compute_M_m(const arma::field<arma::mat>& S_obs,
                 const arma::field<arma::mat>& S_star,
                 const arma::field<arma::vec>& f_obs,
                 const arma::mat& Z,
                 const arma::cube& Phi,
                 const arma::mat& Rho,
                 const arma::mat& nu,
                 arma::mat& Cov,
                 arma::field<arma::mat>& mp_inv,
                 arma::field<arma::vec>& mean_ph_obs,
                 arma::field<arma::vec>& mean_ph_star,
                 arma::field<arma::vec>& m,
                 arma::field<arma::mat>& M);

void compute_M_m(const arma::field<arma::mat>& S_obs,
                 const arma::field<arma::mat>& S_star,
                 const arma::field<arma::vec>& f_obs,
                 const arma::mat& Z,
                 const arma::mat& phi,
                 const arma::mat& nu,
                 arma::field<arma::mat>& mp_inv,
                 arma::field<arma::vec>& mean_ph_obs,
                 arma::field<arma::vec>& mean_ph_star,
                 arma::field<arma::vec>& m,
                 arma::field<arma::mat>& M);

void compute_mi_Mi(const arma::field<arma::mat>& S_obs,
                   const arma::field<arma::mat>& S_star,
                   const arma::field<arma::vec>& f_obs,
                   const arma::mat& Z,
                   const arma::cube& Phi,
                   const arma::mat& Rho,
                   const arma::mat& nu,
                   const int i,
                   arma::mat& Cov,
                   arma::mat& mp_inv,
                   arma::vec& mean_ph_obs,
                   arma::vec& mean_ph_star,
                   arma::vec& m,
                   arma::mat& M);

void compute_mi_Mi(const arma::field<arma::mat>& S_obs,
                   const arma::field<arma::mat>& S_star,
                   const arma::field<arma::vec>& f_obs,
                   const arma::mat& Z,
                   const arma::mat& phi,
                   const arma::mat& nu,
                   const int i,
                   arma::mat& mp_inv,
                   arma::vec& mean_ph_obs,
                   arma::vec& mean_ph_star,
                   arma::vec& m,
                   arma::mat& M);

#endif
