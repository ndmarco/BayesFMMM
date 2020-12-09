#ifndef COMPUTEMM_H
#define COMPUTEMM_H

#include <RcppArmadillo.h>
#include <cmath>

void computeMi(const arma::field<arma::mat>& S_obs,
               const arma::mat& Z,
               const arma::cube& phi,
               const int i,
               arma::mat& M);

void computeMi(const arma::field<arma::mat>& S_obs,
               const arma::mat& Z,
               const arma::mat& phi,
               const int i,
               arma::mat& M);

void computeM(const arma::field<arma::mat>& S_obs,
              const arma::mat& Z,
              const arma::cube& phi,
              arma::field<arma::mat>& M);

void computeM(const arma::field<arma::mat>& S_obs,
              const arma::mat& Z,
              const arma::mat& phi,
              arma::field<arma::mat>& M);

void compute_mi(const arma::field<arma::mat>& S_obs,
                const arma::mat& Z,
                const arma::cube& phi,
                const arma::mat& nu,
                const int i,
                arma::vec& m);

void compute_mi(const arma::field<arma::mat>& S_obs,
                const arma::mat& Z,
                const arma::mat& phi,
                const arma::mat& nu,
                const int i,
                arma::vec& m);

void compute_m(const arma::field<arma::mat>& S_obs,
               const arma::mat& Z,
               const arma::cube& phi,
               const arma::mat& nu,
               arma::field<arma::vec>& m);

void compute_m(const arma::field<arma::mat>& S_obs,
               const arma::mat& Z,
               const arma::mat& phi,
               const arma::mat& nu,
               arma::field<arma::vec>& m);

void compute_tildeMi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::mat& Z,
                    const arma::cube& phi,
                    const int i,
                    arma::mat& Z_plus,
                    arma::mat& M);

void compute_tildeMi(const arma::field<arma::mat>& S_star,
                     const arma::field<arma::mat>& S_obs,
                     const arma::mat& Z,
                     const arma::mat& phi,
                     const int i,
                     arma::mat& Z_plus,
                     arma::mat& M);

void compute_tildeM(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs,
                   const arma::mat& Z,
                   const arma::cube& phi,
                   arma::field<arma::mat>& Z_plus,
                   arma::field<arma::mat>& M);

void compute_tildeM(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::mat& Z,
                    const arma::mat& phi,
                    arma::field<arma::mat>& Z_plus,
                    arma::field<arma::mat>& M);

void compute_tildemi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::vec f_obs,
                    const arma::mat& Z,
                    const arma::cube& phi,
                    const arma::mat& nu,
                    const int i,
                    arma::mat& Z_plus,
                    arma::mat& A_plus,
                    arma::mat& C,
                    arma::vec& tilde_m);

void compute_tildemi(const arma::field<arma::mat>& S_star,
                     const arma::field<arma::mat>& S_obs,
                     const arma::vec f_obs,
                     const arma::mat& Z,
                     const arma::mat& phi,
                     const arma::mat& nu,
                     const int i,
                     arma::mat& Z_plus,
                     arma::mat& A_plus,
                     arma::mat& C,
                     arma::vec& tilde_m);

void compute_tildem(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs,
                   const arma::field<arma::vec>& f_obs,
                   const arma::mat& Z,
                   const arma::cube& phi,
                   const arma::mat& nu,
                   arma::field<arma::mat>& Z_plus,
                   arma::field<arma::mat>& A_plus,
                   arma::field<arma::mat>& C,
                   arma::field<arma::vec>& tilde_m);

void compute_tildem(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::field<arma::vec>& f_obs,
                    const arma::mat& Z,
                    const arma::mat& phi,
                    const arma::mat& nu,
                    arma::field<arma::mat>& Z_plus,
                    arma::field<arma::mat>& A_plus,
                    arma::field<arma::mat>& C,
                    arma::field<arma::vec>& tilde_m);

void compute_tildeMi_tildemi(const arma::field<arma::mat>& S_star,
                             const arma::field<arma::mat>& S_obs,
                             const arma::field<arma::vec>& f_obs,
                             const arma::mat& Z,
                             const arma::cube& phi,
                             const arma::mat& nu,
                             const int i,
                             arma::field<arma::mat>& mp_inv,
                             arma::field<arma::mat>& tilde_M,
                             arma::field<arma::vec>& tilde_m);
void compute_tildeMi_tildemi(const arma::field<arma::mat>& S_star,
                             const arma::field<arma::mat>& S_obs,
                             const arma::field<arma::vec>& f_obs,
                             const arma::mat& Z,
                             const arma::mat& phi,
                             const arma::mat& nu,
                             const int i,
                             arma::field<arma::mat>& mp_inv,
                             arma::field<arma::mat>& tilde_M,
                             arma::field<arma::vec>& tilde_m);
void compute_tildeM_tildem(const arma::field<arma::mat>& S_star,
                           const arma::field<arma::mat>& S_obs,
                           const arma::field<arma::vec>& f_obs,
                           const arma::mat& Z,
                           const arma::cube& phi,
                           const arma::mat& nu,
                           arma::field<arma::mat>& mp_inv,
                           arma::field<arma::mat>& tilde_M,
                           arma::field<arma::vec>& tilde_m);

void compute_tildeM_tildem(const arma::field<arma::mat>& S_star,
                           const arma::field<arma::mat>& S_obs,
                           const arma::field<arma::vec>& f_obs,
                           const arma::mat& Z,
                           const arma::mat& phi,
                           const arma::mat& nu,
                           arma::field<arma::mat>& mp_inv,
                           arma::field<arma::mat>& tilde_M,
                           arma::field<arma::vec>& tilde_m);
#endif
