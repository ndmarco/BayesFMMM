#ifndef COMPUTEMM_H
#define COMPUTEMM_H

#include <RcppArmadillo.h>

void computeMi(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
               const arma::cube& phi, const int i, arma::mat& M);
void computeM(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
              const arma::cube& phi, arma::cube& M);
void compute_mi(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
                const arma::cube& phi, const arma::mat& nu, const int i,
                arma::vec& m);
void compute_m(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
               const arma::cube& phi, const arma::mat& nu, arma::mat& m);
void compute_tileMi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs, const arma::mat& Z,
                    const arma::cube& phi, const int i, arma::mat& M);
void compute_tileM(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs, const arma::mat& Z,
                   const arma::cube& phi, arma::field<arma::mat>& Z_plus,
                   arma::field<arma::mat>& M);
void compute_tilemi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs, const  arma::vec f_obs,
                    const arma::mat& Z, const arma::cube& phi, const arma::mat nu,
                    const int i, arma::mat& Z_plus, arma::mat& A_plus,
                    arma::mat& C,arma::vec& tilde_m);
void compute_tilem(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs, const  arma::vec f_obs,
                   const arma::mat& Z, const arma::cube& phi, const arma::mat nu,
                   arma::mat& Z_plus, arma::mat& A_plus,
                   arma::mat& C,arma::mat& tilde_m);
#endif
