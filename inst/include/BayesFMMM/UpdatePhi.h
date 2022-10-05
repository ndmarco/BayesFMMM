#ifndef BayesFMMM_UPDATE_PHI_H
#define BayesFMMM_UPDATE_PHI_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFMMM{
// Updates the Phi parameters
//
// @name UpdatePhi
// @param y_obs Field of Vectors containing observed time points
// @param B_obs Field of Matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param gamma Cube containing current gamma parameters
// @param tilde_tau vector containing current tilde_tau parameters
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhi(const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& B_obs,
                      const arma::mat& nu,
                      const arma::cube& gamma,
                      const arma::mat& tilde_tau,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const double& sigma_sq,
                      const int& iter,
                      const int& tot_mcmc_iters,
                      arma::vec& m_1,
                      arma::mat& M_1,
                      arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();
  double ph = 0;

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          for(int l = 0; l < y_obs(i,0).n_elem; l++){
            ph = 0;
            ph = y_obs(i,0)(l) - Z(i,j) * arma::dot(nu.row(j),
                       B_obs(i,0).row(l));
            M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) *
              B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
            for(int k = 0; k < nu.n_rows; k++){
              for(int n = 0; n < Phi(iter,0).n_slices; n++){
                if(k == j){
                  if(n != m){
                    ph = ph - (Z(i,j) * chi(i,n) *
                      arma::dot(Phi(iter,0).slice(n).row(k), B_obs(i,0).row(l)));
                  }
                }else{
                  ph = ph - (Z(i,k) * chi(i,n) *
                    arma::dot(Phi(iter,0).slice(n).row(k),  B_obs(i,0).row(l)));
                }
              }
              if(k != j){
                ph = ph - Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
              }
            }
            m_1 = m_1 + Z(i,j) * chi(i,m) * B_obs(i,0).row(l).t() * ph;
          }
        }
      }
      m_1 = m_1 * (1 / sigma_sq);
      M_1 = M_1 * (1 / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau(j,m) * gamma.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}

// Updates the Phi parameters using a Tempered Transition
//
// @name UpdatePhiTempered
// @param beta_i Double containing the current temperature
// @param y_obs Field of Vectors containing observed time points
// @param B_obs Field of Matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param gamma Cube containing current gamma parameters
// @param tilde_tau vector containing current tilde_tau parameters
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhiTempered(const double& beta_i,
                              const arma::field<arma::vec>& y_obs,
                              const arma::field<arma::mat>& B_obs,
                              const arma::mat& nu,
                              const arma::cube& gamma,
                              const arma::mat& tilde_tau,
                              const arma::mat& Z,
                              const arma::mat& chi,
                              const double& sigma_sq,
                              const int& iter,
                              const int& tot_mcmc_iters,
                              arma::vec& m_1,
                              arma::mat& M_1,
                              arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();
  double ph = 0;

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          for(int l = 0; l < y_obs(i,0).n_elem; l++){
            ph = 0;
            ph = y_obs(i,0)(l) - Z(i,j) * arma::dot(nu.row(j),
                       B_obs(i,0).row(l));
            M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) * B_obs(i,0).row(l).t() *
              B_obs(i,0).row(l));
            for(int k = 0; k < nu.n_rows; k++){
              for(int n = 0; n < Phi(iter,0).n_slices; n++){
                if(k == j){
                  if(n != m){
                    ph = ph - (Z(i,j) * chi(i,n) *
                      arma::dot(Phi(iter,0).slice(n).row(k), B_obs(i,0).row(l)));
                  }
                }else{
                  ph = ph - (Z(i,k) * chi(i,n) *
                    arma::dot(Phi(iter,0).slice(n).row(k),  B_obs(i,0).row(l)));
                }
              }
              if(k != j){
                ph = ph - Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
              }
            }
            m_1 = m_1 +  Z(i,j) * chi(i,m) * B_obs(i,0).row(l).t() * ph;
          }
        }
      }
      m_1 = m_1 * (beta_i / sigma_sq);
      M_1 = M_1 * (beta_i / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau(j,m) * gamma.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}

// Updates the Phi parameters for the multivariate model
//
// @name UpdatePhiMV
// @param y_obs Matrix containing observed vectors
// @param nu Matrix containing current nu parameters
// @param gamma Cube containing current gamma parameters
// @param tilde_tau vector containing current tilde_tau parameters
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhiMV(const arma::mat& y_obs,
                        const arma::mat& nu,
                        const arma::cube& gamma,
                        const arma::mat& tilde_tau,
                        const arma::mat& Z,
                        const arma::mat& chi,
                        const double& sigma_sq,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        arma::vec& m_1,
                        arma::mat& M_1,
                        arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          arma::vec ph = arma::zeros(y_obs.n_cols);
          ph = y_obs.row(i).t() - Z(i,j) * nu.row(j).t();
          M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) *
            arma::eye(y_obs.n_cols, y_obs.n_cols));
          for(int k = 0; k < nu.n_rows; k++){
            for(int n = 0; n < Phi(iter,0).n_slices; n++){
              if(k == j){
                if(n != m){
                  ph = ph - (Z(i,j) * chi(i,n) * Phi(iter,0).slice(n).row(k).t());
                }
              }else{
                ph = ph - (Z(i,k) * chi(i,n) * Phi(iter,0).slice(n).row(k).t());
              }
            }
            if(k != j){
              ph = ph - Z(i,k) * nu.row(k).t();
            }
          }
          m_1 = m_1 + Z(i,j) * chi(i,m) * ph;
        }
      }
      m_1 = m_1 * (1 / sigma_sq);
      M_1 = M_1 * (1 / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau(j,m) * gamma.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}

// Updates the Phi parameters for the tempered multivariate model
//
// @name UpdatePhiMVTempered
// @param beta_i Double containing the current temperature
// @param y_obs Matrix containing observed vectors
// @param nu Matrix containing current nu parameters
// @param gamma Cube containing current gamma parameters
// @param tilde_tau vector containing current tilde_tau parameters
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhiTemperedMV(const double& beta_i,
                                const arma::mat& y_obs,
                                const arma::mat& nu,
                                const arma::cube& gamma,
                                const arma::mat& tilde_tau,
                                const arma::mat& Z,
                                const arma::mat& chi,
                                const double& sigma_sq,
                                const int& iter,
                                const int& tot_mcmc_iters,
                                arma::vec& m_1,
                                arma::mat& M_1,
                                arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          arma::vec ph = arma::zeros(y_obs.n_cols);
          ph = y_obs.row(i).t() - Z(i,j) * nu.row(j).t();
          M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) *
            arma::eye(y_obs.n_cols, y_obs.n_cols));
          for(int k = 0; k < nu.n_rows; k++){
            for(int n = 0; n < Phi(iter,0).n_slices; n++){
              if(k == j){
                if(n != m){
                  ph = ph - (Z(i,j) * chi(i,n) * Phi(iter,0).slice(n).row(k).t());
                }
              }else{
                ph = ph - (Z(i,k) * chi(i,n) * Phi(iter,0).slice(n).row(k).t());
              }
            }
            if(k != j){
              ph = ph - Z(i,k) * nu.row(k).t();
            }
          }
          m_1 = m_1 + Z(i,j) * chi(i,m) * ph;
        }
      }
      m_1 = m_1 * (beta_i / sigma_sq);
      M_1 = M_1 * (beta_i / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau(j,m) * gamma.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}


/////////////////////////////
// Covariate Adjusted Code //
/////////////////////////////

// Updates the Phi parameters for a covariate adjusted model
//
// @name UpdatePhiCovariateAdj
// @param y_obs Field of Vectors containing observed time points
// @param B_obs Field of Matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param gamma Cube containing current gamma parameters
// @param tilde_tau_phi vector containing current tilde_tau_phi parameters
// @param xi Field of Cubes containing
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param X Matrix containing covariates
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhiCovariateAdj(const arma::field<arma::vec>& y_obs,
                                  const arma::field<arma::mat>& B_obs,
                                  const arma::mat& nu,
                                  const arma::cube& eta,
                                  const arma::cube& gamma,
                                  const arma::mat& tilde_tau_phi,
                                  const arma::field<arma::cube>& xi,
                                  const arma::mat& Z,
                                  const arma::mat& chi,
                                  const double& sigma_sq,
                                  const arma::mat& X,
                                  const int& iter,
                                  const int& tot_mcmc_iters,
                                  arma::vec& m_1,
                                  arma::mat& M_1,
                                  arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();
  double ph = 0;

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          for(int l = 0; l < y_obs(i,0).n_elem; l++){
            ph = 0;
            ph = y_obs(i,0)(l) - Z(i,j) * (arma::dot(nu.row(j),
                       B_obs(i,0).row(l)) + arma::dot(eta.slice(j) * X.row(i).t(),
                       B_obs(i,0).row(l)));
            M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) *
              B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
            for(int k = 0; k < nu.n_rows; k++){
              for(int n = 0; n < Phi(iter,0).n_slices; n++){
                if(k == j){
                  if(n != m){
                    ph = ph - (Z(i,j) * chi(i,n) *
                      arma::dot(Phi(iter,0).slice(n).row(k), B_obs(i,0).row(l)));
                  }
                  ph = ph - (Z(i,j) * chi(i,n) *
                    arma::dot(xi(iter,k).slice(n) * X.row(i).t(),B_obs(i,0).row(l)));
                }else{
                  ph = ph - (Z(i,k) * chi(i,n) *
                    arma::dot((Phi(iter,0).slice(n).row(k) +
                    (xi(iter,k).slice(n) * X.row(i).t()).t()),  B_obs(i,0).row(l)));
                }
              }
              if(k != j){
                ph = ph - Z(i,k) * (arma::dot(nu.row(k), B_obs(i,0).row(l)) +
                  arma::dot(eta.slice(k) * X.row(i).t(), B_obs(i,0).row(l)));
              }
            }
            m_1 = m_1 + Z(i,j) * chi(i,m) * B_obs(i,0).row(l).t() * ph;
          }
        }
      }
      m_1 = m_1 * (1 / sigma_sq);
      M_1 = M_1 * (1 / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau_phi(j,m) * gamma.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}

// Updates the Phi parameters using a Tempered Transition for a covariate adjusted model
//
// @name UpdatePhiTemperedCovariateAdj
// @param beta_i Double containing the current temperature
// @param y_obs Field of Vectors containing observed time points
// @param B_obs Field of Matrices containing basis functions evaluated at observed time points
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param gamma_phi Cube containing current gamma_phi parameters
// @param tilde_tau_phi vector containing current tilde_tau_phi parameters
// @param xi Field of Cubes containing
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param X Matrix containing covariates
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhiTemperedCovariateAdj(const double& beta_i,
                                          const arma::field<arma::vec>& y_obs,
                                          const arma::field<arma::mat>& B_obs,
                                          const arma::mat& nu,
                                          const arma::cube& eta,
                                          const arma::cube& gamma_phi,
                                          const arma::mat& tilde_tau_phi,
                                          const arma::field<arma::cube>& xi,
                                          const arma::mat& Z,
                                          const arma::mat& chi,
                                          const double& sigma_sq,
                                          const arma::mat& X,
                                          const int& iter,
                                          const int& tot_mcmc_iters,
                                          arma::vec& m_1,
                                          arma::mat& M_1,
                                          arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();
  double ph = 0;

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          for(int l = 0; l < y_obs(i,0).n_elem; l++){
            ph = 0;
            ph = y_obs(i,0)(l) - Z(i,j) * (arma::dot(nu.row(j),
                            B_obs(i,0).row(l)) + arma::dot(eta.slice(j) * X.row(i).t(),
                            B_obs(i,0).row(l)));
            M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) *
              B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
            for(int k = 0; k < nu.n_rows; k++){
              for(int n = 0; n < Phi(iter,0).n_slices; n++){
                if(k == j){
                  if(n != m){
                    ph = ph - (Z(i,j) * chi(i,n) *
                      arma::dot(Phi(iter,0).slice(n).row(k), B_obs(i,0).row(l)));
                  }
                  ph = ph - (Z(i,j) * chi(i,n) *
                    arma::dot(xi(iter,k).slice(n) * X.row(i).t(),B_obs(i,0).row(l)));
                }else{
                  ph = ph - (Z(i,k) * chi(i,n) *
                    arma::dot((Phi(iter,0).slice(n).row(k) +
                    (xi(iter,k).slice(n) * X.row(i).t()).t()),  B_obs(i,0).row(l)));
                }
              }
              if(k != j){
                ph = ph - Z(i,k) * (arma::dot(nu.row(k), B_obs(i,0).row(l)) +
                  arma::dot(eta.slice(k) * X.row(i).t(), B_obs(i,0).row(l)));
              }
            }
            m_1 = m_1 + Z(i,j) * chi(i,m) * B_obs(i,0).row(l).t() * ph;
          }
        }
      }
      m_1 = m_1 * (beta_i / sigma_sq);
      M_1 = M_1 * (beta_i / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau_phi(j,m) * gamma_phi.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}

// Updates the Phi parameters for the covariate adjusted multivariate model
//
// @name UpdatePhiMVCovariateAdj
// @param y_obs Matrix containing observed vectors
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param gamma_phi Cube containing current gamma_phi parameters
// @param tilde_tau_phi vector containing current tilde_tau_phi parameters
// @param xi Field of cubes containing all of the xi parameters
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param X matrix containing the covariates
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhiMVCovariateAdj(const arma::mat& y_obs,
                                    const arma::mat& nu,
                                    const arma::cube& eta,
                                    const arma::cube& gamma_phi,
                                    const arma::mat& tilde_tau_phi,
                                    const arma::field<arma::cube>& xi,
                                    const arma::mat& Z,
                                    const arma::mat& chi,
                                    const double& sigma_sq,
                                    const arma::mat& X,
                                    const int& iter,
                                    const int& tot_mcmc_iters,
                                    arma::vec& m_1,
                                    arma::mat& M_1,
                                    arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          arma::vec ph = arma::zeros(y_obs.n_cols);
          ph = y_obs.row(i).t() - Z(i,j) * (nu.row(j).t() +
            (eta.slice(j) * X.row(i).t()));
          M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) *
            arma::eye(y_obs.n_cols, y_obs.n_cols));
          for(int k = 0; k < nu.n_rows; k++){
            for(int n = 0; n < Phi(iter,0).n_slices; n++){
              if(k == j){
                if(n != m){
                  ph = ph - (Z(i,j) * chi(i,n) * Phi(iter,0).slice(n).row(k).t());
                }
                ph = ph - (Z(i,j) * chi(i,n) * (xi(iter,k).slice(n) * X.row(i).t()));
              }else{
                ph = ph - (Z(i,k) * chi(i,n) * (Phi(iter,0).slice(n).row(k).t() +
                  (xi(iter,k).slice(n) * X.row(i).t())));
              }
            }
            if(k != j){
              ph = ph - Z(i,k) * (nu.row(k).t() +  (eta.slice(k) * X.row(i).t()));
            }
          }
          m_1 = m_1 + Z(i,j) * chi(i,m) * ph;
        }
      }
      m_1 = m_1 * (1 / sigma_sq);
      M_1 = M_1 * (1 / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau_phi(j,m) * gamma_phi.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}

// Updates the Phi parameters for the tempered multivariate covariate adjusted model
//
// @name UpdatePhiMVTemperedCovariateAd
// @param beta_i Double containing the current temperature
// @param y_obs Matrix containing observed vectors
// @param nu Matrix containing current nu parameters
// @param eta Cube containing current eta parameters
// @param gamma_phi Cube containing current gamma_phi parameters
// @param tilde_tau_phi vector containing current tilde_tau_phi parameters
// @param xi Field of cubes containing all of the xi parameters
// @param Z Matrix containing current Z parameters
// @param sigma_sq double containing the sigma_sq variable
// @param X matrix containing the covariates
// @param chi Matrix containing chi values
// @param iter int containing current mcmc sample
// @param m_1 Vector acting as a placeholder for m in mean vector
// @param M_1 Matrix acting as a placeholder for M in covariance
// @param Phi Field of Cubes containing all mcmc samples of Phi
inline void updatePhiTemperedMVCovariateAdj(const double& beta_i,
                                            const arma::mat& y_obs,
                                            const arma::mat& nu,
                                            const arma::cube& eta,
                                            const arma::cube& gamma_phi,
                                            const arma::mat& tilde_tau_phi,
                                            const arma::field<arma::cube>& xi,
                                            const arma::mat& Z,
                                            const arma::mat& chi,
                                            const double& sigma_sq,
                                            const arma::mat& X,
                                            const int& iter,
                                            const int& tot_mcmc_iters,
                                            arma::vec& m_1,
                                            arma::mat& M_1,
                                            arma::field<arma::cube>& Phi){
  m_1.zeros();
  M_1.zeros();

  for(int j =  0; j < Phi(iter,0).n_rows; j ++){
    for(int m = 0; m < Phi(iter,0).n_slices; m++){
      m_1.zeros();
      M_1.zeros();
      for(int i = 0; i < Z.n_rows; i++){
        if(Z(i,j) != 0){
          arma::vec ph = arma::zeros(y_obs.n_cols);
          ph = y_obs.row(i).t() - Z(i,j) * (nu.row(j).t() +
            (eta.slice(j) * X.row(i).t()));
          M_1 = M_1 + Z(i,j) *  Z(i,j) * (chi(i,m) * chi(i,m) *
            arma::eye(y_obs.n_cols, y_obs.n_cols));
          for(int k = 0; k < nu.n_rows; k++){
            for(int n = 0; n < Phi(iter,0).n_slices; n++){
              if(k == j){
                if(n != m){
                  ph = ph - (Z(i,j) * chi(i,n) * Phi(iter,0).slice(n).row(k).t());
                }
                ph = ph - (Z(i,j) * chi(i,n) * (xi(iter,k).slice(n) * X.row(i).t()));
              }else{
                ph = ph - (Z(i,k) * chi(i,n) * (Phi(iter,0).slice(n).row(k).t() +
                  (xi(iter,k).slice(n) * X.row(i).t())));
              }
            }
            if(k != j){
              ph = ph - Z(i,k) * (nu.row(k).t() +  (eta.slice(k) * X.row(i).t()));
            }
          }
          m_1 = m_1 + Z(i,j) * chi(i,m) * ph;
        }
      }
      m_1 = m_1 * (beta_i / sigma_sq);
      M_1 = M_1 * (beta_i / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau_phi(j,m) * gamma_phi.slice(m)(j,k);
      }
      arma::inv(M_1, M_1);

      //generate new sample
      Phi(iter,0).slice(m).row(j) =  arma::mvnrnd(M_1 * m_1, M_1).t();
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Phi(iter + 1,0) = Phi(iter,0);
  }
}

}
#endif
