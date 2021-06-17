#include <RcppArmadillo.h>
#include <cmath>
#include "Distributions.H"

//' Updates the Phi parameters
//'
//' @name UpdatePhi
//' @param y_obs Field of Vectors containing observed time points
//' @param y_star Field of Matrices contianing unobserved time points at all mcmc iterations
//' @param B_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param B_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param nu Matrix containing current nu parameters
//' @param gamma Cube containing current gamma parameters
//' @param tilde_tau vector containing current tilde_tau parameters
//' @param Z Matrix containing current Z parameters
//' @param sigma_sq double containing the sigma_sq variable
//' @param chi Matrix containing chi values
//' @param iter int containing current mcmc sample
//' @param m_1 Vector acting as a placeholder for m in mean vector
//' @param M_1 Matrix acting as a placeholder for M in covariance
//' @param Phi Field of Cubes containing all mcmc samples of Phi

void updatePhi(const arma::field<arma::vec>& y_obs,
               const arma::field<arma::mat>& y_star,
               const arma::field<arma::mat>& B_obs,
               const arma::field<arma::mat>& B_star,
               const arma::mat& nu,
               const arma::cube& gamma,
               const arma::vec& tilde_tau,
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


          // Check to see if there are unobserved time points of interest
          if(B_star(i,0).n_elem > 0){
            for(int l = 0; l < y_star(i,0).n_cols; l++){
              ph = 0;
              ph = y_star(i,0)(iter,l) - Z(i,j) * arma::dot(nu.row(j),
                             B_star(i,0).row(l));
              M_1 = M_1 + Z(i,j) * Z(i,j) * (chi(i,m) * chi(i,m) *
                B_star(i,0).row(l).t() * B_star(i,0).row(l));
              for(int k = 0; k < nu.n_rows; k++){
                for(int n = 0; n < Phi(iter,0).n_slices; n++){
                  if(k == j){
                    if(n != m){
                      ph = ph - (Z(i,j) * chi(i,n) *
                        arma::dot(Phi(iter,0).slice(n).row(k),
                                  B_star(i,0).row(l)));
                    }
                  }else{
                    ph = ph - (Z(i,k) * chi(i,n) *
                      arma::dot(Phi(iter,0).slice(n).row(k),
                                B_star(i,0).row(l)));
                  }
                }
                if(k != j){
                  ph = ph - Z(i,k) * arma::dot(nu.row(k), B_star(i,0).row(l));
                }
              }
              m_1 = m_1 + Z(i,j) * chi(i,m) * B_star(i,0).row(l).t() * ph;
            }
          }
        }
      }
      m_1 = m_1 * (1 / sigma_sq);
      M_1 = M_1 * (1 / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau(m) * gamma.slice(m)(j,k);
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

//' Updates the Phi parameters using a Tempered Transition
//'
//' @name UpdatePhiTempered
//' @param beta_i Double containing the current temperature
//' @param y_obs Field of Vectors containing observed time points
//' @param y_star Field of Matrices contianing unobserved time points at all mcmc iterations
//' @param B_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param B_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param nu Matrix containing current nu parameters
//' @param gamma Cube containing current gamma parameters
//' @param tilde_tau vector containing current tilde_tau parameters
//' @param Z Matrix containing current Z parameters
//' @param sigma_sq double containing the sigma_sq variable
//' @param chi Matrix containing chi values
//' @param iter int containing current mcmc sample
//' @param m_1 Vector acting as a placeholder for m in mean vector
//' @param M_1 Matrix acting as a placeholder for M in covariance
//' @param Phi Field of Cubes containing all mcmc samples of Phi

void updatePhiTempered(const double& beta_i,
                       const arma::field<arma::vec>& y_obs,
                       const arma::field<arma::mat>& y_star,
                       const arma::field<arma::mat>& B_obs,
                       const arma::field<arma::mat>& B_star,
                       const arma::mat& nu,
                       const arma::cube& gamma,
                       const arma::vec& tilde_tau,
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
            M_1 = M_1 + Z(i,j) * (chi(i,m) * chi(i,m) * B_obs(i,0).row(l).t() *
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


          // Check to see if there are unobserved time points of interest
          if(B_star(i,0).n_elem > 0){
            for(int l = 0; l < y_star(i,0).n_cols; l++){
              ph = 0;
              ph = y_star(i,0)(iter,l) - Z(i,j) * arma::dot(nu.row(j),
                          B_star(i,0).row(l));
              M_1 = M_1 +  Z(i,j) * (chi(i,m) * chi(i,m) * B_star(i,0).row(l).t() *
                B_star(i,0).row(l));
              for(int k = 0; k < nu.n_rows; k++){
                for(int n = 0; n < Phi(iter,0).n_slices; n++){
                  if(k == j){
                    if(n != m){
                      ph = ph - (Z(i,j) * chi(i,n) *
                        arma::dot(Phi(iter,0).slice(n).row(k),
                                  B_star(i,0).row(l)));
                    }
                  }else{
                    ph = ph - (Z(i,k) * chi(i,n) *
                      arma::dot(Phi(iter,0).slice(n).row(k),
                                B_star(i,0).row(l)));
                  }
                }
                if(k != j){
                  ph = ph - Z(i,k) * arma::dot(nu.row(k), B_star(i,0).row(l));
                }
              }
              m_1 = m_1 + Z(i,j) * chi(i,m) * B_star(i,0).row(l).t() * ph;
            }
          }
        }
      }
      m_1 = m_1 * (beta_i / sigma_sq);
      M_1 = M_1 * (beta_i / sigma_sq);

      //Add on diagonal component
      for(int k = 0; k < M_1.n_rows; k++){
        M_1(k,k) = M_1(k,k) + tilde_tau(m) * gamma.slice(m)(j,k);
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
