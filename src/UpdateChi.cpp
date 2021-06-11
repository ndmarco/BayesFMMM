#include <RcppArmadillo.h>

//' Updates the chi parameters
//'
//' @name updateChi
//' @param y_obs Field of vectors containing observed time points
//' @param y_obs Field of matrices containing unobserved time points for all iterations
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param B_star Field of matrices containing basis functions evaluated at unobserved time points
//' @param Phi Cube containing current Phi parameters
//' @param nu Matrix containing current nu parameters
//' @param Z Matrix containing current Z parameters
//' @param sigma double containing current sigma parameter
//' @param iter Int containing MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param chi Cube contianing MCMC samples for chi

void updateChi(const arma::field<arma::vec>& y_obs,
              const arma::field<arma::mat>& y_star,
              const arma::field<arma::mat>& B_obs,
              const arma::field<arma::mat>& B_star,
              const arma::cube& Phi,
              const arma::mat& nu,
              const arma::mat& Z,
              const double& sigma,
              const int& iter,
              const int& tot_mcmc_iters,
              arma::cube& chi){
  double w = 0;
  double W = 0;
  double ph = 0;
  for(int i = 0; i < chi.n_rows; i++){
    for(int m = 0; m < chi.n_cols; m++){
      w = 0;
      W = 0;
      for(int l = 0; l < y_obs(i,0).n_elem; l++){
        ph = 0;
        for(int k2 = 0; k2 < Z.n_cols; k2++){
          ph = ph + Z(i, k2) * arma::dot(Phi.slice(m).row(k2),
                      B_obs(i,0).row(l));
        }
        w = w + ph * y_obs(i,0)(l);
        W = W + ph * ph;
        for(int k1 =0; k1 < Z.n_cols; k1++){
          if(Z(i,k1) != 0){
            w = w - Z(i,k1) * ph * arma::dot(nu.row(k1), B_obs(i,0).row(l));
            for(int n = 0; n < chi.n_cols; n++){
              if(n != m){
                w = w - Z(i,k1) * ph * chi(i, n, iter) * arma::dot(Phi.slice(n).row(k1),
                                 B_obs(i,0).row(l));
              }
            }
          }
        }
      }
      if(y_star(i,0).n_elem > 0){
        for(int l = 0; l < y_star(i,0).n_cols; l++){
          ph = 0;
          for(int k2 = 0; k2 < Z.n_cols; k2++){
            ph = ph + Z(i, k2) * arma::dot(Phi.slice(m).row(k2),
                        B_star(i,0).row(l));
          }
          w = w + ph * y_star(i,0)(iter, l);
          W = W + ph * ph;
          for(int k1 = 0; k1 < Z.n_cols; k1++){
            if(Z(i,k1) != 0){
              w = w - Z(i,k1) * ph * arma::dot(nu.row(k1), B_star(i,0).row(l));
              for(int n = 0; n < chi.n_cols; n++){
                if(n != m){
                  w = w - Z(i,k1) * ph * chi(i, n, iter) * arma::dot(Phi.slice(n).row(k1),
                                   B_star(i,0).row(l));
                }
              }
            }
          }
        }
      }
      w = w / sigma;
      W = 1 + (W / sigma);
      W = 1 / W;
      chi(i, m, iter) = R::rnorm(W*w, std::sqrt(W));
    }
  }
  if(iter < (tot_mcmc_iters - 1)){
    chi.slice(iter + 1) = chi.slice(iter);
  }
}

//' Updates the chi parameters using Tempered Transitions
//'
//' @name updateChiTempered
//' @param beta_i Vector containing the current temperature
//' @param y_obs Field of vectors containing observed time points
//' @param y_obs Field of matrices containing unobserved time points for all iterations
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param B_star Field of matrices containing basis functions evaluated at unobserved time points
//' @param Phi Cube containing current Phi parameters
//' @param nu Matrix containing current nu parameters
//' @param Z Matrix containing current Z parameters
//' @param sigma double containing current sigma parameter
//' @param iter Int containing MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param chi Cube containing MCMC samples for chi

void updateChiTempered(const double& beta_i,
                       const arma::field<arma::vec>& y_obs,
                       const arma::field<arma::mat>& y_star,
                       const arma::field<arma::mat>& B_obs,
                       const arma::field<arma::mat>& B_star,
                       const arma::cube& Phi,
                       const arma::mat& nu,
                       const arma::mat& Z,
                       const double& sigma,
                       const int& iter,
                       const int& tot_mcmc_iters,
                       arma::cube& chi){
  double w = 0;
  double W = 0;
  double ph = 0;
  for(int i = 0; i < chi.n_rows; i++){
    for(int m = 0; m < chi.n_cols; m++){
      w = 0;
      W = 0;
      for(int l = 0; l < y_obs(i,0).n_elem; l++){
        ph = 0;
        for(int k2 = 0; k2 < Z.n_cols; k2++){
          ph = ph + Z(i, k2) * arma::dot(Phi.slice(m).row(k2),
                      B_obs(i,0).row(l));
        }
        w = w + ph * y_obs(i,0)(l);
        W = W + ph * ph;
        for(int k1 =0; k1 < Z.n_cols; k1++){
          if(Z(i,k1) != 0){
            w = w - Z(i,k1) * ph * arma::dot(nu.row(k1), B_obs(i,0).row(l));
            for(int n = 0; n < chi.n_cols; n++){
              if(n != m){
                w = w - Z(i,k1) * ph * chi(i, n, iter) * arma::dot(Phi.slice(n).row(k1),
                                 B_obs(i,0).row(l));
              }
            }
          }
        }
      }
      if(y_star(i,0).n_elem > 0){
        for(int l = 0; l < y_star(i,0).n_cols; l++){
          ph = 0;
          for(int k2 = 0; k2 < Z.n_cols; k2++){
            ph = ph + Z(i, k2) * arma::dot(Phi.slice(m).row(k2),
                        B_star(i,0).row(l));
          }
          w = w + ph * y_star(i,0)(iter, l);
          W = W + ph * ph;
          for(int k1 = 0; k1 < Z.n_cols; k1++){
            if(Z(i,k1) != 0){
              w = w - Z(i,k1) * ph * arma::dot(nu.row(k1), B_star(i,0).row(l));
              for(int n = 0; n < chi.n_cols; n++){
                if(n != m){
                  w = w - Z(i,k1) * ph * chi(i, n, iter) * arma::dot(Phi.slice(n).row(k1),
                                   B_star(i,0).row(l));
                }
              }
            }
          }
        }
      }
      w = (w * beta_i) / sigma;
      W = 1 + ((W * beta_i) / sigma);
      W = 1 / W;
      chi(i, m, iter) = R::rnorm(W*w, std::sqrt(W));
    }
  }
  if(iter < (tot_mcmc_iters - 1)){
    chi.slice(iter + 1) = chi.slice(iter);
  }
}

