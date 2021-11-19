#include <RcppArmadillo.h>
#include <cmath>

//' Calculates the log acceptance probability at a specific temperature
//'
//' @name calculatePZeta
//' @param beta_i Double containing the current temperature
//' @param y_obs Field of vectors containing observed time points
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param nu Matrix containing current nu parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param iter Int containing current temperature step
//' @param sigma double containing sigma current parameter
//' @returns logAcceptance Double containing the tempered likelihood pdf

double calculatePZeta(const double& beta_i,
                      const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& B_obs,
                      const arma::mat nu,
                      const arma::cube& Phi,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const int& iter,
                      const double& sigma){
  double logAcceptance = 0;
  double mean = 0;

  for(int i = 0; i < chi.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      mean = 0;
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          mean = mean + Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            mean = mean + Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k),
                              B_obs(i,0).row(l));
          }
        }
      }
      logAcceptance = logAcceptance + ((-(beta_i/2) * std::log(sigma)) -
        (beta_i / (2 * sigma)) * std::pow(y_obs(i,0)(l) - mean, 2.0));
    }
  }
  return logAcceptance;
}

//' Calculates the log acceptance probability of accepting the tempered transitions
//'
//' @name CalculateTTAcceptance
//' @param beta Vector containing the temperature ladder
//' @param y_obs Field of vectors containing observed time points
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param nu Cube containing nu parameters for all tempered transitions steps
//' @param Phi Field of Cubes containing Phi parameters for all tempered transitions steps
//' @param Z Cube containing Z parameters for all tempered transitions steps
//' @param chi Cube containing chi parameters for all tempered transitions steps
//' @param sigma Vector containing sigma parameters for all tempered transition steps
//' @returns log pdf of acceptance probability

double CalculateTTAcceptance(const arma::vec& beta,
                             const arma::field<arma::vec>& y_obs,
                             const arma::field<arma::mat>& B_obs,
                             const arma::cube& nu,
                             const arma::field<arma::cube>& Phi,
                             const arma::cube& Z,
                             const arma::cube& chi,
                             const arma::vec& sigma){
  double logAcceptance = 0;
  int m = sigma.n_elem - 1;
  for(int i = 0; i < (beta.n_elem - 1); i++){
    // calculate for heating up
    logAcceptance = logAcceptance + calculatePZeta(beta(i+1), y_obs,
                                                   B_obs, nu.slice(i),
                                                   Phi(i,0), Z.slice(i),
                                                   chi.slice(i), i, sigma(i));
    logAcceptance = logAcceptance - calculatePZeta(beta(i), y_obs,
                                                 B_obs, nu.slice(i),
                                                 Phi(i,0), Z.slice(i),
                                                 chi.slice(i), i, sigma(i));

    // calculate for cooling down
    logAcceptance = logAcceptance - calculatePZeta(beta(i+1), y_obs,
                                                   B_obs, nu.slice(m-i),
                                                   Phi(m-i,0), Z.slice(m-i),
                                                   chi.slice(m-i), m-i, sigma(m-i));
    logAcceptance = logAcceptance + calculatePZeta(beta(i), y_obs,
                                                   B_obs, nu.slice(m-i),
                                                   Phi(m-i,0), Z.slice(m-i),
                                                   chi.slice(m-i), m-i, sigma(m-i));
  }
  return logAcceptance;
}

//' Calculates the log acceptaMV
//' @param beta_i Double containing the current temperature
//' @param y_obs Matrix containing observed vectors
//' @param nu Matrix containing current nu parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param iter Int containing current temperature step
//' @param sigma double containing sigma current parameter
//' @returns logAcceptance Double containing the tempered likelihood pdf

double calculatePZetaMV(const double& beta_i,
                        const arma::mat& y_obs,
                        const arma::mat nu,
                        const arma::cube& Phi,
                        const arma::mat& Z,
                        const arma::mat& chi,
                        const int& iter,
                        const double& sigma){
  double logAcceptance = 0;
  arma::vec mean = arma::zeros(y_obs.n_cols);

  for(int i = 0; i < chi.n_rows; i++){
    mean = arma::zeros(y_obs.n_cols);
    for(int k = 0; k < Z.n_cols; k++){
      if(Z(i,k) != 0){
        mean = mean + Z(i,k) * nu.row(k);
        for(int n = 0; n < Phi.n_slices; n++){
          mean = mean + Z(i,k) * chi(i,n) * Phi.slice(n).row(k);
        }
      }
    }
    logAcceptance = logAcceptance + ((-(beta_i/2) * std::log(sigma) * y_obs.n_cols) -
      (beta_i / (2 * sigma)) * arma::dot(y_obs.row(i).t() - mean, y_obs.row(i).t() - mean));
  }
  return logAcceptance;
}

//' Calculates the log acceptance probability of accepting the tempered transitions
//'
//' @name CalculateTTAcceptanceMV
//' @param beta Vector containing the temperature ladder
//' @param y_obs Matrix containing observed vectors
//' @param nu Cube containing nu parameters for all tempered transitions steps
//' @param Phi Field of Cubes containing Phi parameters for all tempered transitions steps
//' @param Z Cube containing Z parameters for all tempered transitions steps
//' @param chi Cube containing chi parameters for all tempered transitions steps
//' @param sigma Vector containing sigma parameters for all tempered transition steps
//' @returns log pdf of acceptance probability

double CalculateTTAcceptanceMV(const arma::vec& beta,
                               const arma::mat& y_obs,
                               const arma::cube& nu,
                               const arma::field<arma::cube>& Phi,
                               const arma::cube& Z,
                               const arma::cube& chi,
                               const arma::vec& sigma){
  double logAcceptance = 0;
  int m = sigma.n_elem - 1;
  for(int i = 0; i < (beta.n_elem - 1); i++){
    // calculate for heating up
    logAcceptance = logAcceptance + calculatePZetaMV(beta(i+1), y_obs,
                                                     nu.slice(i),
                                                     Phi(i,0), Z.slice(i),
                                                     chi.slice(i), i, sigma(i));
    logAcceptance = logAcceptance - calculatePZetaMV(beta(i), y_obs,
                                                     nu.slice(i),
                                                     Phi(i,0), Z.slice(i),
                                                     chi.slice(i), i, sigma(i));

    // calculate for cooling down
    logAcceptance = logAcceptance - calculatePZetaMV(beta(i+1), y_obs,
                                                     nu.slice(m-i),
                                                     Phi(m-i,0), Z.slice(m-i),
                                                     chi.slice(m-i), m-i, sigma(m-i));
    logAcceptance = logAcceptance + calculatePZetaMV(beta(i), y_obs,
                                                     nu.slice(m-i),
                                                     Phi(m-i,0), Z.slice(m-i),
                                                     chi.slice(m-i), m-i, sigma(m-i));
  }
  return logAcceptance;
}
