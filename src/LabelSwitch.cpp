#include <RcppArmadillo.h>

double GetDistanceZ(const arma::mat& Z,
                    const arma::mat& Z_ref){
  double distance = arma::accu(arma::abs(Z - Z_ref));
  return distance;
}

//' Corrects for possible label switching
//'
//' @name LabelSwitch
//' @export
// [[Rcpp::export]]
arma::mat LabelSwitch(arma::mat& Z_ref,
                      arma::mat& Z,
                      const arma::mat& perm_mat){

  double dist_current = 0;
  double dist_min = arma::datum::inf;

  // Create placeholders
  arma::mat Z_ph(Z.n_rows, Z.n_cols, arma::fill::zeros);
  arma::mat Z_min(Z.n_rows, Z.n_cols, arma::fill::zeros);
  // arma::mat nu_min(nu.n_rows, nu.n_cols, arma::fill::zeros);
  // arma::vec pi_min(pi.n_rows, arma::fill::zeros);
  // arma::cube gamma_min(gamma(0,0).n_rows, gamma(0,0).n_cols, gamma(0,0).n_slices, arma::fill::zeros);
  // arma::cube Phi_min(Phi(0,0).n_rows, Phi(0,0).n_cols, Phi(0,0).n_slices, arma::fill::zeros);
  // arma::rowvec tau_min(tau.n_cols, arma::fill::zeros);

  dist_min = arma::datum::inf;
  for(int j = 0; j < perm_mat.n_rows; j++){
    for(int k = 0; k < perm_mat.n_cols; k++){
      Z_ph.col(k) = Z.col(perm_mat(j,k));
    }
    dist_current = GetDistanceZ(Z_ph, Z_ref);
    if(dist_current < dist_min){
      dist_min = dist_current;
      Z_min = Z_ph;
      // for(int k = 0; k < perm_mat.n_cols; k++){
      //   nu_min.row(k) = nu.slice(i).row(perm_mat(j,k));
      //   pi_min(k) = pi(perm_mat(j,k), i);
      //   gamma_min.row(k) = gamma(i,0).row(perm_mat(j,k));
      //   Phi_min.row(k) = Phi(i,0).row(perm_mat(j,k));
      //   tau_min(k) = tau(i, perm_mat(j,k));
      // }
    }
  }

  //update chain with corrected parameters
  // Z.slice(i) = Z_min;
  // nu.slice(i) = nu_min;
  // pi.col(i) = pi_min;
  // gamma(i,0) = gamma_min;
  // Phi(i,0) = Phi_min;
  // tau.row(i) = tau_min;

   return Z_min;
}
