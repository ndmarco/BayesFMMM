#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>

arma::field<arma::mat> Kronecker_B_Spline(const arma::field<arma::mat>& t_obs,
                                          const int n_func,
                                          const arma::vec& basis_degree,
                                          const arma::mat& boundary_knots,
                                          const arma::field<arma::vec>& internal_knots){
  int dim = t_obs(0,0).n_cols;
  int P = 1;
  arma::vec dim_counter = arma::ones(dim);
  for(int i = 0; i < dim; i++){
    P = P * (internal_knots(i,0).n_elem + basis_degree(i) + 1);
  }
  for(int i = (dim - 2); i >= 0; i--){
    dim_counter(i) = dim_counter(i+1) * (internal_knots(i+1,0).n_elem + basis_degree(i+1) + 1);
  }

  arma::field<arma::mat> B(n_func,1);
  for(int i = 0; i < n_func; i++){
    B(i,0) = arma::ones(t_obs(i,0).n_rows, P);
  }

  arma::vec counter = arma::zeros(dim);
  int counter_i = 1;
  for(int i = 0; i < P; i++){
    for(int j = 0; j < n_func; j++){
      arma::field<arma::mat> B_ph(dim, 1);
      for(int l = 0; l < dim; l++){
        splines2::BSpline bspline;
        // Create Bspline object
        bspline = splines2::BSpline(t_obs(j,0).col(l), internal_knots(l,0),
                                    basis_degree(l), boundary_knots.row(l).t());
        arma::mat bspline_mat{bspline.basis(true)};
        B_ph(l,0) = bspline_mat;
        for(int k = 0; k < t_obs(j,0).n_rows; k++){
          B(j,0)(k,i) = B(j,0)(k,i) * B_ph(l,0)(k, counter(l));
        }
      }
    }
    for(int l = 0; l < dim; l++){
      counter(l) = std::fmod(counter_i / dim_counter(l), internal_knots(l,0).n_elem + basis_degree(l) + 1.0);
    }
    counter = arma::floor(counter);
    counter_i++;
  }
  return B;
}
