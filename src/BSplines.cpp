#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>

arma::Field<arma::mat> Kronecker_B_Spline(const arma::field<arma::mat>& t_obs,
                                          const int n_func,
                                          const arma::vec& basis_degree,
                                          const arma::field<arma::vec>& boundary_knots,
                                          const arma::field<arma::vec>& internal_knots){
  int dim = t_obs(0,0).n_cols;
  int P = 1;
  for(int i = 0; i < basis_degree.n_elem; i++){
    P = P * (internal_knots(i,0).n_elem + basis_degree(i) + 1);
  }

  arma::field<arma::mat> B(100,1);
  for(int i = 0; i < n_func; i++){
    B(i,0) = arma::zeros(t_obs(i,0).n_rows, P);
  }
  arma::field<arma::mat> B_ph(dim, 0);
  for(int i = 0; i < dim; i++){
    splines2::BSpline bspline;
    // Create Bspline object
    bspline = splines2::BSpline(t_obs(i,0).col(i), internal_knots(i,0),
                                basis_degree(i), boundary_knots(i,0));
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat {bspline.basis(true)};
    B_ph(i,0) = bspline_mat;
  }
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < )
    B(i,0)()
  }
}
