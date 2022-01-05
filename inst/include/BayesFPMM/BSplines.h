#ifndef BayesFPMM_B_SPLINES_H
#define BayesFPMM_B_SPLINES_H

#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>

namespace BayesFPMM {
// Creates a tensor B-spline for multivariate functional data
//
// @name TensorBSpline
// @param t_obs field of matrices that contain the observed time points (each column is a dimension)
// @param n_funct integer containing the number of functions observed
// @param basis_degree vector containing the desired basis degree for each dimension
// @param boundary_knots matrix containing the boundary knots for each dimension (each row is a dimension)
// @param internal_knots field of vectors containing the internal knots for each dimension
// @returns B Field of matrices containing the tensor b-splines
inline arma::field<arma::mat> TensorBSpline(const arma::field<arma::mat>& t_obs,
                                            const int n_funct,
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

  arma::field<arma::mat> B(n_funct,1);
  for(int i = 0; i < n_funct; i++){
    B(i,0) = arma::ones(t_obs(i,0).n_rows, P);
  }

  arma::vec counter = arma::zeros(dim);
  int counter_i = 1;
  for(int i = 0; i < P; i++){
    for(int j = 0; j < n_funct; j++){
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

// Creates the P matrix used when updating the nu parameters
//
// @name GetP
// @param basis_degree vector containing the desired basis degree for each dimension
// @param internal_knots field of vectors containing the internal knots for each dimension
// @returns P_mat matrix used to update nu parameters
inline arma::mat GetP(const arma::vec& basis_degree,
               const arma::field<arma::vec>& internal_knots){
  int dim = basis_degree.n_elem;
  int P = 1;
  arma::vec dim_counter = arma::ones(dim);
  for(int i = 0; i < dim; i++){
    P = P * (internal_knots(i,0).n_elem + basis_degree(i) + 1);
  }
  for(int i = (dim - 2); i >= 0; i--){
    dim_counter(i) = dim_counter(i+1) * (internal_knots(i+1,0).n_elem + basis_degree(i+1) + 1);
  }
  int p_pow = 1;
  for(int i = 0; i < dim; i++){
    p_pow = p_pow * (P-1);
  }
  arma::mat Constraint_mat = arma::zeros(p_pow, P);

  arma::vec counter = arma::zeros(dim);
  arma::field<arma::vec> index(P,1);
  int counter_i = 1;
  for(int i = 0; i < P; i++){
    index(i,0) = counter;
    for(int l = 0; l < dim; l++){
      counter(l) = std::fmod(counter_i / dim_counter(l), internal_knots(l,0).n_elem + basis_degree(l) + 1.0);
    }
    counter = arma::floor(counter);
    counter_i++;
  }

  int diff = 0;
  int abs_diff = 0;
  counter_i = 0;
  for(int i = 0; i < P; i++){
    for(int j = i; j < P; j++){
      diff = 0;
      abs_diff = 0;
      for(int l = 0; l < dim; l++){
        diff = diff + (index(j,0)(l) - index(i,0)(l));
        abs_diff = abs_diff + std::abs(index(j,0)(l) - index(i,0)(l));
      }
      if((diff == 1) && (abs_diff == 1)){
        Constraint_mat(counter_i, i) = 1;
        Constraint_mat(counter_i, j) = -1;
        counter_i++;
      }
    }
  }

  arma::mat P_mat = Constraint_mat.t() * Constraint_mat;
  return P_mat;
}

}
#endif
