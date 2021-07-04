#ifndef LABELSWITCH_H
#define LABELSWITCH_H

#include <RcppArmadillo.h>

arma::mat LabelSwitch(arma::mat& Z_ref,
                      arma::mat& Z,
                      const arma::mat& perm_mat);
#endif
