% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{ReadCube}
\alias{ReadCube}
\title{Reads saved parameter data (nu, chi, Z)}
\usage{
ReadCube(file)
}
\arguments{
\item{file}{String containing location where armadillo cube is stored}
}
\value{
Cube Array containing the saved data
}
\description{
Reads armadillo cube type data and returns it as an array in R. The following
parameters can be read in using this function: nu, chi, and Z.
}
\examples{
## set file path
file <- system.file("test-data", "nu.txt", package = "BayesFMMM")

## Read in file
nu <- ReadCube(file)

#############################################################
## For reading in a group of files you can use the following:
#
## Set number of files you want to read in
# n_files <- 50
#
## Set number of samples per file
# n_samp <- 100
#
## Set dim of parameter
# dim1 <- 3
# dim2 <- 8
#
## Set directory
# dir <- "~/Nu"
#
## initialize placeholder
# nu <- array(0, dim = c(dim1, dim2, n_files * n_samp))
# for(i in 0:(n_files - 1)){
#   nu_i <- ReadCube(paste(dir, as.character(i),".txt", sep = ""))
#   nu[,,((n_samp * i) + 1):(n_samp * (i+1))] <- nu_i
#}
#############################################################

}
