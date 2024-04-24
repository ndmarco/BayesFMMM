
[![R-CMD-check](https://github.com/ndmarco/BayesFMMM/workflows/R-CMD-check/badge.svg)](https://github.com/ndmarco/BayesFMMM/actions)

# BayesFMMM
An R package for fitting Bayesian mixed membership models for functional data and multivariate data. Mixed membership models, also sometimes reffered to as partial membership models, can be thought of as a generalization of traditional clustering models, where each observation can partially belong to multiple clusters or features. 

***

## Functional Mixed Membership Models
We will let $Y_i(t_i)$ be the observed sample paths at the time points $t_i = [t_{i1}, \dots, t_{in_i}]'$ for $i = 1, \dots,$ n_funct. As described in the manuscript [Functional Mixed Membership Models](https://arxiv.org/abs/2206.12084), we will assume that the underlying GPs are smooth and lie in the $P$-dimensional subspace spanned by a set of linearly independent square-integrable basis functions $\{b_1, \dots, b_P\}$. Specifically, we will assume that the basis functions are B-splines. Let $X \in \mathbb{R}^{N \times D}$ be the covariates of interest, where $x_i = [X_{i1} \dots X_{iR}]$ denotes the $i^{th}$ row of the design matrix (or the covariates associated with the $i^{th}$ observation). Letting $B'(t_i):= [b_1(t_i), b_2(t_i), \dots, b_P(t_i)]$ and $S(t_i) = \[B(t_1) \cdots B(t_{n_i})\] \in \mathbb{R}^{P \times n_i}$, we will assume that each observation can be written as a convex combination of the underlying clusters, or features, in our model. Thus assuming that there are $K$ clusters, we can specify our model through equivalence in distribution in the following way:
$$Y_i(\cdot) = \sum_{k=1}^K Z_{ik}f^{(k)}(\cdot) + \epsilon_i,$$
where $f^{(k)}(.)$ denotes the distibution of observations from the $k^{th}$ feature, $Z_{ik}$ are allocation parameters such that $0 \le Z_{ik} \le 1$ and $\sum_{k} Z_{ik} = 1$, and $\epsilon_i$ is a mean-zero GP with variance equal to $\sigma^2$ and covariance equal to 0. We will assume that $f^{(k)} \sim \mathcal{GP}(\mu^{(k)}, C^{(k,k)})$, and the covariance between $f^{(k)}$ and $f^{(j)}$ is $C^{(k,j)}$. Since we assume that the underlying GPs lie in the $P$-dimensional subspace spanced by $\{b_1, \dots, b_P\}$ and utilizing the Mutlivariate KL decompositon, we have that $$\mu^{(k)}(t_i) = S'(t_i) \nu_k$$ and $$C^{(k,j)}(t_i, s_i) \approx \sum_{m=1}^M S'(t_i) \phi_{km}\phi_{jm}' S(s_i),$$ where $\phi_{jm} S(t_i)$ are pseudo-eigenfunctions (not orthogonal). If $M = KP$, then we are using all of the eigenfunctions meaning we are not approximating the covariance structure by using a truncated expansion. Thus, we arrive at the likelihood of our model:

$$Y_i(t_i) \mid \Theta, {\color{purple}X} \sim  \mathcal{N}\left[ \sum_{k=1}^K Z_{ik}\left(S'(t_i) \left(\nu_k + {\color{red}\eta_k x_i'}\right) + \sum_{m=1}^{M}\chi_{im}S'(t_i) \left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right)\right), \sigma^2 I_{n_i}\right].$$

In the above formula, the color ${\color{red}\text{red}}$ denotes the added parameters from a covariate adjusted mixed membership model, where the mean structure depends on the covariates of interest. Under the covariate adjusted mean framework, we can see that each feature's mean is dependent on the covariates of interst ($\mu^{(k)}(t_i) = S'(t_i) (\nu_k + {\color{red}\eta_k x_i'}) $). The color ${\color{blue}\text{blue}}$ denotes the added parameters from a covariate adjusted mixed membership model, where the covariance structure depends on the covariates of interest. Under the covariate adjusted covariance model, we can see that the pseudo-eigenfunctions are now dependent on the covariates of interest (the $m^{th}$ eigenfunction is $S'(t_i) \left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right) $). If we integrate out the $\chi_{im}$ parameters, we arrive at the following model:

$$Y_i(t_i) \mid \Theta_{-\chi_{..}}, {\color{purple}X} \sim  \mathcal{N}\left[ \sum_{k=1}^K Z_{ik}\left(S'(t_i) \left(\nu_k + {\color{red}\eta_k x_i'}\right)\right), V_i + \sigma^2 I_{n_i}\right],$$

where the error-free covariance $V_i$ is 
$$V_i = \sum_{k=1}^K\sum_{k'=1}^K Z_{ik}Z_{ik'}\sum_{m=1}^M S'(t_i) \left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right)\left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right)'S'(t_i)$$
For prior specification, see [Functional Mixed Membership Models](https://arxiv.org/abs/2206.12084).

### Parameters

- MCMC_iters := Number of MCMC iterations
- K := Number of features (or clusters)
- P := Number of basis functions
- n_funct := Number of functions
- M := Number of pseudo-eigenfunctions
- D := Number of covariates

Parameter | Size | Description
--- | --- | --- 
$\nu$ | (K, P, MCMC_iters) | Parameters controlling the mean function for each covariate
$\tau$ | (MCMC_iters, K) | Paramaters used in the prior on $\nu$ which control the smoothing of the mean function
${\color{red}\eta}$ | [MCMC_iters] (P, D, K) | Parameters controlling the covariate dependence of the mean functions
${\color{red}\tau_\eta}$ | (K, D, MCMC_iters) | Paramaters used in the prior on ${\color{red}\eta}$ which control the smoothing of the mean function
$\chi$ |(n_funct, M, MCMC_iters) | Parameters controling the amount of variation from the mean in the directions of the pseudo eigenfunctions
$Z$ | (n_funct, K, MCMC_iters) | Parameters indicating an observation's proportion of membership to each cluster
$\pi$ | (K, MCMC_iters) |  Paramaters used in the prior on the $Z$ parameters
$\alpha_3$ | (MCMC_iters) | Paramater used in the prior on the $Z$ parameters
$\sigma^2$ | (MCMC_iters) | Parameter controlling the variance
$\Phi (\phi_{km})$ | [MCMC_iters] (K, P, M)| Parameters constructing the pseudo eigenfunctions
$\gamma$ | [MCMC_iters] (K, P, M) | Parameters used in the prior on the $\Phi$ parameters
$\delta$ | (K, M, MCMC_iters) | Parameters used in the prior on the $\Phi$ parameters
$A$ | (K, 2, MCMC_iters) | Parameters used in the prior on $\delta$ paramaters
${\color{blue}\Xi (\xi_{km})}$ | [MCMC_iters, K] (P, D, M) | Parameters controlling the covariate dependence of the pseudo-eigenfunctions
${\color{blue}\gamma_\Xi}$ | [MCMC_iters, K] (P, D, M) | Parameters used in the prior on the ${\color{blue}\Xi}$ parameters
${\color{blue}\delta_\Xi}$ | [MCMC_iters] (K, M, D) | Parameters used in the prior on the ${\color{blue}\Xi}$ parameters
${\color{blue}A_\Xi}$ | [MCMC_iters] (K, 2, D) | Parameters used in the prior on the ${\color{blue}\delta_\Xi}$ paramaters

***

## Multivariate Gaussian Mixed Membership Models

Let $Y_i \in \mathbb{R}^P$ be the observed data points for $i = 1, \dots, N$. Let $X \in \mathbb{R}^{N \times D}$ be the covariates of interest, where $x_i = [X_{i1} \dots X_{iR}]$ denotes the $i^{th}$ row of the design matrix (or the covariates associated with the $i^{th}$ observation). Similarly to the functional case, and as explained in detail in [Flexible Regularized Estimation in High-Dimensional Mixed Membership Models](https://arxiv.org/abs/2212.06906), we will assume that
$$Y_i = \sum_{k=1}^K Z_{ik}f^{(k)} + \epsilon_i,$$
where $f^{(k)}$ denotes the distibution of observations from the $k^{th}$ feature, $Z_{ik}$ are allocation parameters such that $0 \le Z_{ik} \le 1$ and $\sum_{k} Z_{ik} = 1$, and $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. We will assume that $f^{(k)} \sim \mathcal{N}(\mu^{(k)}, C^{(k,k)})$, and the covariance between $f^{(k)}$ and $f^{(j)}$ is $C^{(k,j)}$. Utilizing an eigen decompositon, we have that $$\mu^{(k)} = \nu_k$$ and $$C^{(k,j)} \approx \sum_{m=1}^M \phi_{km}\phi_{jm}',$$ where $\phi_{jm} S(t_i)$ are pseudo-eigenvectors (not orthogonal). If $M = KP$, then we are using all of the eigenvectors meaning we are not approximating the covariance structure by using a truncated expansion. Thus, we arrive at the likelihood of our model:

$$Y_i \mid \Theta, {\color{purple}X} \sim  \mathcal{N}\left[ \sum_{k=1}^K Z_{ik}\left( \left(\nu_k + {\color{red}\eta_k x_i'}\right) + \sum_{m=1}^{M}\chi_{im} \left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right)\right), \sigma^2 I_{n_i}\right].$$

In the above formula, the color ${\color{red}\text{red}}$ denotes the added parameters from a covariate adjusted mixed membership model, where the mean structure depends on the covariates of interest. Under the covariate adjusted mean framework, we can see that each feature's mean is dependent on the covariates of interst ($\mu^{(k)} =\nu_k + {\color{red}\eta_k x_i'} $). The color ${\color{blue}\text{blue}}$ denotes the added parameters from a covariate adjusted mixed membership model, where the covariance structure depends on the covariates of interest. Under the covariate adjusted covariance model, we can see that the pseudo-eigenvectors are now dependent on the covariates of interest (the $m^{th}$ eigenvector is $\left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right) $). If we integrate out the $\chi_{im}$ parameters, we arrive at the following model:

$$Y_i \mid \Theta_{-\chi_{..}}, {\color{purple}X} \sim  \mathcal{N}\left[ \sum_{k=1}^K Z_{ik}\left( \left(\nu_k + {\color{red}\eta_k x_i'}\right)\right), V_i + \sigma^2 I_{P}\right],$$

where the error-free covariance $V_i$ is 
$$V_i = \sum_{k=1}^K\sum_{k'=1}^K Z_{ik}Z_{ik'}\sum_{m=1}^M \left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right)\left(\phi_{km} + {\color{blue}\xi_{km}x_i'}\right)'$$
For prior specification, see [Flexible Regularized Estimation in High-Dimensional Mixed Membership Models](https://arxiv.org/abs/2212.06906).

### Parameters

- MCMC_iters := Number of MCMC iterations
- K := Number of features (or clusters)
- P := Dimension of multivariate Data
- N := Number of observations
- M := Number of pseudo-eigenfunctions
- D := Number of covariates

Parameter | Size | Description
--- | --- | --- 
$\nu$ | (K, P, MCMC_iters) | Parameters controlling the mean function for each covariate
$\tau$ | (MCMC_iters, K) | Paramaters used in the prior on $\nu$ which control the smoothing of the mean function
${\color{red}\eta}$ | [MCMC_iters] (P, D, K) | Parameters controlling the covariate dependence of the mean functions
${\color{red}\tau_\eta}$ | (K, D, MCMC_iters) | Paramaters used in the prior on ${\color{red}\eta}$ which control the smoothing of the mean function
$\chi$ |(N, M, MCMC_iters) | Parameters controling the amount of variation from the mean in the directions of the pseudo eigenfunctions
$Z$ | (N, K, MCMC_iters) | Parameters indicating an observation's proportion of membership to each cluster
$\pi$ | (K, MCMC_iters) |  Paramaters used in the prior on the $Z$ parameters
$\alpha_3$ | (MCMC_iters) | Paramater used in the prior on the $Z$ parameters
$\sigma^2$ | (MCMC_iters) | Parameter controlling the variance
$\Phi (\phi_{km})$ | [MCMC_iters] (K, P, M)| Parameters constructing the pseudo eigenfunctions
$\gamma$ | [MCMC_iters] (K, P, M) | Parameters used in the prior on the $\Phi$ parameters
$\delta$ | (K, M, MCMC_iters) | Parameters used in the prior on the $\Phi$ parameters
$A$ | (K, 2, MCMC_iters) | Parameters used in the prior on $\delta$ paramaters
${\color{blue}\Xi (\xi_{km})}$ | [MCMC_iters, K] (P, D, M) | Parameters controlling the covariate dependence of the pseudo-eigenfunctions
${\color{blue}\gamma_\Xi}$ | [MCMC_iters, K] (P, D, M) | Parameters used in the prior on the ${\color{blue}\Xi}$ parameters
${\color{blue}\delta_\Xi}$ | [MCMC_iters] (K, M, D) | Parameters used in the prior on the ${\color{blue}\Xi}$ parameters
${\color{blue}A_\Xi}$ | [MCMC_iters] (K, 2, D) | Parameters used in the prior on the ${\color{blue}\delta_\Xi}$ paramaters

***

#### Associated Repositories
 1. [Simulation Studies and Case Studies for Functional Data](https://github.com/ndmarco/BFMMM_Functional_Sims)
 2. [Simulation Studies and Case Studies for Multivariate Data](https://github.com/ndmarco/BFMMM_Multivariate_Sims)
 
#### Associated Papers
 1. [Functional Mixed Membership Models](https://www.tandfonline.com/doi/full/10.1080/10618600.2024.2304633)
 2. [Flexible Regularized Estimation in High-Dimensional Mixed Membership Models](https://www.sciencedirect.com/science/article/abs/pii/S016794732400015X)
