library(BayesFPMM)

######
######
######
## Run SImulations
##
set.seed(1)
for(i in 1:10){
  x <- TestBFPMM_Nu_Z_multiple_try(2000, 0.001, 1/2, 10, 10000, 100, 2)
  y <- TestBFPMM_Theta(5000, 0.001, x$Z, x$nu, 0.8, 2)
  #dir <- paste("c:\\Projects\\Simulation\\High_Variance\\Trace", as.character(i), "\\", sep = "")
  dir <- "C:\\Projects\\Simulation\\Optimal_K\\2_clusters\\"
  z <- TestBFPMM_MTT_warm_start(1/5, 10, 1000000, 500000, 10000, dir, 0.001, x$Z,
                                x$pi, x$alpha_3, y$delta, y$gamma, y$Phi, y$A, x$nu,
                                x$tau, y$sigma, y$chi, 0.8, 2)
  saveRDS(z, paste(dir, "x_results.RDS", sep = ""))
}

x <- readRDS("c:\\Projects\\Simulation\\High_Variance\\Trace1\\x_results.RDS")
nu_true <- x$nu_true
all_nu <- array(0, dim = c(2,8,3500,10))
all_Z <- array(0, dim =c(100,2,3500,10))
all_f1_97_5 <- array(0,dim = c(100, 10))
all_f2_2_5 <- array(0,dim = c(100,10))
all_f2_97_5 <- array(0,dim = c(100, 10))
all_f1_2_5 <- array(0,dim = c(100,10))
all_Z_97_5 <- array(0,dim = c(100, 10))
all_Z_2_5 <- array(0,dim = c(100,10))
for(j in 1:10){
  print(j)
  nu <- array(0,dim=c(2,8,4000))
  dir = paste("c:\\Projects\\Simulation\\High_Variance\\Trace",j,"\\Nu", sep ="")
  for(i in 0:19){
    nu_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
    nu[,,(200*(i) + 1):(200*(i+1))] <- nu_i
  }
  Z <- array(0,dim=c(100,2,4000))
  dir = paste("c:\\Projects\\Simulation\\High_Variance\\Trace",j,"\\Z", sep ="")
  for(i in 0:19){
    Z_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
    Z[,,(200*(i) + 1):(200*(i+1))] <- Z_i
  }
  nu <- nu[,,501:4000]
  Z <- Z[,,501:4000]

  ## Fix label switching problem
  nu_ph <- nu
  Z_ph <- Z
  nu_ph[1,,] <- nu[2,,]
  Z_ph[,1,] <- Z[,2,]
  nu_ph[2,,] <- nu[1,,]
  Z_ph[,2,] <- Z[,1,]
  if(sum(abs(nu[,,3500]-nu_true)) > sum(abs(nu_ph[,,3500] - nu_true))){
    all_nu[,,,j] <- nu_ph
    all_Z[,,,j] <- Z_ph
    nu <- nu_ph
    Z <- Z_ph
  }else{
    all_nu[,,,j] <- nu
    all_Z[,,,i=j] <- Z
  }

  y <- GetStuff(0.001)
  x <- readRDS("c:\\Projects\\Simulation\\High_Variance\\Trace2\\x_results.RDS")
  f_obs1 <- matrix(0, 2500, 100)
  for(i in 1:1500){
    f_obs1[i,] <- t(y$B[[1]] %*% t(t(nu[1,,i])))
  }

  f1_97_5 <- rep(0,100)
  f1_2_5 <- rep(0,100)
  for(i in 1:100){
    q <- quantile(f_obs1[,i], probs = c(0.025, 0.975))
    f1_2_5[i] <- q[1]
    f1_97_5[i] <- q[2]
  }
  all_f1_2_5[,j]<- f1_2_5
  all_f1_97_5[,j] <- f1_97_5

  f_obs2 <- matrix(0, 3500, 100)
  for(i in 1:3500){
    f_obs2[i,] <- t(y$B[[1]] %*% t(t(nu[2,,i])))
  }

  f2_97_5 <- rep(0,100)
  f2_2_5 <- rep(0,100)
  for(i in 1:100){
    q <- quantile(f_obs2[,i], probs = c(0.025, 0.975))
    f2_2_5[i] <- q[1]
    f2_97_5[i] <- q[2]
  }
  all_f2_2_5[,j]<- f2_2_5
  all_f2_97_5[,j] <- f2_97_5


  Z_97_5 <- rep(0,100)
  Z_2_5 <- rep(0,100)
  for(i in 1:100){
    q <- quantile(Z[i,1,], probs = c(0.025, 0.975))
    Z_2_5[i] <- q[1]
    Z_97_5[i] <- q[2]
  }

  all_Z_2_5[,j] <- Z_2_5
  all_Z_97_5[,j] <- Z_97_5

}

f1_true <- rep(0,100)
f1_true <- x$nu_true[2,] %*% t(y$B[[1]])
plot(f1_true[1,], type = "l", ylim = c(-3,4), lwd = 3)
for(i in 1:10){
  lines(all_f2_2_5[,i], col=i, lty = 2)
  lines(all_f2_97_5[,i], col=i, lty=2)
}

library(plotrix)
Z_true <- x$Z_true[1:10,1]
Z_true <- rep(Z_true, 10)
Z_true <- (as.vector(all_Z_97_5[1:10,1:10]) + as.vector(all_Z_2_5[1:10,1:10]))/2
obs_num <- 1:10
obs_num <- rep(obs_num, 10)
col_num <- 1:100
for(i in 1:10){
  for(j in 1:10){
    col_num[(i-1)*10 + j] <- rainbow(10)[i]
  }
}

plot(1:10, x$Z_true[1:10,1])
plotCI(obs_num, Z_true, ui=as.vector(all_Z_97_5[1:10,1:10]), li = as.vector(all_Z_2_5[1:10,1:10]), scol = col_num, col ="#1C00ff00", add=T)

######
######
######
######
## Get Z estimates
Z <- array(0,dim=c(100,3,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Z"
for(i in 0:19){
  Z_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  Z[,,(200*(i) + 1):(200*(i+1))] <- Z_i
}


## Get nu estimates
nu <- array(0,dim=c(3,8,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Nu"
for(i in 0:19){
  nu_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  nu[,,(200*(i) + 1):(200*(i+1))] <- nu_i
}

chi <- array(0, dim=c(100,3,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Chi"
for(i in 0:19){
  chi_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  chi[,,(200*(i) + 1):(200*(i+1))] <- chi_i
}

Phi <- array(0,dim=c(3,8,3,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Phi"
for(i in 0:19){
  Phi_i <- TestReadField(paste(dir, as.character(i),".txt", sep = ""))
  for(j in 1:200){
    Phi[,,,(i)*200 + j] = Phi_i[[j]]
  }
}

##
## Calculate DIC and BIC
##

## Estimate curve level fit
##
##
##
Phi <- Phi[,,,500:4000]
nu <- nu[,,500:4000]
Z <- Z[,,500:4000]
chi <- chi[,,500:4000]

y <- GetStuff(0.001)
x <- readRDS("c:\\Projects\\trace\\x_results.RDS")

for(k in 1:100){
  f_i <- matrix(0, 3500, 100)
  fun = k
  for(i in 1:3500){
    f_i[i,] <- f_i[i,] + Z[fun, , i] %*% (nu[,,i]) %*% t(y$B[[1]])
    for(j in 1:3){
      f_i[i,] <- f_i[i,] + chi[fun,j,i] * Z[fun,,i] %*% Phi[,,j, i] %*% t(y$B[[1]])
    }
  }

  for(i in 1:100){
    z <- quantile(f_i[,i], probs = c(0.025, 0.975))
    f3_2_5[i] <- z[1]
    f3_97_5[i] <- z[2]
  }
  plot(x$y_obs[[fun]], type = 'l', ylab = "observed data")
  lines(f3_2_5, col = "red")
  lines(f3_97_5, col = "red")
}

##
##
##

## Get rid of burn-in
nu <- nu[,,1001:4000]
y <- GetStuff(0.001)
x <- readRDS("C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\x_results.RDS")
f_obs1 <- matrix(0, 1500, 100)
for(i in 1:1500){
  f_obs1[i,] <- t(y$B[[1]] %*% t(t(nu[1,,i])))
}
f1_97_5 <- rep(0,100)
f1_2_5 <- rep(0,100)
for(i in 1:100){
  z <- quantile(f_obs1[,i], probs = c(0.025, 0.975))
  f1_2_5[i] <- z[1]
  f1_97_5[i] <- z[2]
}
f1_true <- rep(0,100)
f1_true <- x$nu_true[3,] %*% t(y$B[[1]])


plot(f1_true[1,], type = 'l', ylab = "function 1")
lines(f1_2_5, col = "red")
lines(f1_97_5, col = "red")

## Function 2
f_obs2 <- matrix(0, 1000, 100)
for(i in 1:1000){
  f_obs2[i,] <- t(y$B[[1]] %*% t(t(nu[2,,i])))
}
f2_97_5 <- rep(0,100)
f2_2_5 <- rep(0,100)
for(i in 1:100){
  z <- quantile(f_obs2[,i], probs = c(0.025, 0.975))
  f2_2_5[i] <- z[1]
  f2_97_5[i] <- z[2]
}
f2_true <- rep(0,100)
f2_true <- x$nu_true[2,] %*% t(y$B[[1]])

plot(f2_true[1,], type = 'l', ylab = "function 2")
lines(f2_2_5, col = "red")
lines(f2_97_5, col = "red")

## Function 3

f_obs3 <- matrix(0, 1500, 100)
for(i in 1:1500){
  f_obs3[i,] <- t(y$B[[1]] %*% t(t(nu[3,,i])))
}
f3_97_5 <- rep(0,100)
f3_2_5 <- rep(0,100)
for(i in 1:100){
  z <- quantile(f_obs3[,i], probs = c(0.025, 0.975))
  f3_2_5[i] <- z[1]
  f3_97_5[i] <- z[2]
}
f3_true <- rep(0,100)
f3_true <- x$nu_true[1,] %*% t(y$B[[1]])

plot(f3_true[1,], type = 'l', ylab = "function 3")
lines(f3_2_5, col = "red")
lines(f3_97_5, col = "red")


### Phi

## Get nu estimates
Phi <- array(0,dim=c(2,8,3,600))
dir = "c:\\Projects\\trace\\Phi"
for(i in 0:2){
  Phi_i <- TestReadField(paste(dir, as.character(i),".txt", sep = ""))
  for(j in 1:200){
    Phi[,,,(i)*200 + j] = Phi_i[[j]]
  }
}

## Get rid of Burn-in
Phi <- Phi[,,,200:600]


f1_var <- array(0, dim= c(100, 100, 400))
for(i in 1:400){
  for(j in 1:100){
    for(k in 1:100){
      for(m in 1:3){
        f1_var[j,k,i] <- f1_var[j,k,i] + Phi[1, ,m ,i] %*% t(t(y$B[[1]][j,])) %*% y$B[[1]][k,] %*% t(t(Phi[1, ,m ,i]))
      }
    }
  }
  print(i)
}
cov1_97_5 <- matrix(0,100,100)
cov1_2_5 <- matrix(0,100,100)
for(j in 1:100){
  for(k in 1:100){
    z <- quantile(f1_var[j,k,], probs = c(0.025, 0.975))
    cov1_2_5[j,k] <- z[1]
    cov1_97_5[j,k] <- z[2]
  }
}

cov1_true <- matrix(0, 100, 100)

y <- readRDS("c:\\Projects\\trace\\x_results.RDS")
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov1_true[j,k] <- cov1_true[j,k] + y$Phi_true[1, ,m] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(y$Phi_true[1, ,m]))
    }
  }
}

library(plotly)

fig <- plot_ly(showscale = FALSE)
fig <- fig %>% add_surface(z = ~cov1_true)
fig <- fig %>% add_surface(z = ~cov1_2_5, opacity = 0.50)
fig <- fig %>% add_surface(z = ~cov1_97_5, opacity = 0.50)

fig



#### Cov 2

f2_var <- array(0, dim= c(100, 100, 3000))
for(i in 1:3000){
  for(j in 1:100){
    for(k in 1:100){
      for(m in 1:3){
        f2_var[j,k,i] <- f2_var[j,k,i] + Phi[2, ,m ,i] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(Phi[2, ,m ,i]))
      }
    }
  }
  print(i)
}
cov2_97_5 <- matrix(0,100,100)
cov2_2_5 <- matrix(0,100,100)
for(j in 1:100){
  for(k in 1:100){
    z <- quantile(f2_var[j,k,], probs = c(0.025, 0.975))
    cov2_2_5[j,k] <- z[1]
    cov2_97_5[j,k] <- z[2]
  }
}

cov2_true <- matrix(0, 100, 100)

y <- readRDS("/Users/nicholasmarco/Projects/FDA/Trace/rstudio-export/x_results.RDS")
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov2_true[j,k] <- cov2_true[j,k] + y$Phi_true[2, ,m] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(y$Phi_true[2, ,m]))
    }
  }
}

library(plotly)

fig2 <- plot_ly(showscale = FALSE)
fig2 <- fig2 %>% add_surface(z = ~cov2_true)
fig2 <- fig2 %>% add_surface(z = ~cov2_2_5, opacity = 0.50)
fig2 <- fig2 %>% add_surface(z = ~cov2_97_5, opacity = 0.50)

fig2


### Cov 3

f3_var <- array(0, dim= c(100, 100, 3000))
for(i in 1:3000){
  for(j in 1:100){
    for(k in 1:100){
      for(m in 1:3){
        f3_var[j,k,i] <- f3_var[j,k,i] + Phi[3, ,m ,i] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(Phi[3, ,m ,i]))
      }
    }
  }
  print(i)
}
cov3_97_5 <- matrix(0,100,100)
cov3_2_5 <- matrix(0,100,100)
for(j in 1:100){
  for(k in 1:100){
    z <- quantile(f3_var[j,k,], probs = c(0.025, 0.975))
    cov3_2_5[j,k] <- z[1]
    cov3_97_5[j,k] <- z[2]
  }
}

cov3_true <- matrix(0, 100, 100)

y <- readRDS("/Users/nicholasmarco/Projects/FDA/Trace/rstudio-export/x_results.RDS")
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov3_true[j,k] <- cov3_true[j,k] + y$Phi_true[3, ,m] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(y$Phi_true[3, ,m]))
    }
  }
}

library(plotly)

fig3 <- plot_ly(showscale = FALSE)
fig3 <- fig3 %>% add_surface(z = ~cov3_true)
fig3 <- fig3 %>% add_surface(z = ~cov3_2_5, opacity = 0.50)
fig3 <- fig3 %>% add_surface(z = ~cov3_97_5, opacity = 0.50)

fig3
