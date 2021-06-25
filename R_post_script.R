library(BayesFPMM)

## Get Z estimates
Z <- array(0,dim=c(100,2,2000))
dir = "c:\\Projects\\Simulation\\Z"
for(i in 0:1){
  Z_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  Z[,,(200*(i) + 1):(200*(i+1))] <- Z_i
}


## Get nu estimates
nu <- array(0,dim=c(2,8,600))
dir = "c:\\Projects\\trace\\Nu"
for(i in 0:2){
  nu_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  nu[,,(200*(i) + 1):(200*(i+1))] <- nu_i
}

## Get rid of burn-in
nu <- nu[,,200:600]
y <- GetStuff(0.001)
x <- readRDS("c:\\Projects\\trace\\x_results.RDS")
f_obs1 <- matrix(0, 200, 100)
for(i in 1:400){
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
f1_true <- x$nu_true[1,] %*% t(y$B[[1]])

plot(f1_true[1,], type = 'l', ylab = "function 1")
lines(f1_2_5, col = "red")
lines(f1_97_5, col = "red")

## Function 2
f_obs2 <- matrix(0, 2000, 100)
for(i in 1:2000){
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

f_obs3 <- matrix(0, 400, 100)
for(i in 1:400){
  f_obs3[i,] <- t(y$B[[1]] %*% t(t(nu[2,,i])))
}
f3_97_5 <- rep(0,100)
f3_2_5 <- rep(0,100)
for(i in 1:100){
  z <- quantile(f_obs3[,i], probs = c(0.025, 0.975))
  f3_2_5[i] <- z[1]
  f3_97_5[i] <- z[2]
}
f3_true <- rep(0,100)
f3_true <- x$nu_true[2,] %*% t(y$B[[1]])

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
