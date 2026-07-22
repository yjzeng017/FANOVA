
# User should specify the paths
root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'
simulation_path = paste(root_path, 'simulation/additive/', sep = '')
data_path = paste(root_path, 'simulation/additive/data/', sep = '')


# load the methods for comprison
library(foreach)
library(doSNOW)
source(paste(root_path, 'benchmarks/mars.R', sep = ''))
source(paste(root_path, 'benchmarks/acosso.R', sep = ''))
source(paste(root_path, 'benchmarks/bssanova.R', sep = ''))


########### for loading the .pkl data  ##################
require(reticulate) 
pickle = import('pickle')
builtins = import('builtins')


file_name = function(cov, param){
  if(cov == 'CompSymm') return(paste(cov, 't', param, '100_replications', sep = '_'))
  if(cov == 'AR(1)') return(paste(cov, 'rho', param, '100_replications', sep = '_'))
}

################### simulation: model 1 - additive ####################################
# parameter setting
covs = c("CompSymm", "AR(1)")
t = c(0, 1, 3)
rho = c(-0.5, 0, 0.5)
method = c('MARS', 'COSSO', 'ACOSSO')
# BSSANOVA will take more time. When running testing, BSSANOVA could be excluded.
n = 100 
p = 10
npar = 10 # User can run less replicated simulations for testing.


# open a .txt file
writeLines("#######################################", paste(simulation_path, 'benchmarks_results.txt',sep = ''))
cat(" Simulation: additive model \n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
cat("####################################### \n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)

for (cov in covs) {
  for (j in 1:3) {
    if(cov == "CompSymm") param = t[j]
    if(cov == "AR(1)") param = rho[j]
    
    # load data
    # X_train = get(load(paste(R_data_path,'/X_train_', file_name(cov, param), '.Rdata', sep = '')))
    # X_test = get(load(paste(R_data_path,'/X_test_', file_name(cov, param), '.Rdata', sep = '')))
    # Y_train = get(load(paste(R_data_path,'/Y_train_', file_name(cov, param), '.Rdata', sep = '')))
    # Y_test = get(load(paste(R_data_path,'/Y_test_', file_name(cov, param), '.Rdata', sep = '')))
    # 
    
    X_train = pickle$load(builtins$open(paste(data_path,'X_train_', file_name(cov, param), '.pkl', sep = ''), 'rb'))
    X_test = pickle$load(builtins$open(paste(data_path,'X_test_', file_name(cov, param), '.pkl', sep = ''), 'rb'))
    Y_train = pickle$load(builtins$open(paste(data_path,'Y_train_', file_name(cov, param), '.pkl', sep = ''), 'rb'))
    Y_test = pickle$load(builtins$open(paste(data_path,'Y_test_', file_name(cov, param), '.pkl', sep = ''), 'rb'))
    
    cat("Covariance structure:", cov, "t or rho:", param, "\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
    cat("--------------------------------------- \n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
    for (m in method) {
      # cat("Method", " & ISE", paste("& ", 1:ncol(X)), "\n", file = paste(simulation_path, '/benchmarks_results.txt',sep = ''), append = TRUE)
      # set up progress bar
      pb <- txtProgressBar(max = npar, style = 3)
      progress <- function(n) setTxtProgressBar(pb, n)
      opts <- list(progress = progress)
      # adapt number of kernels used
      cl = makeCluster(5)
      registerDoSNOW(cl)
      
      result = foreach(i=1:npar, .combine = 'rbind', .options.snow = opts, .packages = c('earth','MASS','quadprog')) %dopar% {
        
        X = X_train[[i]]
        Y = as.numeric(Y_train[[i]])
        Xt = X_test[[i]]
        Yt = as.numeric(Y_test[[i]])
        
        if(m == 'MARS'){
          model = MARS(X=X, y=Y, degree = 1)
          ISE = mean((Yt - MARS.predict(model, Xt))**2)
          main = MARS.selection(model)$main
        }
        if(m == 'COSSO') {
          model = ACOSSO(X=X, y=Y, order = 1, wt.pow = 0, cv='gcv')
          ISE = mean((Yt - ACOSSO.predict(model, Xt))**2)
          main = ACOSSO.selection(model)$main
        }
        if(m == 'ACOSSO') {
          model = ACOSSO(X=X, y=Y, order = 1, cv='gcv')
          ISE = mean((Yt - ACOSSO.predict(model, Xt))**2)
          main = ACOSSO.selection(model)$main
        }
        
        if(m == 'BSSANOVA') {
          model = BSSANOVA(y=Y, x=X, main = T, twoway = F, runs = 3000, burn = 1000)
          ISE = mean((Yt - BSSANOVA.predict(model, Xt))**2)
          main = BSSANOVA.selection(model)$main
        }
        variable = rep(0, ncol(X))
        variable[main] = 1
        model_size = rep(0, ncol(X))
        if(length(main) >0) { model_size[length(main)] = 1 }
        return(c(round(ISE, 3), variable, model_size))
      }
      stopCluster(cl)
      
      cat("Method: ", m, "\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
      
      cat('ISE: ', paste(mean(result[, 1]), '(', sd(result[, 1])/sqrt(npar), ')', sep = ''), "\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
      cat('Variable: ', paste(" & ", 1:p), "\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
      cat('Frequency:', paste(" & ", apply(result[, 2:(p+1)], 2, sum)), "\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
      cat('Model size: ', paste(" & ", 1:p), "\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
      cat('Frequency:', paste(" & ", apply(result[, (p+2):(2*p+1)], 2, sum)), "\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
      cat('\n', file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
    }
    
    cat("--------------------------------------- \n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
    cat("\n", file = paste(simulation_path, 'benchmarks_results.txt',sep = ''), append = TRUE)
    
  }
}
