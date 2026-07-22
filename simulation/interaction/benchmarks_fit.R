
# User should specify the paths
root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'
simulation_path = paste(root_path, 'simulation/interaction/', sep = '')
data_path = paste(root_path, 'simulation/interaction/data/', sep = '')
file_path = paste(simulation_path, 'benchmarks_results.txt', sep = '')


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




file_name = function(N_train, p){
  return(paste('Uniform_N_train', N_train, 'p', p, '100_replications', sep = '_'))
}

################### simulation: model 2 - interaction ####################################


# parameter setting
method = c('MARS', 'COSSO', 'ACOSSO', 'BSSANOVA') 
# BSSANOVA will take more time. When running testing, BSSANOVA could be excluded.

cores = 8 # for parallel computing
npar = 100 # User can run fewer simulations for testing.


# open a .txt file
writeLines("#######################################", file_path)
cat(" Simulation: Interaction model \n", file = file_path, append = TRUE)
cat("####################################### \n", file = file_path, append = TRUE)


for(p in c(10)) {
  for (n in c(100, 200, 400)) {
    # load data
    
    X_train = pickle$load(builtins$open(paste(data_path,'X_train_', file_name(n, p), '.pkl', sep = ''), 'rb'))
    X_test = pickle$load(builtins$open(paste(data_path,'X_test_', file_name(n, p), '.pkl', sep = ''), 'rb'))
    Y_train = pickle$load(builtins$open(paste(data_path,'Y_train_', file_name(n, p), '.pkl', sep = ''), 'rb'))
    Y_test = pickle$load(builtins$open(paste(data_path,'Y_test_', file_name(n, p), '.pkl', sep = ''), 'rb'))
    
    cat("Unform, N_train = ", n, 'p = ', p, "\n", file = file_path, append = TRUE)
    cat("--------------------------------------- \n", file = file_path, append = TRUE)
    for (m in method) {
      # set up progress bar
      pb <- txtProgressBar(max = npar, style = 3)
      progress <- function(n) setTxtProgressBar(pb, n)
      opts <- list(progress = progress)
      # adapt number of kernels used
      cl = makeCluster(cores)
      registerDoSNOW(cl)
      
      result = foreach(i=1:npar, .options.snow = opts, .packages = c('earth','MASS','quadprog')) %dopar% {
        
        X = X_train[[i]]
        Y = as.numeric(Y_train[[i]])
        Xt = X_test[[i]]
        Yt = as.numeric(Y_test[[i]])
        
        start_time = Sys.time()

        if(m == 'MARS'){
          model = MARS(X=X, y=Y, degree = 2, penalty = 3)
          ISE = mean((Yt - MARS.predict(model, Xt))**2)
          selected.main = MARS.selection(model)$main
          selected.interaction = matrix(MARS.selection(model)$interaction, ncol=2)
          model.size = length(selected.main) + dim(selected.interaction)[1]
        }
        
        if(m == 'COSSO') {
          model = ACOSSO(X=X, y=Y, order = 2, wt.pow = 0, cv='gcv')
          ISE = mean((Yt - ACOSSO.predict(model, Xt))**2)
          selected.main = ACOSSO.selection(model)$main
          selected.interaction = ACOSSO.selection(model)$interaction
          model.size = length(selected.main)
          if(!is.null(selected.interaction)) {
            model.size = model.size + dim(selected.interaction)[1]
          }
        }
        
        if(m == 'ACOSSO') {
          model = ACOSSO(X=X, y=Y, order = 2, cv='gcv')
          ISE = mean((Yt - ACOSSO.predict(model, Xt))**2)
          selected.main = ACOSSO.selection(model)$main
          selected.interaction = ACOSSO.selection(model)$interaction
          model.size = length(selected.main)
          if(!is.null(selected.interaction)) {
            model.size = model.size + dim(selected.interaction)[1]
          }
        }
        
        if(m == 'BSSANOVA') {
          model = BSSANOVA(y=Y, x=X, main = T, twoway = T, runs = 3000, burn = 1000)
          ISE = mean((Yt - BSSANOVA.predict(model, Xt))**2)
          selected.main = BSSANOVA.selection(model)$main
          selected.interaction = BSSANOVA.selection(model)$interaction
          model.size = length(selected.main)
          if(!is.null(selected.interaction)) {model.size = model.size + dim(selected.interaction)[1]}
        }
        
        end_time = Sys.time()
        runtime = as.numeric(difftime(end_time, start_time, units = "secs"))
        
        main.freq = rep(0, p)
        main.freq[selected.main] = 1
        
        # Find matching row indices
        pairs <- t(combn(p, 2))
        
        match_rows <- match(
          apply(selected.interaction, 1, paste, collapse = "-"),
          apply(pairs, 1, paste, collapse = "-")
        )
        
        interaction.freq = rep(0, (p**2 - p)/2)
        interaction.freq[match_rows] = 1
        
        # FPR, FNR, Precision, Recall, F1 score
        true_main = c(1, 2, 3, 4)
        sel_main = selected.main
        true_int = matrix(c(1, 2, 1, 3, 3, 4), ncol = 2, byrow = T)
        sel_int = selected.interaction
          
        tm_str <- if(length(true_main) > 0) as.character(true_main) else character(0)
        sm_str <- if(length(sel_main) > 0) as.character(sel_main) else character(0)
        
        process_interactions <- function(int_mat) {
          if (is.null(int_mat) || nrow(int_mat) == 0) return(character(0))
          
          # Sort each row so an interaction [2, 1] securely becomes [1, 2]
          # (Using apply with margin=1 sorts row by row)
          sorted_mat <- t(apply(int_mat, 1, sort))
          
          # Paste the sorted pairs together with an underscore
          paste(sorted_mat[, 1], sorted_mat[, 2], sep = "_")
        }
        ti_str <- process_interactions(true_int)
        si_str <- process_interactions(sel_int)
        
        true_all <- c(tm_str, ti_str)
        sel_all  <- c(sm_str, si_str)
        
        TP <- length(intersect(true_all, sel_all))
        FP <- length(setdiff(sel_all, true_all))
        FN <- length(setdiff(true_all, sel_all))
        
        # Total possible effects: p main effects + p(p-1)/2 interactions
        total_possible <- p + (p * (p - 1)) / 2
        
        # Total actual positives (P) and actual negatives (N)
        P_total <- length(true_all)
        N_total <- total_possible - P_total
        
        TN <- N_total - FP
        
        # ---------------------------------------------------------
        # 5. Calculate Metrics (with zero-division safeguards)
        # ---------------------------------------------------------
        Precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
        Recall    <- ifelse((TP + FN) > 0, TP / (TP + FN), 0) # TPR
        
        F1 <- 0
        if ((Precision + Recall) > 0) {
          F1 <- 2 * (Precision * Recall) / (Precision + Recall)
        }
        
        FPR <- ifelse(N_total > 0, FP / N_total, 0)
        FNR <- ifelse(P_total > 0, FN / P_total, 0)
        
        return(list(ISE=ISE, runtime=runtime, model.size=model.size, effect.freq=c(main.freq, interaction.freq),
                    FPR=FPR, FNR=FNR, Precision=Precision, Recall=Recall, F1=F1))
      }
      
      stopCluster(cl)
      
      
      ISE = sapply(result, function(x) x$ISE)
      FPR = sapply(result, function(x) x$FPR)
      FNR = sapply(result, function(x) x$FNR)
      Precision = sapply(result, function(x) x$Precision)
      Recall = sapply(result, function(x) x$Recall)
      F1 = sapply(result, function(x) x$F1)
      
      
      runtime = sapply(result, function(x) x$runtime)
      model.size = sapply(result, function(x) x$model.size)
      effect.freq = sapply(result, function(x) x$effect.freq)

      cat("Method: ", m, "\n", file = file_path, append = TRUE)
      cat('Ave-ISE: ', paste(round(mean(ISE),3), '(', round(sd(ISE)/sqrt(npar), 3), ')', sep = ''), "\n", file = file_path, append = TRUE)
      cat('Med-ISE: ', round(median(ISE), 3), "\n", file = file_path, append = TRUE)
      cat('Ave-Model size: ', paste(round(mean(model.size),3), '(', round(sd(model.size)/sqrt(npar), 3), ')', sep = ''), "\n", file = file_path, append = TRUE)
      cat('Runtime: ', paste(round(mean(runtime), 3), '(', round(sd(runtime)/sqrt(npar), 3), ')', sep = ''), "\n", file = file_path, append = TRUE)
      cat('-The frequency of effects -', "\n", file = paste(simulation_path, 'benchmarks_results.txt', sep = ''), append = TRUE)
      cat(paste(round(rowMeans(effect.freq), 3), ', ', sep = ''), "\n", file = paste(simulation_path, 'benchmarks_results.txt', sep = ''), append = TRUE)
      cat(' & FPR & FNR & Precision & Recall & F1', "\n", file = paste(simulation_path, 'benchmarks_results.txt', sep = ''), append = TRUE)
      cat(' & ', round(median(FPR), 3), ' & ', round(median(FNR), 3), ' & ', round(median(Precision), 3), ' & ', round(median(Recall), 3), ' & ', round(median(F1), 3),"\n", file = file_path, append = TRUE)
      cat('\n', file = file_path, append = TRUE)
    }
    
    cat("--------------------------------------- \n", file = paste(simulation_path, '/benchmarks_results.txt',sep = ''), append = TRUE)
    cat("\n", file = paste(simulation_path, '/benchmarks_results.txt',sep = ''), append = TRUE)
}
    
}
