# load the methods for comprison
library(foreach)
library(doSNOW)
library(caret)

# User should specify the paths
root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'
# The real datasets were stored in: root_path + 'real/datasets/'
data_path = paste(root_path, 'real/datasets/', sep = '')
# The splits were stored in: root_path + 'real/splits/'
split_path = paste(root_path, 'real/splits/', sep = '')
# The results are will be stored in the path "root_path/real/results", make sure the dictionary has been created.
result_path = paste(root_path, 'real/results/', sep = '')
isExists = dir.exists(result_path)
if(!isExists){dir.create(result_path)}


source(paste(root_path, 'benchmarks/mars.R', sep = ''))
source(paste(root_path, 'benchmarks/acosso.R', sep = ''))
source(paste(root_path, 'benchmarks/bssanova.R', sep = ''))
library(reticulate) # for loading .pkl files
pickle = import('pickle')
builtins = import('builtins')

data = read.csv(paste(data_path, "jfk.data", sep=''), header = TRUE, sep = ',')
data$plane_age =  2024 - data$year
splits = pickle$load(builtins$open(paste(split_path, 'jfk_splits.pkl', sep = ''), 'rb'))
data = na.omit(data)
features = c(
  'distance', 'sched_dep_time', 'sched_arr_time', # flight
  'temp', 'dewp', 'humid', 'pressure', 'wind_dir', 'wind_speed', 'precip', 'visib', # weather
  'plane_age', 'seats' # aircraft
  
)

X.scaled = apply(data[features], 2, function(x) {
  return((x - min(x)) / (max(x) - min(x)))})
y = data$arr_delay / 60
p = ncol(X.scaled)
pairs <- t(combn(p, 2))



# Do random splitting 50 times, then length(splits)=50
# Users can set a smaller value for npar when testing.
npar = length(splits) 
n.cores = 5

file_name = paste(root_path, 'real/jfk_baselines_results.txt', sep = '')
# open a .txt file
writeLines("#######################################", file_name)
cat(" jfk data: Interaction model with random splits for ", npar, 'repetitions.', '\n', file = file_name, append = TRUE)
cat("####################################### \n", file = file_name, append = TRUE)



for(m in c('MARS', 'COSSO', 'ACOSSO')) {
  
  cat('\n','jfk data: method = ', m, '\n')
  
  pb <- txtProgressBar(max = npar, style = 3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  # adapt number of kernels used
  cl = makeCluster(n.cores)
  registerDoSNOW(cl)
  
  results = foreach(i=1:npar, .options.snow = opts, .packages = c('earth','MASS','quadprog')) %dopar% {
    # load training and testing samples(indexes)
    train_idx = splits[i][[1]][[1]]
    test_idx = splits[i][[1]][[2]]
    X = X.scaled[train_idx+1, ]
    Xt = X.scaled[test_idx+1,]
    Y = y[train_idx+1]
    Yt = y[test_idx+1]
    
    N.test = length(Yt)
    N.train = length(Y)
    
    
    if(m == 'MARS'){
      model = MARS(X=X, y=Y, degree = 2, penalty = 3)
      Y.pred = MARS.predict(model, Xt)
      ISE = mean((Yt - Y.pred)**2)
      selected.main = MARS.selection(model)$main
      selected.interaction = MARS.selection(model)$interaction
      model.size = length(selected.main) + dim(selected.interaction)[1]
      curves = MARS.predict.component(model, Xt)
    }
    
    if(m == 'COSSO') {
      model = ACOSSO(X=X, y=Y, order = 2, wt.pow = 0, cv='gcv', linear = c(8, 9, 10, 11, 12, 13))
      Y.pred = ACOSSO.predict(model, Xt)
      ISE = mean((Yt - Y.pred)**2)
      selected.main = ACOSSO.selection(model)$main
      selected.interaction = ACOSSO.selection(model)$interaction
      model.size = length(selected.main)
      if(!is.null(selected.interaction)) {
        model.size = model.size + dim(selected.interaction)[1]
      }
      curves = ACOSSO.predict.component(model, Xt)
    }
    
    if(m == 'ACOSSO') {
      model = ACOSSO(X=X, y=Y, order = 2, cv='gcv', linear = c(8, 9, 10, 11, 12, 13))
      Y.pred = ACOSSO.predict(model, Xt)
      ISE = mean((Yt - Y.pred)**2)
      selected.main = ACOSSO.selection(model)$main
      selected.interaction = ACOSSO.selection(model)$interaction
      model.size = length(selected.main)
      if(!is.null(selected.interaction)) {
        model.size = model.size + dim(selected.interaction)[1]
      }
      curves = ACOSSO.predict.component(model, Xt)
    }
    if(m == 'BSSANOVA') {
      model = BSSANOVA(y=Y, x=X, linear = c(rep(F, 7), rep(T, 6)), main = T, twoway = T, runs = 3000, burn = 1000)
      Y.pred = BSSANOVA.predict(model, Xt)
      ISE = mean((Yt - Y.pred)**2)
      selected.main = BSSANOVA.selection(model)$main
      selected.interaction = BSSANOVA.selection(model)$interaction
      model.size = length(selected.main)
      if(!is.null(selected.interaction)) {model.size = model.size + dim(selected.interaction)[1]}
      curves = BSSANOVA.predict.component(model, Xt)
    }
    
    # evaluation
    main.freq = rep(0, p)
    main.freq[selected.main] = 1
    
    # Find matching row indices
    match_rows <- match(
      apply(selected.interaction, 1, paste, collapse = "-"),
      apply(pairs, 1, paste, collapse = "-")
    )
    
    interaction.freq = rep(0, (p**2 - p)/2)
    interaction.freq[match_rows] = 1
    
    # compute correlation: <f_j, f> / ||f||^2
    f = rowSums(curves)
    effect.corr = (t(curves) %*% f / sum(f*f))[, 1]
    sobol.var = apply(curves, 2, function(x) {mean((x-mean(x))**2)})
    sobol.indice = sobol.var/ sum(sobol.var)
    
    return(list(ISE=ISE, model.size=model.size,
                effect.freq=c(main.freq, interaction.freq), effect.corr=effect.corr, sobol.indice=sobol.indice))
  }
  stopCluster(cl)
  
  # transfer the results (consist of 100 lists) to 1 big list
  list.names = c('ISE', 'model.size', 'effect.freq', 'effect.corr', 'sobol.indice')
  df.res <- lapply(list.names, function(nm) {
    sapply(results, function(x) x[[nm]])
  })
  
  names(df.res) = list.names 
  save(df.res, file = paste(result_path, 'jfk_', m, '_', npar, 'splits_results.Rdata', sep = ''))
  # sumamry
  ISE = sapply(results, function(x) x$ISE)
  model.size = sapply(results, function(x) x$model.size)
  effect.freq = sapply(results, function(x) x$effect.freq)
  effect.corr = sapply(results, function(x) x$effect.corr)
  sobol.indice = sapply(results, function(x) x$sobol.indice)
  
  # write the results into the .txt file
  cat('---------------------------------------------------', "\n", file = file_name, append = TRUE)
  cat("Method: ", m, "\n", file = file_name, append = TRUE)
  cat('normalized: ',' & Ave.ISE: ', ' & Med.ISE', ' & Sd.ISE', "\n", file = file_name, append = TRUE)
  cat(' &', round(mean(ISE), 3), ' &', round(median(ISE), 3), ' &', round(sd(ISE)/sqrt(npar), 3), "\n", file = file_name, append = TRUE)
  cat('& Ave.Size', 'Med.Size', ' & Sd.Size', ' & Range', "\n", file = file_name, append = TRUE)
  cat(' &', round(mean(model.size), 3), ' &', round(median(model.size), 3), ' &', round(sd(model.size)/sqrt(npar), 3), ' & [', min(model.size), ',' , max(model.size), ']', "\n", file = file_name, append = TRUE)
  cat('-The frequency of effects -', "\n", file = file_name, append = TRUE)
  cat(paste(", ", round(rowMeans(effect.freq), 3)), "\n", file = file_name, append = TRUE)
  cat('-The cosine of effects -', "\n", file = file_name, append = TRUE)
  cat(paste(", ", round(rowMeans(effect.corr), 3)), "\n", file = file_name, append = TRUE)
  cat('-The Sobol indice of effects -', "\n", file = file_name, append = TRUE)
  cat(paste(", ", round(rowMeans(sobol.indice), 3)), "\n", file = file_name, append = TRUE)
  cat('---------------------------------------------------', "\n", file = file_name, append = TRUE)
  cat("\n", file = file_name, append = TRUE)
}
