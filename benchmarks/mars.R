require(earth)

MARS = function(X, y, degree=1, penalty=3){
  mars.model = earth(x=X, y=y, degree = degree, penalty = penalty)
  mars.model$p = ncol(X)
  mars.model$degree = degree
  return(mars.model)
}


MARS.predict = function(mars.model, Xnew){
  return(predict(mars.model, Xnew)[, 1])
}

MARS.selection = function(mars.model){
  p = mars.model$p  
  if(mars.model$degree == 1){
    col.all.zero = apply(mars.model$dirs, 2, function(col) all(col == 0))
    select.main = which(!col.all.zero)
    select.int = NULL
  }
  if(mars.model$degree > 1){
    dirs = as.data.frame((mars.model$dirs != 0) * 1)
    row.sum = apply(dirs, 1, sum)
    dirs.main = dirs[row.sum == 1, ]
    select.main = which(apply(dirs.main, 2, sum) > 0)
    select.main = select.main[order(select.main, decreasing = F)]
    
    dirs.int = dirs[row.sum > 1, ]
    select.int = unique(t(apply(dirs.int, 1, function(row) which(row == 1))))
    select.int = select.int[order(select.int[, 1], select.int[, 2]), ]
    select.int = matrix(select.int, ncol = 2, byrow = T)
  }
  return(list(main=select.main, interaction=select.int))
}

MARS.predict.component = function(mars.model, Xnew){
  pred.m = model.matrix(mars.model, x=Xnew)
  mars.coef = mars.model$coefficients
  n = dim(pred.m)[1]
  d = dim(pred.m)[2]
  pred.m = matrix(rep(mars.coef, n), ncol = d, nrow=n, byrow = TRUE) * pred.m
  pred.m.names = colnames(pred.m)
  
  p = mars.model$p
  dirs = as.data.frame((mars.model$dirs != 0) * 1)
  row.sum = apply(dirs, 1, sum)
  main.names = rownames(dirs)[which(row.sum == 1)]
  dirs.main = dirs[which(row.sum == 1), ]
    
  ncurves = matrix(0, nrow = n, ncol = p)
  for (i in 1:p) {
    for (j in 1:dim(dirs.main)[1]) {
      if(dirs.main[j, i] != 0) {
        if(main.names[j] %in% pred.m.names) {
          idx = which(pred.m.names == main.names[j])
          ncurves[, i] = ncurves[, i] + pred.m[, idx]
        }
        }
      }
    }
  
  if(mars.model$degree == 2) {
    pairs <- t(combn(p, 2))
    ncurves = cbind(ncurves, matrix(0, nrow = n, ncol = dim(pairs)[1]))
    int.names = rownames(dirs)[which(row.sum > 1)]
    dirs.int = dirs[which(row.sum > 1), ]
    for (i in 1:dim(dirs.int)[1]) {
      row.idx = which(apply(pairs, 1, function(r) all(r == which(dirs.int[i, ]!=0))))
      if(int.names[i] %in% pred.m.names) {
        idx = which(pred.m.names == int.names[i])
        ncurves[, p+row.idx] = ncurves[, p+row.idx] + pred.m[, idx]
      }
    }
  }
  return(ncurves)
}
