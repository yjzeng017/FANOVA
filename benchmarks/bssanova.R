# BSSANOVA is the main function to do the MCMC sampling.  There
# is an example of its use at the end of this file.


lambda<-2	 #Half-Cauchy hyperparameter

BSSANOVA<-function(y,x,categorical=NA, linear=NA,
    runs=10000,burn=2000,update=100,
    ae=.01,be=.01,priorprob=0.5,nterms=25,lambda=2,
    main=T,twoway=F,rest=F,const=100){

#########  Definitions of the input    ################:
#  y is the n*1 vector of data
#  x is the n*p design matrix
#  categorical is the p-vector indicating (TRUE/FALSE) which
#       columns of x are categorical variables
#  linear is the p-vector indicating (TRUE/FALSE) which
#       columns of x are linear effects  
#  runs is the number of MCMC samples to generate
#  burn is the number of samples to discard as burnin
#  update is the number of interations between displays
#  error variance sigma^2~Gamma(ae,be)
#  priorprob is the prior includion probability for 
#       each functional component
#  nterms is the number of eigenvector to retain
#  lambda is the hyperparameters in the half-Cauchy prior
#  main indicates whether to include main effects
#  twoway incicates whether to include interactions
#  rest indicates whether to include higher-order terms (f_0)
#  const is the relative variance for the polynomial trend

 
  #set some sample size parameters 
  n<-length(y)
  ncurves<-0
  p<-ncol(x)
  if(is.na(mean(categorical))){categorical<-rep(F,p)}
  if(is.na(mean(linear))){linear<-rep(F,p)}
  
  if(main){ncurves<-ncurves+p}
  if(twoway){ncurves<-ncurves+p*(p-1)/2}
  if(rest){ncurves<-ncurves+1}
  term1<-term2<-rep(0,ncurves)
  if(p>0){o1<-order(x[,1])}
  if(p>1){o2<-order(x[,2])}
  if(p>2){o3<-order(x[,3])}
  if(p>3){o4<-order(x[,4])}

  #Set up the covariance matrices for the main effects
  Gamma<-array(0,c(ncurves,n,nterms))
  CovMat<-array(0,c(ncurves,n,n))
  D<-array(0,c(nterms,ncurves))
  count<-1
  if(rest){totC<-matrix(1,n,n)}

  #set up the covariances for the main effects
  if(main){for(j in 1:p){
     term1[count]<-term2[count]<-j
     if((!categorical[j]) & (!linear[j])){COV<-makeCOV.ME(xxx=x[,j],xxx.new = NULL, const=const)}
     if(categorical[j]){COV<-makeCOV.ME.cat(xxx=x[,j], xxx.new = NULL)}
     if(linear[j]){COV<-makeCOV.ME.lin(xxx=x[,j], xxx.new = NULL, const=const)}
     
     CovMat[count,,] <- COV
     COV<-COV/mean(diag(COV[1:n,1:n]))
     if(rest){totC<-COV+totC}
     eig<-eigen(COV);
     Gamma[count,,]<-eig$vectors[,1:nterms];
     D[,count]<-abs(1/eig$values[1:nterms])
     count<-count+1
  }}

  #set up the covariances for the two-way interactions
  if(twoway){for(j1 in 2:p){for(j2 in 1:(j1-1)){
     term1[count]<-j1;term2[count]<-j2
     COV<-makeCOV.INT(xxx1=x[,j1],xxx2=x[,j2],xxx1.new = NULL, xxx2.new = NULL,const=const,linear1=linear[j1], linear2=linear[j2])
     CovMat[count,,] <- COV
     COV<-COV/mean(diag(COV[1:n,1:n]))
     if(rest){totC<-COV+totC}
     eig<-eigen(COV);
     Gamma[count,,]<-eig$vectors[,1:nterms];
     D[,count]<-abs(1/eig$values[1:nterms])
    count<-count+1
  }}}

  #Set up the covariance matrices for the remainder part of the GP
  if(rest){
     term1[count]<-term2[count]<-1
     rCOV<-matrix(1,n,n)
     for(j in 1:p){
       if((!categorical[j]) & (!linear[j])){rCOV<-rCOV*(1+makeCOV.ME(xxx=x[,j],xxx.new = NULL, const=1))}
       if(categorical[j]){rCOV<-rCOV*(1+makeCOV.ME.cat(xxx=x[,j], xxx.new = NULL))}
       if(linear[j]){rCOV<-rCOV*(1+makeCOV.ME.lin(xxx=x[,j], xxx.new = NULL, const=1))}
       #if(linear[j])rCOV<-rCOV*(1+makeCOV.ME(xxx = x[,j],xxx.new = NULL,const=1))
       }
     COV<-rCOV-totC
     CovMat[count,,] <- COV
     COV<-COV/mean(diag(COV[1:n,1:n]))
     eig<-eigen(COV);
     Gamma[count,,]<-eig$vectors[,1:nterms];
     D[,count]<-abs(1/eig$values[1:nterms])
  }

  ########                 Initial values      ###################
  int<-mean(y)
  sige<-sd(y);taue<-1/sige^2
  curves<-matrix(0,n,ncurves)
  curfit<-int+sige*apply(curves,1,sum)
  r<-rep(0,ncurves)

  #keep track of the mean of the fitted values
  afterburn<-0
  sumfit<-sumfit2<-rep(0,n)
  suminout<-rep(0,ncurves)
  sumcurves<-sumcurves2<-matrix(0,n,ncurves)
  keepr<-keepl2<-matrix(0,runs,ncurves)
  dev<-keepsige<-keepint<-rep(0,runs)

  npts<-50
  if(priorprob==1){mxgx<-grid<-c(seq(0.0001,2,length=npts/2),seq(2,1000,length=npts/2))}
  if(priorprob<1){mxgx<-grid<-c(seq(0,2,length=npts/2),seq(2,1000,length=npts/2))}
 
  ########             Start the sampler       ###################
  countiter<-0
  start<-proc.time()[3]
  for(i in 1:runs){

   #new taue
    cantaue<-rnorm(1,taue,0.05*sd(y))
    if(cantaue>0){
      cansige<-1/sqrt(cantaue)
      MHrate<-sum(dnorm(y,int+cansige*apply(curves,1,sum),cansige,log=T))
      MHrate<-MHrate-sum(dnorm(y,int+sige*apply(curves,1,sum),sige,log=T))
      MHrate<-MHrate+dgamma(cantaue,ae,rate=be,log=T)-dgamma(taue,ae,rate=be,log=T) 
      if(runif(1,0,1)<exp(MHrate)){taue<-cantaue;sige<-cansige}
    }

   #new intercept
    int<-rnorm(1,mean(y-sige*apply(curves,1,sum)),sige/sqrt(n))

   #new curves
   for(j in 1:ncurves){
     ncp<-min(c(median(keepr[,j]),2))
     #first draw the sd:
      rrr<-y-int-sige*apply(curves[,-j],1,sum)
      z<-sqrt(taue)*t(Gamma[j,,])%*%rrr
      for(jjj in 1:npts){
         mxgx[jjj]<-g(grid[jjj],z,sige,D[,j],priorprob=priorprob)/
                    dcan(grid[jjj],ncp,priorprob=priorprob)
      } 
      highpt<-1.05*max(mxgx)
      ratio<-0
      while(ratio<1){
        r[j]<-rcan(1,ncp,priorprob)
        ratio<-g(r[j],z,sige,D[,j],priorprob=priorprob)/highpt/runif(1,0,1)/
               dcan(r[j],ncp,priorprob=priorprob)
       }
  
    #then draw the curve
      if(r[j]==0){curves[,j]<-0}
      if(r[j]>0){
        var<-1/(1+D[,j]/r[j])
        curves[,j]<-Gamma[j,,]%*%(rnorm(nterms,0,sqrt(var))+var*z)
      }
   }

   #Record results:
   keepr[i,]<-r  
   keepl2[i,]<-apply(curves^2,2,mean)
   fit<-int+sige*apply(curves,1,sum)
   dev[i]<- -2*sum(dnorm(y,fit,sige,log=T))
   keepsige[i]<-sige
   keepint[i]<-int

   if(i>burn){
      afterburn<-afterburn+1
      sumfit<-sumfit+fit
      sumfit2<-sumfit2+fit^2
      suminout<-suminout+ifelse(r>0,1,0)
      sumcurves<-sumcurves+sige*curves
      sumcurves2<-sumcurves2+(sige*curves)^2
    }

    #display current value of the chain
    # if(i%%update==0){ 
    #  par(mfrow=c(2,2))
    #  if(p>0){plot(x[o1,1],y[o1],main=i);
    #    lines(x[o1,1],int+sige*curves[o1,1],col=4)}    
    #  if(p>1){plot(x[o2,2],y[o2],main=i);
    #    lines(x[o2,2],int+sige*curves[o2,2],col=4)}    
    #  if(p>2){plot(x[o3,3],y[o3],main=i);
    #    lines(x[o3,3],int+sige*curves[o3,3],col=4)}    
    #  if(p>3){plot(x[o4,4],y[o4],main=i);
    #    lines(x[o4,4],int+sige*curves[o4,4],col=4)}    
    # }
  }
  stop<-proc.time()[3]
  print(paste("Sampling took",round((stop-start)/60, 3),"minutes."))

  #Calculate posterior means:
  fitmn<-sumfit/afterburn
  fitsd<-sqrt(sumfit2/afterburn-fitmn^2)
  curves<-sumcurves/afterburn
  curvessd<-sqrt(sumcurves2/afterburn-curves^2)
  probin<-suminout/afterburn


#########  Definitions of the output    ################:
# fittedvalues,fittedsds are the posterior means and sds 
#                        of f at the data points:
# inprob is the posterior inclusion probability
# l2 is the posterior distribution of the l2 norm of each component
# curves and curvessd are the posterior means and sds of the
#                     individual components f_{ij}
# r is the posterior distribution of the variance r
# term1 and term2 index the compnents f_{ij}.  For example, if
#     term1[j]=3 and term2[j]=4 then curves[,j] is the posterior 
#     mean of f_{3,4}.  Terms with terms1[j]=terms2[j] 
#     are main effects
# dev is the posterior samples of the deviance
# int is the posterior samples of the intercept
# sigma is the posterior samples of the error sd

list(x=x, y=y, main=main,twoway=twoway, rest=rest, categorical=categorical, linear=linear, const=const, fittedvalues=fitmn,fittedsds=fitsd,inprob=probin,l2=keepl2[burn:runs,],
curves=curves,curvessd=curvessd,r=keepr[burn:runs,],
term1=term1,term2=term2,dev=dev,int=keepint,sigma=keepsige,eigenmatix=Gamma, eigenvalue=D, covmat=CovMat)}


############### prediction and selection #############

BSSANOVA.predict = function(fit, x.new){
  # fit: output of BSSANOVA
  # x.new: new design
  
  x = fit$x
  y = fit$y
  p = ncol(fit$x)
  x.new = as.matrix(x.new)
  categorical = fit$categorical
  linear = fit$linear
  const = fit$const
  ncurves = dim(fit$curves)[2]
  n = dim(fit$curves)[1]
  
  # curves.sum = apply(fit$curves,1,sum)
  int = fit$int[length(fit$int)]
  r = fit$r[dim(fit$r)[1], ]
  sige = fit$sigma[length(fit$sigma)]
  Cov = sige**2 * diag(1, n, n)
  # Gamma = fit$eigenmatix
  # D = fit$eigenvalue
  Cov.Mat = fit$covmat
  for (j in 1:ncurves) {
    if(r[j]>0){
      Cov = Cov + (r[j]/mean(diag(Cov.Mat[j, , ])))*Cov.Mat[j, , ]
    }
  }
  invCov = solve(Cov)
  c.hat = invCov%*%(matrix(y - int))
  totC.new = matrix(1, dim(x)[1], dim(x.new)[1])
  Cov.new = matrix(0, dim(x)[1], dim(x.new)[1])
  
  main = fit$main
  twoway = fit$twoway
  rest = fit$rest
  if(main){for(j in 1:p){
    if((!categorical[j]) & (!linear[j])){Cov.tmp=makeCOV.ME(xxx=x[,j],xxx.new = x.new[,j], const=const)}
    if(categorical[j]){Cov.tmp=makeCOV.ME.cat(xxx=x[,j], xxx.new = x.new[,j])}
    if(linear[j]){Cov.tmp=makeCOV.ME.lin(xxx=x[,j], xxx.new = x.new[,j], const=const)}
    
    if(rest){totC.new = Cov.tmp+totC.new}
    Cov.new = Cov.new + (r[j]/mean(diag(Cov.Mat[j,,])))*Cov.tmp
  }}
  
  #set up the covariances for the two-way interactions
  if(twoway){for(j1 in 2:p){for(j2 in 1:(j1-1)){
    index = which(fit$term1 == j1 & fit$term2 == j2)
    Cov.tmp = makeCOV.INT(xxx1=x[,j1],xxx2=x[,j2],xxx1.new = x.new[, j1], xxx2.new = x.new[, j2],const=const,linear1=linear[j1], linear2=linear[j2])
    if(rest){totC.new=Cov.tmp+totC.new}
    if(r[index]>0){Cov.new = Cov.new + (r[index]/mean(diag(Cov.Mat[index,,])))*Cov.tmp}
  }}}
  
  if(rest){
    rCov.new = matrix(1,dim(x)[1],dim[x.new][1])
    for(j in 1:p){
      if((!categorical[j]) & (!linear[j])){rCOV.new=rCOV.new*(1+makeCOV.ME(xxx=x[,j],xxx.new=x.new[, j], const=1))}
      if(categorical[j]){rCOV.new=rCOV.new*(1+makeCOV.ME.cat(xxx=x[,j], xxx.new = x.new[, j]))}
      if(linear[j]){rCOV.new=rCOV.new*(1+makeCOV.ME.lin(xxx=x[,j], xxx.new = x.new[, j], const=1))}
      # rCOV.new=rCOV.new*(1+makeCOV.ME(xxx = x[,j],xxx.new = x.new[, j],const=1))
      }
    Cov.rest.new = (r[ncurves]/mean(diag(Cov.Mat[ncurves,,])))*(rCov.new - totC.new)
    Cov.new = Cov.new + Cov.rest.new
  }
  f.pred = int +  (t(Cov.new)%*%c.hat)[, 1]
  return(f.pred)
  
}

BSSANOVA.selection = function(fit){
  p = ncol(fit$x)
  main = fit$main
  twoway = fit$twoway
  r = fit$r[dim(fit$r)[1], ]
  select.main = c()
  select.int = NULL
  if(main){select.main = which(r[1:p] > 0.0)
  }
  if(twoway){
    select.int =c()
    for (j1 in 2:p) {
      for (j2 in 1:(j1-1)) {
        index = which(fit$term1 == j1 & fit$term2 == j2)
        if(r[index] > 0.0){
          select.int = c(select.int, c(j2, j1))
        }
      }
    }
    select.int = matrix(select.int, ncol = 2, byrow = T)
  }
  return(list(main=select.main, interaction=select.int))
}

BSSANOVA.predict.component = function(fit, x.new){
  # fit: output of BSSANOVA
  # x.new: new design
  
  categorical = fit$categorical
  linear = fit$linear
  const = fit$const
  ncurves = dim(fit$curves)[2]
  n = dim(fit$curves)[1]
  int = fit$int[length(fit$int)]
  r = fit$r[dim(fit$r)[1], ]
  sige = fit$sigma[length(fit$sigma)]
  Cov = sige**2 * diag(1, n, n)
  # Gamma = fit$eigenmatix
  # D = fit$eigenvalue
  Cov.Mat = fit$covmat
  for (j in 1:ncurves) {
    if(r[j]>0){
      Cov = Cov + (r[j]/mean(diag(Cov.Mat[j, , ])))*Cov.Mat[j, , ]
    }
  }
  #
  x = fit$x
  y = fit$y
  p = ncol(fit$x)
  x.new = as.matrix(x.new)
  invCov = solve(Cov)
  c.hat = invCov%*%(matrix(y - int))

  totC.new = matrix(1, dim(x)[1], dim(x.new)[1])
  Cov.new = matrix(0, dim(x)[1], dim(x.new)[1])
  
  curves = matrix(0, nrow = dim(x.new)[1], ncurves) # total curves: main + interaction + rest (if they exist)
  
  if(fit$main){
    for(j in 1:p){
      if(r[j] > 0) {
        if((!categorical[j]) & (!linear[j])){Cov.tmp=makeCOV.ME(xxx=x[,j],xxx.new = x.new[,j], const=const)}
        if(categorical[j]){Cov.tmp=makeCOV.ME.cat(xxx=x[,j], xxx.new = x.new[,j])}
        if(linear[j]){Cov.tmp=makeCOV.ME.lin(xxx=x[,j], xxx.new = x.new[,j], const=const)}
        if(fit$rest){totC.new = Cov.tmp+totC.new}
        
        Cov.new.tmp = (r[j]/mean(diag(Cov.Mat[j,,])))*Cov.tmp
        curves[, j] = (t(Cov.new.tmp)%*%c.hat)[, 1]
        Cov.new = Cov.new + Cov.new.tmp
      }
    }
    index = p
  }
  
  #set up the covariances for the two-way interactions
  if(fit$twoway){for(j1 in 2:p){for(j2 in 1:(j1-1)){
    index = which(fit$term1 == j1 & fit$term2 == j2)
    if(r[index]>0){
      Cov.tmp = makeCOV.INT(xxx1=x[,j1],xxx2=x[,j2],xxx1.new = x.new[, j1], xxx2.new = x.new[, j2],const=const,linear1=linear[j1], linear2=linear[j2])
      if(fit$rest){totC.new=Cov.tmp+totC.new}
      Cov.new.tmp = (r[index]/mean(diag(Cov.Mat[index,,])))*Cov.tmp
      curves[, index] = (t(Cov.new.tmp)%*%c.hat)[, 1]
      Cov.new = Cov.new + Cov.new.tmp
    }
  }}}
  
  if(fit$rest){
    rCov.new = matrix(1,dim(x)[1],dim[x.new][1])
    for(j in 1:p){
      if((!categorical[j]) & (!linear[j])){rCOV.new=rCOV.new*(1+makeCOV.ME(xxx=x[,j],xxx.new=x.new[, j], const=1))}
      if(categorical[j]){rCOV.new=rCOV.new*(1+makeCOV.ME.cat(xxx=x[,j], xxx.new = x.new[, j]))}
      if(linear[j]){rCOV.new=rCOV.new*(1+makeCOV.ME.lin(xxx=x[,j], xxx.new = x.new[, j], const=1))}
      # rCOV.new=rCOV.new*(1+makeCOV.ME(xxx = x[,j],xxx.new = x.new[, j],const=1))
    }
    Cov.rest.new = (r[ncurves]/mean(diag(Cov.Mat[ncurves,,])))*(rCov.new - totC.new)
    curves[, index+1] = (t(Cov.rest.new)%*%c.hat)[, 1]
    Cov.new = Cov.new + Cov.rest.new
  }
  
  # rearrange the curves
  if(fit$twoway) {
    orig_pairs = do.call(rbind, lapply(2:p, function(j1) cbind(j1, 1:(j1-1))))
    target_pairs  = t(combn(p, 2))
    map_idx = match(
      apply(target_pairs, 1, paste, collapse = ","),
      apply(orig_pairs, 1, function(x) paste(sort(x), collapse = ","))
    )
    perm = map_idx
    if(fit$main) {perm = c(1:p, p + map_idx)}
    if(fit$rest) {perm = c(perm, length(perm)+1)}
    curves = curves[, perm]
  }
  return(curves)
}


##############  subfunctions #############################
priorr<-function(r,priorprob=0.5){ifelse(r==0,1-priorprob,priorprob*2*dt(sqrt(r)/lambda,1)/lambda)}

rcan<-function(n,ncp,priorprob=0.5){
  if(priorprob==1){rrr<-abs(rt(n,1,ncp=ncp))}
  if(priorprob<1){rrr<-ifelse(runif(n,0,1)<0.975,abs(rt(n,1,ncp=ncp)),0)}
rrr}

dcan<-function(r,ncp,priorprob=0.5){
  if(priorprob==1){rrr<-abs(rt(n,1,ncp=ncp))}
  if(priorprob<1){rrr<-ifelse(r==0,1-0.975,0.975*2*dt(r,1,ncp=ncp))}
rrr}
g<-function(r,z,sige,d,priorprob=.5){prod(dnorm(z,0,sqrt(1+r/d)))*priorr(r,priorprob)}

#Define Bernoulli polynomials
B0<-function(x){1+0*x}
B1<-function(x){x-.5}
B2<-function(x){x^2-x+1/6}
B3<-function(x){x^3-1.5*x^2+.5*x}
B4<-function(x){x^4-2*x^3+x^2-1/30}
B5<-function(x){x^5-2.5*x^4+1.667*x^3-x/6}
B6<-function(x){x^6-3*x^5+2.5*x^4-.5*x^2+1/42}

makeCOV.ME<-function(xxx,xxx.new=NULL, const=10){
  if(is.null(xxx.new)){
    xxx.new = xxx
    # sss<-matrix(xxx,length(xxx),length(xxx),byrow=T)
    # ttt<-matrix(xxx.new,length(xxx.new),length(xxx.new),byrow=F)
    # diff<-as.matrix(dist(xxx,diag=T,upper=T))
  }
  sss<-matrix(xxx,length(xxx),length(xxx.new),byrow=F)
  ttt<-matrix(xxx.new,length(xxx),length(xxx.new),byrow=T)
  diff<-abs(sss - ttt)
  xxx = matrix(xxx, length(xxx), 1)
  xxx.new = matrix(xxx.new, length(xxx.new), 1)
  const*(B1(xxx)%*%t(B1(xxx.new))+B2(xxx)%*%t(B2(xxx.new))/4) - B4(diff)/24
}
        
    
makeCOV.ME.cat<-function(xxx, xxx.new=NULL){
  if(is.null(xxx.new)){xxx.new=xxx}
  g<-length(unique(xxx))
  sss<-matrix(xxx,length(xxx),length(xxx.new),byrow=F)
  ttt<-matrix(xxx.new,length(xxx),length(xxx.new),byrow=T)

  #sss<-matrix(xxx,length(xxx),length(xxx),byrow=T)
  #ttt<-matrix(xxx,length(xxx),length(xxx),byrow=F)
  equals<-ifelse(sss==ttt,1,0)
(g-1)*equals/g -(1-equals)/g}      

makeCOV.ME.lin<-function(xxx, xxx.new=NULL, const=10){
  if(is.null(xxx.new)){xxx.new=xxx}
  # sss<-matrix(xxx,length(xxx),length(xxx.new),byrow=F)
  # ttt<-matrix(xxx.new,length(xxx),length(xxx.new),byrow=T)
  # diff<-abs(sss - ttt)
  xxx = matrix(xxx, length(xxx), 1)
  xxx.new = matrix(xxx.new, length(xxx.new), 1)
  const*(B1(xxx)%*%t(B1(xxx.new)))
}

makeCOV.INT<-function(xxx1,xxx2, xxx1.new=NULL, xxx2.new=NULL, const=10, linear1=FALSE, linear2=FALSE){
  if(is.null(xxx1.new)&(!is.null(xxx2.new))){
    print("Both of dim 1 and dim 2 should be specfied.")
  }
  if((!is.null(xxx1.new)) & is.null(xxx2.new)){
    print("Both of dim 1 and dim 2 should be specfied.")
  }
  if(is.null(xxx1.new) & is.null(xxx2.new)){xxx1.new = xxx1;xxx2.new=xxx2}
  
  sss1<-matrix(xxx1,length(xxx1),length(xxx1.new),byrow=F)
  ttt1<-matrix(xxx1.new,length(xxx1),length(xxx1.new),byrow=T)
  diff1<-abs(sss1 - ttt1)
  xxx1 = matrix(xxx1, length(xxx1), 1)
  xxx1.new = matrix(xxx1.new, length(xxx1.new), 1)
  KP1<-B1(xxx1)%*%t(B1(xxx1.new))+B2(xxx1)%*%t(B2(xxx1.new))/4
  KN1<- -B4(diff1)/24    
  
  #sss1<-matrix(xxx1,length(xxx1),length(xxx1),byrow=T)
  #ttt1<-matrix(xxx1,length(xxx1),length(xxx1),byrow=F)
  #diff1<-as.matrix(dist(xxx1,diag=T,upper=T))
  #KP1<-B1(sss1)*B1(ttt1)+B2(sss1)*B2(ttt1)/4
  #KN1<- -B4(diff1)/24      
  if(length(unique(xxx1))<10){KP1<-0*KP1;KN1<-makeCOV.ME.cat(xxx1,xxx1.new)}
  if(linear1){KN1 <- B1(xxx1)%*%t(B1(xxx1.new)); KP1 <- 0*KP1}
  
  sss2<-matrix(xxx2,length(xxx2),length(xxx2.new),byrow=F)
  ttt2<-matrix(xxx2.new,length(xxx2),length(xxx2.new),byrow=T)
  diff2<-abs(sss2 - ttt2)
  xxx2 = matrix(xxx2, length(xxx2), 1)
  xxx2.new = matrix(xxx2.new, length(xxx2.new), 1)
  KP2<-B1(xxx2)%*%t(B1(xxx2.new))+B2(xxx2)%*%t(B2(xxx2.new))/4
  KN2<- -B4(diff2)/24    
  
  # sss2<-matrix(xxx2,length(xxx2),length(xxx2),byrow=T)
  # ttt2<-matrix(xxx2,length(xxx2),length(xxx2),byrow=F)
  # diff2<-as.matrix(dist(xxx2,diag=T,upper=T))
  # KP2<-B1(sss2)*B1(ttt2)+B2(sss2)*B2(ttt2)/4
  # KN2<- -B4(diff2)/24      
  if(length(unique(xxx2))<10){KP2<-0*KP2;KN2<-makeCOV.ME.cat(xxx2, xxx2.new)}
  if(linear2){KN2 <- B1(xxx2)%*%t(B1(xxx2.new)); KP2 <- 0*KP2}
  
(KP1+KN1)*(KP2+KN2) + (const-1)*(KP1*KP2)}

## example#
# g1 <- function(x){
#   return(x)
# }
# 
# g2 <- function(x){
#   return(2*x - 1)^2
# }
# 
# g3 <- function(x){
#   return(sin(2*pi*x)/(2 - sin(2*pi*x)))
# }
# 
# g4 <- function(x){
#   return(0.1*sin(2*pi*x) + 0.2*cos(2*pi*x) + 0.3*sin(2*pi*x)^2 + 0.4*cos(2*pi*x)^3 + 0.5*sin(2*pi*x)^3)
# }
# #
# #
# nc = 6
# X = matrix(runif(100*nc, 0, 1), nc=nc)
# # X=cbind(rbinom(100, 1,.7), X)
# 
# y= 5*g1(X[, 1]) + 3*g2(X[, 2]) + 4*g3(X[, 3]) + 6*g4(X[, 4]) + sqrt(3.03) * rnorm(100)
# 
# fit = BSSANOVA(y=y, x=X, twoway = T, runs = 2000, burn = 1000)
# 
# Xnew = matrix(runif(1000*nc, 0, 1), nc=nc)
# # Xnew = cbind(rbinom(1000, 1,.7), Xnew)
# 
# Ynew = 5*g1(Xnew[, 1]) + 3*g2(Xnew[, 2]) + 4*g3(Xnew[, 3]) + 6*g4(Xnew[, 4])
# 
# y_pred = BSSANOVA.predict(fit, Xnew)

# mean((y_pred - Ynew)^2)
# 
# 
