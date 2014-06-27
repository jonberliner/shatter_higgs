PTRAIN = 0.80;
PTEST = 0.10;
PVAL = 1 - (PTRAIN+PTEST);

NPAR = 3;
NHID = NPAR*300;

LRATE = 0.01;
%NEPOCH = 400;
NPER = 1;
STOPWHEN = 0.05;

PLOT_HUH = 1;


[X Y] = prepdata(PTRAIN,PTEST,1); % one means shuffle



[nTrain nIn] = size(X.train);
[nTrain0 nOut] = size(Y.train);


snet = shatterinit(nIn,NHID,nOut,NPAR);
snet = shattertrain_v(snet,X.train,Y.train,X.val,Y.val,STOPWHEN,NPER,LRATE,PLOT_HUH);