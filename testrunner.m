PTRAIN = 0.80;
PTEST = 0.10;
PVAL = 1 - (PTRAIN+PTEST);

NPAR = 7;
NHID = NPAR*50;

LRATE = 0.01;
%NEPOCH = 400;
NPER = 1;
STOPWHEN = 0.05;

PLOT_HUH = 1;

USED = [2, 5];

[Xtest Ytest] = preptestdata(USED,PTRAIN,PTEST);

[nPer nIn ~] = size(Xtest.train);
[nPer0 nOut ~] = size(Ytest.train);



snet = shatterinit(nIn,NHID,nOut,NPAR);
snet = shattertrain_v(snet,Xtest.train,Ytest.train,Xtest.val,Ytest.val,STOPWHEN,NPER,LRATE,PLOT_HUH);