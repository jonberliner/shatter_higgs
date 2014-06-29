PTRAIN = 0.80;
PTEST = 0.10;
PVAL = 1 - (PTRAIN+PTEST);

NPAR = 2;
NHID = NPAR*50;

LRATE = 0.05;

NPER = 1;
STOPWHEN = 0.05;

PLOT_HUH = 1;

USED = [2, 5];

STOP_CRIT = 'epoch';

% STOP_P = 0.05;
NEPOCH = 5;

DROPOUT = 1;

% [Xtest Ytest] = preptestdata(USED,PTRAIN,PTEST);
% 
% [nPer nIn ~] = size(Xtest.train);
% [nPer0 nOut ~] = size(Ytest.train);


snet = shatterinit(nIn,NHID,nOut,NPAR);
snet = shattertrain_ALL(snet, Xtest.train, Ytest.train,...
                        NPER, LRATE, DROPOUT, STOP_CRIT,...
                        NEPOCH);
