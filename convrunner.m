NPER = 10000;
PTRAIN = 0.75;
PTEST = 0.20;
PVAL = 1 - (PTRAIN+PTEST);
MOM = 0.5;

NPAR = 8;
NHID = 8*100;

LRATE = @(t) 1/(t+1);
NEPOCH = 100;

SHUF_HUH = 1; % shuffle input data

[X Y] = prepdata(NPER, PTRAIN, PTEST, SHUF_HUH);

[nPer nIn ~] = size(X.train);
[nPer0 nOut ~] = size(Y.train);


%csnet = convshatterinit(nIn,NHID,nOut,NPAR);
%csnet = convshattertrain(csnet,X.train,Y.train,LRATE,MOM,NEPOCH);
convpreds = convshatterpredict(csnet,X.test);
convconferr = rmse(convpreds,Y.test);
[toss, convimax] = max(convpreds,[],2);
convcatguess = zeros(size(Y.test));
convcaterr = size(nTest,1);
for i=1:nTest
    convcatguess(i,convimax(i)) = 1;
    convcaterr(i) = isequal(convcatguess(i,:),Y.test(i,:));
end
convpcaterr = 1 - (sum(convcaterr) / nTest)





%% TRAIN SHATTER NET
snet = shatterinit(nIn,NHID,nOut,NPAR);
snet = shattertrain(snet,X.train,Y.train,LRATE,NEPOCH);

%% GET PREDICTIONS
preds = shatterpredict(snet,X.test);

%% GET ERROR RATES
conferr = rmse(preds,Y.test);
[toss, imax] = max(preds,[],2);
nTest = size(Y.test,1);
catguess = zeros(size(Y.test));
caterr = size(nTest,1);
for i=1:nTest
    catguess(i,imax(i)) = 1;
    caterr(i) = isequal(catguess(i,:),Y.test(i,:));
end

pcaterr = 1 - (sum(caterr) / nTest)
