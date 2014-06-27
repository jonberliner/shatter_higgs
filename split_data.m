function [out] = split_data(X,Y,nTrain,nTest,nVal)
% function [out] = split_data(X,Y,nTrain,nTest) takes a nxm data set X
% w 1xn labels Y and randomly partitions the set into train, test, and
% validation sets (nValid = nTotal - (nTrain+nTest))

% Jon Berliner 3.6.14 & Jordan Ash 4.27.14 <333
% changed valid to val 6.9.14

assert(ismatrix(X),'X must be a nxm matrix');
assert(size(X,1)==size(Y,1),'X and Y must have same number of rows');
nSam = size(X,1); % number of samples
assert(nSam >= nTrain+nTest+nVal,'nTrain+nTest+nVal must be <= number of rows in X');

inds = randperm(nSam); % shuffle indices

% get inds to of rows of X to be sorted into train/test/val sets
train = inds(1:nTrain);
test = inds(nTrain+1:nTrain+nTest);
val = inds(nTrain+nTest+1:end);

% split samples X
out.X.train = X(train,:);
out.X.test = X(test,:);
out.X.val = X(val,:);

% split labels Y
out.Y.train = Y(train,:);
out.Y.test = Y(test,:);
out.Y.val = Y(val,:);

% store original inds in dataset
out.inds.train = train;
out.inds.test = test;
out.inds.val = val;