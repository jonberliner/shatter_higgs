function [X Y] = preptestdata(USED,PTRAIN,PTEST)

%% LOAD DATA
% load in and format the data
Xall = loadMNISTImages('mnist/train-images-idx3-ubyte')';
nIn = size(Xall,2);
labs = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

%% Use only samples from categories with labels in USED
nUsed = length(USED);
nSam = length(find(ismember(labs,USED)));
Y = zeros(nSam,nUsed);
X = nan(nSam,nIn);
USED(USED==0) = 10; % so don't index at 0
i0 = 1;
for i = 1:nUsed
    used = find(labs==USED(i));
    i1 = i0+length(used)-1;
    Y(i0:i1,i) = 1;
    X(i0:i1,:) = Xall(used,:);
    i0 = i1+1;
end

assert(sum(isnan(X(:)))==0);


%% cut it
nTrain = floor(PTRAIN * nSam);
nTest = floor(PTEST * nSam);
pVal = 1-(PTRAIN+PTEST);
nVal = floor(pVal * nSam);
cut = split_data(X, Y, nTrain, nTest, nVal);

% split into batches
X.train = cut.X.train;
Y.train = cut.Y.train;
X.test = cut.X.test;
Y.test = cut.Y.test;
X.val = cut.X.val;
Y.val = cut.Y.val;
