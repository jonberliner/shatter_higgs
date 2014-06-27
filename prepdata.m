function [X Y] = prepdata(PTRAIN,PTEST,shuf_huh)

    if nargin<3 || isempty(shuf_huh)
        shuf_huh = 1; % default shuffle
    end


    %% LOAD DATA
    % load in and format the data
    load ./higgsdata/Xraw.mat
    % zscore errything
    Xraw.train(Xraw.train==-999) = NaN;
    Xraw.test(Xraw.test==-999) = NaN;
    Xflat = [Xraw.train;Xraw.test];
    Xzflat = zscore(Xflat,1);
    X.train = Xzflat(1:250000,:);
    X.test = Xzflat(250001:end,:);
    
    % turn strings into 01 categorization
    load ./higgsdata/Yraw.mat
    Y.train = cellfun(@(r) r=='b',Yraw.train); % s=0,b=1

    % separated from the true testing set, where we don't have labels
    X = X.train;
    Y = Y.train;
    
    [nSam nIn] = size(X);
    [nSam0 nOut] = size(Y);

    assert(nSam==nSam0,'num sams and tars not equal');

    %% cut it
    nTrain = floor(PTRAIN * nSam);
    nTest = floor(PTEST * nSam);
    pVal = 1-(PTRAIN+PTEST);
    nVal = floor(pVal * nSam);
    cut = split_data(X, Y, nTrain, nTest, nVal);

    % cut into train test val
    X.train = cut.X.train;
    Y.train = cut.Y.train;
    X.test = cut.X.test;
    Y.test = cut.Y.test;
    X.val = cut.X.val;
    Y.val = cut.Y.val;

end