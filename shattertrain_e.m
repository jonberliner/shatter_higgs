function [net, trmse] = shattertrain_e(net, IN, TAR, nEpoch, nPer, lrate, plot_huh)


[nPar, nInPer, nHidPer] = size(net.Wih);
[nHid, nOut] = size(net.Who);
[nIn0, nSam] = size(IN);
[nOut0, nSam0] = size(TAR);

assert(nSam==nSam0,'diff amount of samples for samples and labels');
assert(nInPer*nPar==nIn0,'diff number input units in samples and weight matrix');
assert(nOut==nOut0,'diff number output units in targets and weight matrix');
assert(nHidPer*nPar==nHid,'diff number hidden units between W.ih and W.ho');




sig = @(x) 1./(1+exp(-x)); % sigmoidal transfer function
rmse = @(out,tar) sqrt( mean( (tar-out).^2, 2) );



trmse0 = 1;
trmse = nan(nEpoch,2); % col1=mu,col2=sd
for ei = 1:nEpoch
    % reshuffle into minibatches every epoch
    [bIN, bTAR] = makebatches(IN,TAR,nPer,1);
    assert(isequal( size(bIN), [nPer,nIn,nBatch] ));
    assert(isequal( size(bTAR), [nPer,nOut,nBatch] ));
    
    % train net on one epoch
    net = trainbatchepoch(net,bIN,bTAR,nPer,lrate);
    
    
    % get training error
    tyhat = shatterpredict(net,IN);
    % get err
    trmse0 = rmse(tyhat,vTAR);
    trmse(ei,:) = [mean(trmse0), std(trmse0)];
    
    if plot_huh
        errorbar(trmse(:,1),trmse(:,2));
    end
    
end