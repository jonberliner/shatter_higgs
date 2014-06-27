function [net vrmse trmse] = shattertrain_v(net,IN,TAR,vIN,vTAR,stopwhen,nPer,lrate,plot_huh)


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



vrmse0 = 1;
ei = 1; % epoch number
while vrmse0 > stopwhen
    % reshuffle into minibatches every epoch
    [bIN, bTAR] = makebatches(IN,TAR,nPer,1);
    assert(isequal( size(bIN), [nPer,nIn,nBatch] ));
    assert(isequal( size(bTAR), [nPer,nOut,nBatch] ));
    
    % train net on one epoch
    net = trainbatchepoch(net,bIN,bTAR,nPer,lrate);
    
    % make guesses on validation set
    vyhat = shatterpredict(net,vIN);
    % get err
    vrmse0 = rmse(vyhat,vTAR);
    verr(ei,:) = [mean(vrmse0), std(vrmse0)];
    
    % same for training set
    tyhat = shatterpredict(net,IN);
    % get err
    trmse0 = rmse(ytar,vTAR);
    terr(ei,:) = [mean(trmse0), std(trmse0)];
    
    if plot_huh
        errorbar(verr(:,1),verr(:,2));
        hold on;
        errorbar(terr(:,1),terr(:,2));
        hold off;
    end
    
    ei = ei+1;
end