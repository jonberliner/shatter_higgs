function [net vrmse trmse] = shattertrain_v_DROPOUT(net, IN, TAR, vIN, vTAR, stopwhen, nPer, lrate, plot_huh)


[nPar, nInPer, nHidPer] = size(net.Wih);
[nHid, nOut] = size(net.Who);
[nSam, nIn0] = size(IN);
[nSam0, nOut0] = size(TAR);

assert(nSam==nSam0,'diff amount of samples for samples and labels');
assert(nInPer*nPar==nIn0,'diff number input units in samples and weight matrix');
assert(nOut==nOut0,'diff number output units in targets and weight matrix');
assert(nHidPer*nPar==nHid,'diff number hidden units between W.ih and W.ho');


%% SET NAN TO ZERO
vIN = nan2zero(vIN);

rmse = @(out,tar) sqrt( mean( (tar-out).^2, 2) );

nBatch = floor(nSam / nPer );

vepocherr = 1;
tepocherr = 1;
ei = 1; % epoch number
while vepocherr > stopwhen
    % reshuffle into minibatches every epoch
    [bIN, bTAR] = makebatches(IN,TAR,nPer,1);
    assert(isequal( size(bIN), [nPer,nIn0,nBatch] ));
    assert(isequal( size(bTAR), [nPer,nOut,nBatch] ));
    
    
    %% SET NAN TO ZERO
    bIN = nan2zero(bIN);
    
    
    % train net on one epoch
    net = trainbatchepoch_DROPOUT(net,bIN,bTAR,lrate);
    
    
    % make guesses on validation set
    vyhat = shatterpredict(net,vIN);
    % get err
    vrmse0 = rmse(vyhat,vTAR);
    vrmse(ei,:) = [mean(vrmse0), std(vrmse0)];
    vepocherr = vrmse(ei,1);
    
    % same for training set
    tyhat = shatterpredict(net,nan2zero(IN));
    % get err
    trmse0 = rmse(tyhat,TAR);
    trmse(ei,:) = [mean(trmse0), std(trmse0)];
    tepocherr = trmse(ei,1);
    
    if plot_huh
        errorbar(vrmse(:,1),vrmse(:,2));
        hold on;
        errorbar(trmse(:,1),trmse(:,2),'r');
        hold off;
        drawnow;
    end
    
    disp(['train error: ' num2str(tepocherr)])
    disp(['val error: ' num2str(vepocherr)])
    
    ei = ei+1;
end