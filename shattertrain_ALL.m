function [net vrmse trmse] = shattertrain(net, IN, TAR, nPer, lrate, dropout, stop_crit, varargin)

figure;

[nPar, nInPer, nHidPer] = size(net.Wih);
[nHid, nOut] = size(net.Who);
[nSam, nIn0] = size(IN);
[nSam0, nOut0] = size(TAR);

assert(nSam==nSam0,'diff amount of samples for samples and labels');
assert(nInPer*nPar==nIn0,'diff number input units in samples and weight matrix');
assert(nOut==nOut0,'diff number output units in targets and weight matrix');
assert(nHidPer*nPar==nHid,'diff number hidden units between W.ih and W.ho');


switch stop_crit
    case 'val_percent' % validation set hits a given percentage
        vIN = varargin{1};
        vTAR = varargin{2};
        p = varargin{3};
        
        % set nans to zero
        vIN = nan2zero(vIN);
    case 'train_percent'
        p = varargin{1};
    case 'epoch'
        nEpoch = varargin{1};
end




rmse = @(out,tar) sqrt( mean( (tar-out).^2, 2) );

nBatch = floor(nSam / nPer );


ei = 1; % epoch number
stop = false;
while ~stop
    [bIN, bTAR] = makebatches(IN,TAR,nPer,1);
    assert(isequal( size(bIN), [nPer,nIn0,nBatch] ));
    assert(isequal( size(bTAR), [nPer,nOut,nBatch] ));
    
    % set nan to zero
    bIN = nan2zero(bIN);
    
    % train net on one epoch
    if dropout
        net = trainbatchepoch(net,bIN,bTAR,lrate);
    else
        net = trainbatchepoch_dropout(net,bIN,bTAR,lrate);
    end
    
    % get training error
    tyhat = shatterpredict(net,nan2zero(IN));
    % get err
    trmse0 = rmse(tyhat,TAR);
    trmse(ei,:) = [mean(trmse0), std(trmse0) / sqrt(trmse0)];
    tepocherr = trmse(ei,1);
    
    switch stop_crit
        case 'val_percent'    
            % make guesses on validation set
            vyhat = shatterpredict(net,vIN);
            % get err
            vrmse0 = rmse(vyhat,vTAR);
            vrmse(ei,:) = [mean(vrmse0), std(vrmse0) / sqrt(vrmse0)];
            vepocherr = vrmse(ei,1);

            errorbar(vrmse(:,1),vrmse(:,2));
            
            if vepocherr <= p
                stop = true;
            end
            
        case 'train_percent'
            if tepocherr <= p
                stop = true;
            end
            
        case 'epoch'
            if ei == nEpoch
                stop = true;
            end
    end
    
    hold on
    errorbar(trmse(:,1),trmse(:,2),'r');
    hold off
    drawnow;
    
    ei = ei + 1;
end