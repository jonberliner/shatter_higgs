function [snet, out] = shattertrain(snet,IN,TAR,vIN,vTAR,nEpoch,nPer,lrate,plot_huh)


[nPar, nInPer, nHidPer] = size(snet.Wih);
[nHid, nOut] = size(snet.Who);
[nIn0, nSam] = size(IN);
[nOut0, nSam0] = size(TAR);



assert(nSam==nSam0,'diff amount of samples for samples and labels');
assert(nInPer*nPar==nIn0,'diff number input units in samples and weight matrix');
assert(nOut==nOut0,'diff number output units in targets and weight matrix');
assert(nHidPer*nPar==nHid,'diff number hidden units between W.ih and W.ho');

assert( xor( ((isempty(vIN) && isempty(vTAR))),isempty(nEpoch) ),...
    'should have either nEpoch specified or a validation set given.  cannot have neither or both due to ambiguity');

if ~isempty( vIN )
    crit = 'asymp';
else
    crit = 'epoch';
end

sig = @(x) 1./(1+exp(-x)); % sigmoidal transfer function


        % for plotting
        eperr(e,b) =  mean( sum( (tar-out).^2, 2) );

nBatch = floor( nSam / nPer );

if plot_huh
    figure; % for plotting
end
% train error over each epoch
eperr = nan(nEpoch,nBatch);

hid = nan(nPer,nHid);
for e = 1:nEpoch
    % reshuffle into minibatches every epoch
    [bIN, bTAR] = makebatches(IN,TAR,nPer,1);
    assert(isequal( size(bIN), [nPer,nIn,nBatch] ));
    assert(isequal( size(bTAR), [nPer,nOut,nBatch] ));
    
    
    
    
end