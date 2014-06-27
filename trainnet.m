function [Wih Who] = trainnet(Wih,Who,IN,TAR,lrate,nEpoch)
% function [Wih Who] = trainnet(Wih,Who,IN,TAR,lrate,nEpoch)
% Wih : in to hid weights
% Who : hid to out weights
% IN : nPer x nIn x nBatch matrix of training samples cut into batches
%   made with makebatches
%  IN : nPer x nOut x nBatch matrix of training targets cut into batches
%   made with makebatches
%


[nIn,nHid] = size(Wih);
[nHid0,nOut] = size(Who);
[nPer,nIn0,nBatch] = size(IN);
[nPer0,nOut0,nBatch0] = size(TAR);

assert(nPer==nPer0,'diff amounts per batch for samples and labels');
assert(nBatch==nBatch0,'diff amount of batches for samples and labels');
assert(nIn==nIn0,'diff number input units in samples and weight matrix');
assert(nOut==nOut0,'diff number output units in targets and weight matrix');
assert(nHid==nHid0,'diff number hidden units between W.ih and W.ho');

sig = @(x) 1./(1+exp(-x)); % sigmoidal transfer function

figure; % for plotting
i = 1; % for plotting
eperr = nan(nEpoch,nBatch);

for e = 1:nEpoch
    
    bshuf = randperm(nBatch);
    
    for b = bshuf

        in = squeeze(IN(:,:,b));
        tar = squeeze(TAR(:,:,b));

        hid = sig(in * Wih);
        out = sig(hid * Who);

        eperr(e,b) = norm(tar-out) / nPer;
%         plot(i,err,'o');

        % bp
        % update ho
        dout = out.*(1-out) .* (tar-out); % nPerxnOut .x nPerxnOut .x nPerxnOut
        dWho = (hid' * dout) / nPer ; % nHidxnPer x nPerxnOut
        Who = Who + lrate * dWho;
        % update ih
        dhid = hid.*(1-hid) .* (dout*Who'); % nPerxnHid .x nPerxnHid .x (nPerxnOut x nOutxnHid)
        dWih = (in' * dhid) / nPer ; % nInxnPer x nPerxnHid
        Wih = Wih + lrate * dWih;
        
        i=i+1;
    end
    plot(nanmean(eperr,2));
    drawnow;
end