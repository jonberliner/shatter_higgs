function csnet = convshattertrain(csnet,IN,TAR,lrate,mom,nEpoch)

[nPer nIn nBatch] = size(IN);

nPar = csnet.nPar;
[nInPer, nHid] = size(csnet.Wih);
[nHid0, nOut] = size(csnet.Who);
[nPer, nIn0, nBatch] = size(IN);
[nPer0, nOut0, nBatch0] = size(TAR);

assert(nPer==nPer0,'diff amounts per batch for samples and labels');
assert(nBatch==nBatch0,'diff amount of batches for samples and labels');
assert(nInPer*nPar==nIn0,'diff number input units in samples and weight matrix');
assert(nOut==nOut0,'diff number output units in targets and weight matrix');
assert(nHid==nHid0,'diff number hidden units between W.ih and W.ho');


sig = @(x) 1./(1+exp(-x)); % sigmoidal transfer function

figure(1); hold on% for plotting
eperr = nan(nEpoch,nBatch);

for e = 1:nEpoch
    for b = 1:nBatch

        in = squeeze(IN(:,:,b));
        tar = squeeze(TAR(:,:,b));

        %% FORWARD PASS
        % csnet input to hidden
        hid = zeros(nPer,nHid);
        for p = 1:nPar
            in0 = in(:, csnet.in(p,:));
            hid = hid + sig(in0*csnet.Wih);
        end
        hid = hid / nPar;
        % combine for hidden to out
        out = sig(hid*csnet.Who);

        % for plotting
        eperr(e,b) =  mean( sum( (tar-out).^2, 2) );


        %% BACKPROP
        % update ho
        dout = out.*(1-out) .* (tar-out); % nPerxnOut .x nPerxnOut .x nPerxnOut
        dWho = (hid' * dout) / nPer ; % nHidxnPer x nPerxnOut
        dWho = ((1-mom)*dWho + mom*csnet.pdWho);
        csnet.Who = csnet.Who + lrate * dWho;
        csnet.pdWho = dWho;

        % update ih
        dWih = zeros(nInPer,nHid);
        for p = 1:nPar

            in0 = in(:, csnet.in(p, :));

            dhid = hid.*(1-hid) .* (dout*csnet.Who'); % nPerxnHid .x nPerxnHid .x (nPerxnOut x nOutxnHid)
            
            dWih = dWih + (in0' * dhid) / nPer ; % nInxnPer x nPerxnHid
            dWih = ((1-mom)*dWih + mom*csnet.pdWih);
        end
        csnet.Wih = csnet.Wih + lrate * dWih/nPar;
        csnet.dWih = dWih;
    end
    plot(nanmean(eperr,2));
    drawnow;
end