function out = convshatterpredict(csnet,IN)


    assert(ismatrix(IN),'input must be in matrix form for predict');

    nPar = size(csnet.in,1);
    [nInPer, nHid] = size(csnet.Wih);
    [nHid0, nOut] = size(csnet.Who);
    [nSam, nIn] = size(IN);

    assert(nInPer*nPar==nIn,'diff number input units in samples and weight matrix');
    assert(nHid==nHid0,'diff number hidden units between W.ih and W.ho');

    sig = @(x) 1./(1+exp(-x)); % sigmoidal transfer function

    out = nan(nSam,nOut);
    %% FORWARD PASS
    % csnet input to hidden
    hid = zeros(nSam,nHid);
    for p = 1:nPar
        in0 = IN(:, csnet.in(p,:));
        hid = hid + sig(in0*csnet.Wih);
    end
    hid = hid / nPar;
    % combine for hidden to out
    out(:,:) = sig(hid*csnet.Who);
    
end