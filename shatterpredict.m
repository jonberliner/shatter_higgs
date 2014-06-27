function out = shatterpredict(snet,IN)

    assert(ismatrix(IN),'input must be in matrix form for predict');

    [nPar, nInPer0] = size(snet.in);
    [nPar, nInPer, nHidPer] = size(snet.Wih);
    [nHid, nOut] = size(snet.Who);
    [nSam, nIn] = size(IN);

    assert(nInPer*nPar==nIn,'diff number input units in samples and weight matrix');
    assert(nHidPer*nPar==nHid,'diff number hidden units between W.ih and W.ho');
    
    
    sig = @(x) 1./(1+exp(-x)); % sigmoidal transfer function

    hid = nan(nSam,nHid);

    out = nan(nSam,nOut);
    %% FORWARD PASS
    % snet input to hidden
    for p = 1:nPar
        Wih0 = squeeze(snet.Wih(p,:,:));
        in0 = IN(:, snet.in(p,:));

        hid(:,snet.hid(p,:)) = sig(in0*Wih0);
    end
    % combine for hidden to out
    out(:,:) = sig(hid*snet.Who);

end