function [outin outtar] = makebatches(in,tar,nPer,shuf_huh)

    [nSam nIn] = size(in);
    [nSam0 nOut] = size(tar);

    assert(nSam==nSam0, 'diff amount of samples and targets');

    % nPer = floor( nSam / nBatch );
    nBatch = floor( nSam / nPer );

    if shuf_huh
        shuf = randperm(nSam);

        in = in(shuf,:);
        tar = tar(shuf,:);
    end

    outin = nan(nPer,nIn,nBatch);
    outtar = nan(nPer,nOut,nBatch);

    i0 = 1;
    i1 = nPer;
    for b = 1:nBatch
        outin(:,:,b) = in(i0:i1,:);
        outtar(:,:,b) = tar(i0:i1,:);
        
        i0 = i0 + nPer;
        i1 = i1 + nPer;
    end

end