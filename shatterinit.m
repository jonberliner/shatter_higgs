function snet = shatterinit3(nIn,nHid,nOut,nPar)
% function net = shatterinit2(nIn,nHid,nOut,nPar)
% IN:
%   nIn : number input nodes
%   nHid : number hidden nodes (currently only 1 hidden layer)
%   nOut : number output nodes
%   nPar : number of partitions
%
% % Jon Berliner 6.9.14

    nPerParIn = nIn/nPar; % how many input units per mini-network?
    nPerParHid = nHid/nPar; % how many hidden units per mini-network?

    assert(nPerParIn==round(nPerParIn),'nPart must divide nIn');
    assert(nPerParHid==round(nPerParHid),'nPart must divide nHid');
    
    shatterin = randperm(nIn); % shuffle input
    shatterhid = randperm(nHid); % shuffle hidden
    
    i0 = 1;
    i1 = nPerParIn;
    h0 = 1;
    h1 = nPerParHid;
    for p = 1:nPar
        
        
        snet.in(p,:) = shatterin(i0:i1);
        snet.hid(p,:) = shatterhid(h0:h1);
        
        snet.Wih = randn(nPar,nPerParIn,nPerParHid) / nPerParIn;
        snet.Who = randn(nHid,nOut) / nHid;
        
        i0 = i0 + nPerParIn;
        i1 = i1 + nPerParIn;
        h0 = h0 + nPerParHid;
        h1 = h1 + nPerParHid;
        
    end
    
end