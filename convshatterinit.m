function csnet = convshatterinit(nIn,nHid,nOut,nPar)
% function net = convshatterinit(nIn,nHid,nOut,nPar)
% IN:
%   nIn : number input nodes
%   nHid : number hidden nodes (currently only 1 hidden layer)
%   nOut : number output nodes
%   nPar : number of partitions
%
% % Jon Berliner 6.9.14

    nPerParIn = nIn/nPar; % how many input units per mini-network?

    assert(nPerParIn==round(nPerParIn),'nPart must divide nIn');
    
    shatterin = randperm(nIn); % shuffle input
    
    i0 = 1;
    i1 = nPerParIn;
    for p = 1:nPar
        
        csnet.in(p,:) = shatterin(i0:i1);
     
        i0 = i0 + nPerParIn;
        i1 = i1 + nPerParIn; 
    end
           
    csnet.Wih = randn(nPerParIn,nHid) / nPerParIn;
    csnet.Who = randn(nHid,nOut) / nHid;
    
    csnet.pdWih = randn(nPerParIn,nHid) / nPerParIn;
    csnet.pdWho = randn(nHid,nOut) / nHid;
    
    csnet.nPar = nPar;
end