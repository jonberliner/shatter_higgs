function bestnet = early_search(nSearch,nIn,nHid,nOut,nPar,lrate,nEpoch,IN,TAR)

nSearch = 30;

nIn = 100;
nHid = 1000;
nOut = 10;
nPar = 10;

nets{nSearch} = []; % init cell array
for i = 1:nSearch
    nets{i} = shatterinit(nIn,nHid,nOut,nPar);
end


% parameterized 1 epoch of minibatch
shattertrain0 = @(net) shattertrain(net,IN,TAR,lrate,nEpoch);




[snet, out] = shattertrain(snet,IN,TAR,lrate,nEpoch)

out = cellfun(@shattertrain0,nets);