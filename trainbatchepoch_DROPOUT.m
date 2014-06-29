function [net,out] = trainbatchepoch_dropout(net,bIN,bTAR,lrate)
    
    [nPar, nInPer, nHidPer] = size(net.Wih);
    nHid = nPar * nHidPer;
    
    [nPer, nIn, nBatch] = size(bIN);
    [nPer, nOut, nBatch] = size(bTAR);
    
    
    nHidD2 = floor(nHid/2);
    nHidPerD2 = floor(nHidPer/2);
    hid = nan(nPer, nHidD2);
    hused = nan(nPar, nHidPerD2);
    ihused = nan(nPar, nHidPerD2);
    
    
    sig = @(x) 1./(1 + exp(-x)); % sigmoidal transfer function
    nonans = @(in) sum( isnan( in(:) ) ) == 0; % helper fcn
    
    
    for b = 1:nBatch

        in = squeeze(bIN(:, :, b));
        tar = squeeze(bTAR(:, :, b));
        
        %% CHOOSE DROPOUT NODES
        for p = 1:nPar
            hused(p, :) = randsample(nHidPer, nHidPerD2); % choose nodes
            ihused(p, :) = net.hid(p, hused(p, :)); % get shatterIDs
        end
        
        %% FORWARD PASS
        % net input to hidden
        ii = 1:nHidPerD2;
        for p = 1:nPar
            
            hused0 = hused(p, :);
            
            % filter2used
            Wih0 = squeeze(net.Wih(p, :, hused0));
            
            in0 = in(:, net.in(p, :));

            hid(:, ii) = sig(in0 * Wih0);
            
            ii = ii + nHidPerD2;
        end
        
        assert(nonans(hid));
        
        % combine for hidden to out
        out = sig(hid * net.Who(ihused(:), :));

        %% BACKPROP
        % update ho
        dout = out.*(1 - out) .* (tar - out); % nPerxnOut .x nPerxnOut .x nPerxnOut
        dWho = (hid' * dout) / nPer ; % nHidxnPer x nPerxnOut

        net.Who(ihused(:), :) = net.Who(ihused(:), :) + lrate * dWho; % update DROPOUT nodes

        assert(nonans(net.Who));
        
        % update ih
        ii = 1:nHidPerD2;
        for p = 1:nPar
               
            hused0 = hused(p, :); % which hid units in partition are on
            % which hidden units IN WHOLE NET used in this dropped-out partition
            ihused0 = ihused(p, :);
            % in2hid weights for dropout-selected units in partition p
            Wih0 = squeeze(net.Wih(p, :, hused0));
            in0 = in(:, net.in(p, :));
            hid0 = hid(ii);
            
            dhid0 = hid0 .* (1 - hid0) .* (dout * net.Who(ihused0, :)'); % nPerxnHid .x nPerxnHid .x (nPerxnOut x nOutxnHid)
            dWih0 = (in0' * dhid0) / nPer; % nInxnPer x nPerxnHid
            Wih0 = Wih0 + lrate .* dWih0;

            net.Wih(p, :, hused0) = Wih0;
            
            ii = ii + nHidPerD2;
        end
    end