function [net,out] = trainbatchepoch(net,bIN,bTAR,lrate)
    
    [nPar nInPer nHidPer] = size(net.Wih);
    nHid = nPar * nHidPer;
    
    [nPer nIn nBatch] = size(bIN);
    [nPer nOut nBatch] = size(bTAR);
    
    hid = nan(nPer,nHid);
    
    
    sig = @(x) 1./(1+exp(-x)); % sigmoidal transfer function
    
    for b = 1:nBatch

        in = squeeze(bIN(:,:,b));
        tar = squeeze(bTAR(:,:,b));
        

        %% FORWARD PASS
        % net input to hidden
        for p = 1:nPar
            Wih0 = squeeze(net.Wih(p,:,:));
            in0 = in(:, net.in(p,:));

            hid(:,net.hid(p,:)) = sig(in0*Wih0);
            
            
        end
        % combine for hidden to out
        out = sig(hid*net.Who);

        %% BACKPROP
        % update ho
        dout = out.*(1-out) .* (tar-out); % nPerxnOut .x nPerxnOut .x nPerxnOut
        dWho = (hid' * dout) / nPer ; % nHidxnPer x nPerxnOut
        net.Who = net.Who + lrate * dWho;

        % update ih
        for p = 1:nPar

            Wih0 = squeeze(net.Wih(p, :, :));
            in0 = in(:, net.in(p, :));

            h0 = net.hid(p, :);
            hid0 = hid(:,h0);
            

            dhid0 = hid0.*(1-hid0) .* (dout*net.Who(h0,:)'); % nPerxnHid .x nPerxnHid .x (nPerxnOut x nOutxnHid)
            dWih0 = (in0' * dhid0) / nPer ; % nInxnPer x nPerxnHid
            Wih0 = Wih0 + lrate .* dWih0;

            net.Wih(p,:,:) = Wih0;

        end
    end