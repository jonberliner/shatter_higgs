function [percent, binary, conf, rmse] = get_err(yhat, y)

    % get err
    rmse = sqrt( mean( (yhat-y).^2, 2) );
    conf = [mean(rmse), std(rmse)];
    
    % force binary choice
    [toss biny] = max(y,[],2);
    [toss binyhat] = max(yhat,[],2);
    binary = biny==binyhat;
    percent = sum(~binary)/length(binary);
    
end
        