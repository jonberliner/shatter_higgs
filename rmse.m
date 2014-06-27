function rmse = rmse(a,b)

    rmse = sqrt( mean( sum((a-b).^2,2) ,2) );
    
    
    
end