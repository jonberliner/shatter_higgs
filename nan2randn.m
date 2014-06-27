function a = nan2randn(a)

    a(isnan(a)) = randn(sum(isnan(a(:))),1);
    
end