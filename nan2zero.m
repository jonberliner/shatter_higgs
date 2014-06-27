function a = nan2zero(a)

    a(isnan(a)) = zeros(sum(isnan(a(:))),1);
    
end