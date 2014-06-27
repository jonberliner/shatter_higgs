function ams = ams(yhat,y)

both = 

b_r = 10;

ams = sqrt( 2*(s+b+b_r) * ln(1+(s/(b+b_r)) - s ) );