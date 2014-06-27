function eta = adadelta(leak,epsilon)

% g gradient of t-th term

%E[g^2]_t = leak * E[g^2]_{t-1} + (1-leak) g^2_t

leak * prev_grad 

D = - rms(prev_D



gradMag = (leak * prev_grad.^2) + ((1-leak) * grad.^2);
RMSg_t = sqrt( prev_gradMag + epsilon );

D = -( sqrt(mean(prev_D.^2)) ./ 


prevGrad = ( leak * prevGrad ) + ((1-leak) * grad);


prev_D

prevGrad2 = 0;