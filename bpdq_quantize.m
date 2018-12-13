% bpdq_quantize : Uniform scalar quantization
% xq = bpdq_quantize(x,NumBits,delta)

function x_quantize = bpdq_quantize(x,NumBits,delta)
[m,n] = size(x);
x = x(:);

b = (-2^NumBits/2 + 1):2^NumBits/2;
R_set = (-1/2 + b).*delta;

alpha = delta;


index = 2^NumBits/2 + sign(x).*min(2^NumBits/2,ceil(abs(x)/alpha)) + (1-sign(x))/2;

x_quantize = reshape(R_set(index),m,n);




