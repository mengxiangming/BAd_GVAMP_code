function AQ = gen_matrix(Ai,b,Q)
[m,n,~] = size(Ai);
AQ = zeros(m,n);
for i = 1:Q
    AQ = AQ+b(i)*Ai(:,:,i);
end
end

