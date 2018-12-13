function AQ = gen_matrix_c_known(Ai,c,Q)
[m,~,~] = size(Ai);
AQ = zeros(m,Q);
for i = 1:Q
    AQ(:,i) = Ai(:,:,i)*c;
end
end

