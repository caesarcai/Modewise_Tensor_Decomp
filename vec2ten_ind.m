function P2 = vec2ten_ind(V, n_dims, mod)
% This function is used to transform the scalar indices to the indices for tensor
% Input: n_dimes: tensor size, e.g. tensor A is of the size [2 3 4]
%        mod: the mode that tensor is unfloding along with,  e.g. mod = emptyset or 1 or 2 or 3     
%        V: indices vector, if mod = [], then V is a vector with each entry in {1,...,24};
%                           if mod = 1, then V is a vector with each entry in {1,...,12};
%                           if mod = 2, then V is a vector with each entry in {1,...,8}
%                           if mod = 3, then V is a vector with each entry in {1,...,6}
%
% Output: P2:  a cell which contains the indices for tensor, e.g. the size of P2{1,1} is length(V) * 3

num_mod = length(n_dims);
num_indices = length(V);
if nargin<3
    mod = num_mod+1;
    n_dims = [n_dims 1];
end
P = zeros(num_indices*n_dims(mod),num_mod);
prod_all_mod = prod(n_dims)/n_dims(mod);
P1 = zeros(num_indices,num_mod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = num_mod:-1:mod+1
    prod_all_mod = prod_all_mod/n_dims(i);
    P1(:,i) = ceil(V/prod_all_mod);
    V = rem(V-1,prod_all_mod)+1;
end
for i = mod-1:-1:1
    prod_all_mod = prod_all_mod/n_dims(i);
    P1(:,i) = ceil(V/prod_all_mod);
    V = rem(V-1,prod_all_mod)+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:num_indices
    if num_mod ~= length(n_dims)-1
        P((i-1)*n_dims(mod)+1:i*n_dims(mod),mod) = (1:n_dims(mod))';
    end
    if mod ~= 1
        P((i-1)*n_dims(mod)+1:i*n_dims(mod),1:mod-1) = ones(n_dims(mod),1)*(P1(i,1:mod-1));
    end
    if mod ~= num_mod
        P((i-1)*n_dims(mod)+1:i*n_dims(mod),mod+1:num_mod) = ones(n_dims(mod),1)*(P1(i,mod+1:num_mod));
    end
end
P2 = cell(1,1);
P2{1,1} = P;

