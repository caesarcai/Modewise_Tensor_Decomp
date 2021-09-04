function P2 = vec2ten_ind(V, n_dims, mod)
% This function transforms the scalar indices to tensor indices.
% If mod ~= [], then the V stands for the column indices of the matrix
% unfloded from mode mod. Otherwise, V stands for the indices of the
% entries of the tensor.

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

