function [Core, X_sub_mat] = Fiber_CUR(X, R, const_R, const_C)
% Syntex:
% [Core, X_sub_mat] = Fiber_CUR(X, R, const)
%
% Environment：
% This function is developed with Tensor Toolbox v3.1
% URL: https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.1
% 
% Inputs:
% X: Inputed tensor.  
% R: Targetted multilinear rank.
% const_R: Sampling constant for core tensor. Default value: 2
% const_C: Sampling constant for {C_i}. Default value: 4
%
% Outputs:
% Core : Core tensor, i.e., \nathcal{R}.
% X_sub_mat : Fiber CUR components, i.e. {C_i U_i^\dagger}.
%
% To obtain the full estimated tensor:
% X_est = tensor(ttensor(Core,X_sub_mat));
%
%
% Please cite the following paper if you find this code helpful:
%  HQ Cai, K Hamm, L Huang, and D Needell. Mode-wise Tensor Decompositions: 
%    Multi-dimensional Generalizations of CUR Decompositions. Journal of 
%    Machine Learning Research, 22.185: 1-36, 2021.
% 
% By:
%    HanQin Cai,     hqcai@math.ucla.edu
%    Keaton Hamm,    keaton.hamm@uta.edu
%    Longxiu Huang,  huangl3@math.ucla.edu
%    Deanna Needell, deanna@math.ucla.edu

if nargin == 2
    const_R = 2;
    const_C = 2*const_R;
elseif nargin == 3
    const_C = 2*const_R;
end


n_dim = size(X);
mod_num = ndims(X);

Sub_ten_ind = struct('type','()','subs',{[]});
prod_dim_all = prod(n_dim);


%%%%%%%%% Randomly generate the sub-matrices and sub-tensor %%%%%%%%%%%%%%%
I = cell(mod_num,1);
J = cell(mod_num,1);
for it_mod = 1:mod_num
    r = R(it_mod);
    n = n_dim(it_mod);
    len1 = min(ceil(const_R*r*log(n)),n);
    len2 = min(ceil(const_C*r*log(prod_dim_all/n)),prod_dim_all/n);
    I{it_mod} = randi(n,[len1 1]);
    J{it_mod} = randi(prod_dim_all/n,[len2 1]);
end
X_sub_mat = cell(mod_num,1);
for it_mod = 1:mod_num
    P = vec2ten_ind(J{it_mod},n_dim,it_mod);
    Sub_ten_ind.subs = P; %temp_ind';
    C = subsref(X,Sub_ten_ind);
    C = reshape(double(C),n_dim(it_mod),length(J{it_mod}));
    U = C(I{it_mod},:);
    [u,s,v] = svd(U,'econ');
    r = R(it_mod);
    X_sub_mat{it_mod} = C*v(:,1:r)*(pinv(s(1:r,1:r)))*(u(:,1:r))'; %
end

Sub_ten_ind.subs = I;
Core = subsref(X,Sub_ten_ind);
