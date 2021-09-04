function [Core, X_sub_mat] = Chidori_CUR(X, R, const)
% Syntex:
% [Core,X_sub_mat] = Chidori_CUR(X,R,const)
%
% Environment：
% This function is developed with Tensor Toolbox v3.1
% URL: https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.1
% 
% Inputs:
% X: Inputed tensor.  
% R: Targeted multilinear rank.
% const: Sampling constant, see paper for details. Default value: 2
%
% Outputs:
% Core : Core tensor, i.e., \mathcal{R}.
% X_sub_mat : Chidori CUR components, i.e. {C_i U_i^\dagger}.
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
    const = 2;
end


n_dim = size(X);
mod_num = ndims(X);


Sub_ten_ind = struct('type','()','subs',{[]});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Randomly generate the sub matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I = cell(mod_num,1);
for it_mod = 1:mod_num
    len2 = min(ceil(const*R(it_mod)*log(n_dim(it_mod))),n_dim(it_mod));
    I{it_mod} = randi(n_dim(it_mod),[len2 1]);
end
X_sub_mat = cell(mod_num,1);
I_temp = I;
for it_mod = 1:mod_num
    %I_temp = I;
    I_temp{it_mod} = (1:n_dim(it_mod))';
    Sub_ten_ind.subs = I_temp;
    C = subsref(X,Sub_ten_ind);
    C = double(tenmat(C,it_mod));
    U = C(I{it_mod},:);
    [u,s,v] = svd(U,'econ');
    r = R(it_mod);
    X_sub_mat{it_mod} = C*v(:,1:r)*(pinv(s(1:r,1:r)))*(u(:,1:r))';
    I_temp{it_mod} = I{it_mod};
end
Sub_ten_ind.subs = I;
Core = subsref(X,Sub_ten_ind);
