%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This scrip is a demo test case for tensor CUR.
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
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;

addpath(genpath('tensor_toolbox_3.1')) % load the tensor toolbox
% URL: https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.1

mod_num = 3; %% mode number
n  = 300; 
N = n*ones(1,mod_num);   % tensor dim = (n,...,n)
r = 5; 
R = r*ones(1,mod_num);   % targetted multilinear rank = (r,...r)
it_max = 5;              % run 5 trials for avg time and err
sig = 1e-3;              % noise variance
const_Chidori = 2;
const_Fiber1  = 2;
const_Fiber2  = 2*const_Fiber1;



disp('Generating a low-multilinear rank tensor.')
%%% generate a random tensor of size [n1 n2 n3] 
%%%%%%%%%%%%%%%%%%%%%%%%% generate a tensor with size n-by-n-by-n and
%%%%%%%%%%%%%%%%%%%%%%%%% multilinear rank [r,r,r]
X_origin = randn(R);
X_origin = tensor(X_origin);
for i = 1:mod_num
    X_origin = ttm(X_origin,randn(N(i),R(i)),i);
end


Err_Fiber = 0;
Tim_Fiber = 0;
Err_Chidori = 0;
Tim_Chidori = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ite = 1:it_max
    %%%% add random noise to the low-multilinear-rank tensor
    E = tensor(sig*randn(N));
    X = X_origin + E;
    clear E
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%% Calling Chidori CUR %%%%%%%%%%%%%%%%%%%%
    tic
    [Core_Chidori, X_sub_mat] = Chidori_CUR(X,R,const_Chidori);
    temp = toc;
    
    Y_cur_est = tensor(ttensor(Core_Chidori,X_sub_mat));
    Err_Chidori = Err_Chidori + norm(Y_cur_est-X_origin)/norm(X_origin);
    Tim_Chidori = Tim_Chidori + temp;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%% Calling Fiber CUR %%%%%%%%%%%%%%%%%%%%%
    tic
    [Core_Fiber, X_sub_mat] = Fiber_CUR(X,R,const_Fiber1,const_Fiber2);
    temp = toc;
    
    Y_cur_est = tensor(ttensor(Core_Fiber,X_sub_mat));
    Err_Fiber = Err_Fiber+norm(Y_cur_est-X_origin)/norm(X_origin);
    Tim_Fiber = Tim_Fiber+temp;
end

Err_Fiber = Err_Fiber/it_max;
fprintf('Relative error for Fiber CUR: %d \n',Err_Fiber)
Tim_Fiber = Tim_Fiber/it_max;
fprintf('Runtime for Fiber CUR: %d\n',Tim_Fiber)
Err_Chidori = Err_Chidori/it_max;
fprintf('Relative error for Chidori CUR: %d\n',Err_Chidori)
Tim_Chidori = Tim_Chidori/it_max;
fprintf('Runtime for Chidori CUR: %d\n',Tim_Chidori)
