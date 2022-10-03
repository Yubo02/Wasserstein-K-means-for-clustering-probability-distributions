% Author:       Mixon, Villar, Ward.
% Filename:     kmeans_sdp.m
% Last edited:  9 May 2016 
% Description:  This function uses SDPNAL+ [3] to solve the Peng and Wei's 
%               kmeans SDP solution [2] according to the formulation in [1].
% Inputs:       
%               -P: 
% 
%               A d x N array of data points where d denotes the
%               dimension of the data and N denotes the number of points to 
%               cluster. Each column of P corresponds with a data point.
% 
%               -k: 
% 
%               The number of clusters.
%
%               -D:
%
%               Affinity matrix: Gaussian kernel (default).
%
%               -t:
%
%               Diffusion steps.
%
%               -lazy:
%
%               Sticky parameter for the lazy random walk (default: 0).
% Outputs:
%               -X:
%               
%               A N x N array corresponding to the solution of Peng and Wei's 
%               kmeans SDP solution. 
% 
% References:
% 
% [1] Mixon, Villar, Ward. Clustering subgaussian mixtures via semidefinite
%       programming
% [2] Peng, Wei. Approximating k-means-type clustering via semidefinite 
%       programming.
% [3] Yang, Sun, Toh. Sdpnal+: a majorized semismooth newton-cg augmented
%       lagrangian method for semidefinite programming with nonnegative 
%       constraints.
% -------------------------------------------------------------------------

function X=kmeans_sdp_2(D, k, t, lazy)

N=size(D,2); 

%construction of distance squared matrix.
% D = -P'*P;

%SDP definition for SDPNAL+
n=N;
C{1}=D;
blk{1,1}='s'; blk{1,2}=n;
b=zeros(n+1,1);
Auxt=spalloc(n*(n+1)/2, n+1, 5*n);
Auxt(:,1)=svec(blk(1,:), eye(n),1);
%Auxt(:,1) = svec(blk(1,:), zeros(n),1);
%b(1,1)=0;
b(1,1)=k;
idx=2;
for i=1:n
    A=zeros(n,n);
    A(:,i)=ones(n,1);
    A(i,:)=A(i,:)+ones(1,n);
    b(idx,1)=2;
    Auxt(:,idx)= svec(blk(1,:), A,1);
    idx=idx+1;
end
At{1}=sparse(Auxt);

OPTIONS.maxiter=1000;
OPTIONS.tol=1e-6;
OPTIONS.printlevel = 0;

%SDPNAL+ call
[obj,X,s,y,S,Z,y2,v,info,runhist]=sdpnalplus(blk,At,C,b,0,[],[],[],[],OPTIONS);

X=cell2mat(X);
end