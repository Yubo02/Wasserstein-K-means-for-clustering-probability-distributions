% Get the distance matrix D
D = readmatrix('Dis_mat.txt');

% Specify the number of clusters K
K = 2;


% WSDP
X_2 = kmeans_sdp_2(D, K);

% Rounding the result to assignments
[U_2sd,S_2sd,V_2sd] = svd(X_2);
X_sdp= U_2sd(:,1:K);

assign_WSDP = kmeansplus(X_sdp',K)';

 