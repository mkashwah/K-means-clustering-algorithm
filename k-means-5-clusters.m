%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Author: Mohammed Kashwah %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Please note that this code is slightly slow and it might take up to 2
%minutes to return. Thank you :)
clear
clc


%% Loading the image

I = imread('house.tiff');
figure
imshow(I)
title('original picture before segmentation')

X = reshape(I, 256 * 256, 3);       %reshaping the 3d matrix into 2d matrix r,g,b
X = double(X);                      %convert unit8 to double

figure
plot3(X(:,1),X(:,2),X(:,3),'.','Color',[ 0.8 0.2 0.3 ])         %3d plot of unclustered pixels in RGB space
title('Samples (X) in RGB space')
xlabel('R')
ylabel('G')
zlabel('B')
%% Initialization
c = 5;
iterations = 10;         %iterations after which algorithm stops
theta = 0.001;                %threshold
i = 1;                      %counter for iterations
c1_i = 0;                   %cluster 1 counter
c2_i = 0;                   %cluster 2 counter
c3_i = 0;                   %cluster 3 counter
c4_i = 0;                   %cluster 4 counter
c5_i = 0;                   %cluster 5 counter

error = zeros(iterations,c);                  %initialize error criterion array
exit_criterion = 1;         %initialize exit criterion > theta
X_labeled = [X zeros(size(X,1),1)];       %X after clustering and labeling

%initial means pixels at index X(1), X(10000),X(20200),X(35600),X(61500) these where chosen far
%enough from each other to allow for better clustering

mu_1 = X(1,:);
mu_2 = X(10000,:);
mu_3 = X(20200,:);
mu_4 = X(35600,:);
mu_5 = X(61500,:);

%different initial means. Please uncomment to run 2nd initial states
%%% 2nd initia pixels at indices X(120), X(250),X(20200),X(64000),X(65000)
% mu_1 = X(120,:);
% mu_2 = X(250,:);
% mu_3 = X(20200,:);
% mu_4 = X(64000,:);
% mu_5 = X(65000,:);

new_mu_1 = [0,0,0];
new_mu_2 = [0,0,0];
new_mu_3 = [0,0,0];
new_mu_4 = [0,0,0];
new_mu_5 = [0,0,0];

%mu matrix to keep values of mu for plotting not needed for this question
%but included anyway
mu1_matrix = zeros(iterations,3);
mu2_matrix = zeros(iterations,3);
mu3_matrix = zeros(iterations,3);
mu4_matrix = zeros(iterations,3);
mu5_matrix = zeros(iterations,3);
%% Start K means algorithm

while (i <= iterations && exit_criterion > theta)
    mu1_matrix(i,:) = mu_1;
    mu2_matrix(i,:) = mu_2;
    mu3_matrix(i,:) = mu_3;
    mu4_matrix(i,:) = mu_4;
    mu5_matrix(i,:) = mu_5;
    
    %cluster all the points with their respective mu
    for ii = 1: size(X,1)
        euc_dist1 = pdist2(mu_1, X(ii,:));        %Euclidian distance between X(ii) and mu1
        euc_dist2 = pdist2(mu_2, X(ii,:));        %Euclidian distance between X(ii) and mu2
        euc_dist3 = pdist2(mu_3, X(ii,:));        %Euclidian distance between X(ii) and mu3
        euc_dist4 = pdist2(mu_4, X(ii,:));        %Euclidian distance between X(ii) and mu4
        euc_dist5 = pdist2(mu_5, X(ii,:));        %Euclidian distance between X(ii) and mu5
        
        euc_dist_array = [euc_dist1, euc_dist2, euc_dist3, euc_dist4, euc_dist5];       %array containing all the euclidian distances
        
        [min_euc, indx] = min(euc_dist_array);      %find minimum euclidean distance and return it with its index
        
        %labeling
        X_labeled(ii,:) = [X(ii,:) indx];          %indx as label
        error(i,indx) = error(i,indx) + euc_dist_array(indx);    %add to the error criterion of the current iteration
        
        if indx == 1
            new_mu_1 = new_mu_1 + X(ii,:);      %will be used to find the new mean
            c1_i = c1_i + 1;                    %counter++
        elseif indx == 2
            new_mu_2 = new_mu_2 + X(ii,:);      %will be used to find the new mean
            c2_i = c2_i + 1;                    %counter++
        elseif indx == 3
            new_mu_3 = new_mu_3 + X(ii,:);      %will be used to find the new mean
            c3_i = c3_i + 1;                    %counter++
        elseif indx == 4
            new_mu_4 = new_mu_4 + X(ii,:);      %will be used to find the new mean
            c4_i = c4_i + 1;                    %counter++
        elseif indx == 5
            new_mu_5 = new_mu_5 + X(ii,:);      %will be used to find the new mean
            c5_i = c5_i + 1;                    %counter++
        end
        
    end
    
    %Recompute mu_1 .. mu_5
    new_mu_1 = new_mu_1 ./ c1_i;        %new mean
    new_mu_2 = new_mu_2 ./ c2_i;        %new mean
    new_mu_3 = new_mu_3 ./ c3_i;        %new mean
    new_mu_4 = new_mu_4 ./ c4_i;        %new mean
    new_mu_5 = new_mu_5 ./ c5_i;        %new mean
    
    
    c1_i = 0;                   %zero cluster 1 counter
    c2_i = 0;                   %zero cluster 2 counter
    c3_i = 0;                   %zero cluster 3 counter
    c4_i = 0;                   %zero cluster 4 counter
    c5_i = 0;                   %zero cluster 5 counter
    
    %update exit criterion
    exit_criterion = (pdist2(mu_1, new_mu_1) + pdist2(mu_2, new_mu_2) + pdist2(mu_3, new_mu_3) + pdist2(mu_4, new_mu_4) + pdist2(mu_5, new_mu_5));
    
    %update mu1 .. mu5 for next iteration
    mu_1 = new_mu_1;
    mu_2 = new_mu_2;
    mu_3 = new_mu_3;
    mu_4 = new_mu_4;
    mu_5 = new_mu_5;
   
    
    new_mu_1 = [0,0,0];     %zero new mu1
    new_mu_2 = [0,0,0];     %zero new mu2
    new_mu_3 = [0,0,0];     %zero new mu3
    new_mu_4 = [0,0,0];     %zero new mu4
    new_mu_5 = [0,0,0];     %zero new mu5
    
    %next iteration
    i = i+1;
    
end

X_labeled_copy = X_labeled;     %keeping a copy of X_labeled

%plotting the image back
for iii = 1:size(X_labeled,1)
    if X_labeled(iii, 4) == 1
        X_labeled(iii, 1:3) = mu_1;
    elseif X_labeled(iii, 4) == 2
        X_labeled(iii, 1:3) = mu_2;
    elseif X_labeled(iii, 4) == 3
        X_labeled(iii, 1:3) = mu_3;
    elseif X_labeled(iii, 4) == 4
        X_labeled(iii, 1:3) = mu_4;
    elseif X_labeled(iii, 4) == 5
        X_labeled(iii, 1:3) = mu_5;
    end
end

I_labeled = reshape(X_labeled(:,1:3),256,256,3);
figure
imshow(uint8(I_labeled))
title('Plot of the image in its mean colors')

figure
plot(error(1:i-1,1)+error(1:i-1,2)+error(1:i-1,3)+error(1:i-1,4)+error(1:i-1,5))
xlabel('#iterations')
ylabel('Error Criterion (J)')
title('Error Criterion Function vs #Iterations')

%%
% figure
% plot3(mu1_matrix(1:i-1,1),mu1_matrix(1:i-1,2),mu1_matrix(1:i-1,3),'s','Color',[136.393011118675 91.4699846204271 93.6968711393490]/255)
% hold on
% plot3(mu2_matrix(1:i-1,1),mu2_matrix(1:i-1,2),mu2_matrix(1:i-1,3),'x','Color',[162.156070034399 196.664244579291 216.105206199513]/255)
% hold off

%%
figure
plot3(X_labeled_copy(find(X_labeled_copy(:,4)==1),1),X_labeled_copy(find(X_labeled_copy(:,4)==1),2),X_labeled_copy(find(X_labeled_copy(:,4)==1),3),'.','Color',mu_1/255)
hold on
plot3(X_labeled_copy(find(X_labeled_copy(:,4)==2),1),X_labeled_copy(find(X_labeled_copy(:,4)==2),2),X_labeled_copy(find(X_labeled_copy(:,4)==2),3),'.','Color',mu_2/255)
plot3(X_labeled_copy(find(X_labeled_copy(:,4)==3),1),X_labeled_copy(find(X_labeled_copy(:,4)==3),2),X_labeled_copy(find(X_labeled_copy(:,4)==3),3),'.','Color',mu_3/255)
plot3(X_labeled_copy(find(X_labeled_copy(:,4)==4),1),X_labeled_copy(find(X_labeled_copy(:,4)==4),2),X_labeled_copy(find(X_labeled_copy(:,4)==4),3),'.','Color',mu_4/255)
plot3(X_labeled_copy(find(X_labeled_copy(:,4)==5),1),X_labeled_copy(find(X_labeled_copy(:,4)==5),2),X_labeled_copy(find(X_labeled_copy(:,4)==5),3),'.','Color',mu_5/255)
hold off
xlabel('R')
ylabel('G')
zlabel('B')
title('(iii) Labeled Data Samples in RGB space')
legend('Cluster #1', 'Cluster #2','Cluster #3','Cluster #4','Cluster #5')
