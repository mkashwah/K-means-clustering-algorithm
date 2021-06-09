%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Author: Mohammed Kashwah %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Please note that this code is slightly slow and it might take up to 1
%minute to return. Thank you :)
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
c = 2;
iterations = 10;         %iterations after which algorithm stops
theta = 0.001;                %threshold
i = 1;                      %counter for iterations
c1_i = 0;                   %cluster 1 counter
c2_i = 0;                   %cluster 2 counter
error = zeros(iterations,c);                  %initialize error criterion array
exit_criterion = 1;         %initialize exit criterion > theta
X_labeled = [X zeros(size(X,1),1)];       %X after clustering and labeling

%initial means pixels at index X(1) and X(30000), these where chosen far
%enough from each other to allow for better clustering

mu_1 = X(1,:);
mu_2 = X(30000,:);

new_mu_1 = [0,0,0];
new_mu_2 = [0,0,0];

%mu matrix to keep values of mu for plotting
mu1_matrix = zeros(iterations,3);
mu2_matrix = zeros(iterations,3);
%% Start K means algorithm

while (i <= iterations && exit_criterion > theta)
    mu1_matrix(i,:) = mu_1;
    mu2_matrix(i,:) = mu_2;
    %cluster all the points with their respective mu
    for ii = 1: size(X,1)
        euc_dist1 = pdist2(mu_1, X(ii,:));        %Euclidian distance between X(ii) and mu1
        euc_dist2 = pdist2(mu_2, X(ii,:));        %Euclidian distance between X(ii) and mu2
        
        %labeling
        if euc_dist1 <= euc_dist2
            X_labeled(ii,:) = [X(ii,:) 0];          %0 is label for 1st cluster
            error(i,1) = error(i,1) + euc_dist1;    %add to the error criterion of the current iteration
            new_mu_1 = new_mu_1 + X(ii,:);      %will be used to find the new mean
            c1_i = c1_i + 1;                    %counter++
        else
            X_labeled(ii,:) = [X(ii,:) 1];          %1 is label for 2nd cluster
            error(i,2) = error(i,2) + euc_dist2;    %add to the error criterion of the current iteration
            new_mu_2 = new_mu_2 + X(ii,:);      %will be used to find the new mean
            c2_i = c2_i + 1;                    %counter++
        end 
    end
    
    %Recompute mu_1 and mu_2
    new_mu_1 = new_mu_1 ./ c1_i;        %new mean
    new_mu_2 = new_mu_2 ./ c2_i;        %new mean
    
    c1_i = 0;                   %zero cluster 1 counter
    c2_i = 0;                   %zero cluster 2 counter
    
    %update exit criterion
    exit_criterion = (pdist2(mu_1, new_mu_1) + pdist2(mu_2, new_mu_2));
    
    %update mu1 and mu2 for next iteration
    mu_1 = new_mu_1;
    mu_2 = new_mu_2;
    
    new_mu_1 = [0,0,0];     %zero new mu1
    new_mu_2 = [0,0,0];     %zero new mu2
    
    %next iteration
    i = i+1;
    
end
X_labeled_copy = X_labeled;          %keeping a copy of X_labeled

%plotting the image back
for iii = 1:size(X_labeled,1)
    if X_labeled(iii, 4) == 0
        X_labeled(iii, 1:3) = mu_1;
    elseif X_labeled(iii, 4) == 1
        X_labeled(iii, 1:3) = mu_2;
    end
end

I_labeled = reshape(X_labeled(:,1:3),256,256,3);
figure
imshow(uint8(I_labeled))
title('(iv) Plot of the image in its mean colors')

figure
plot(error(1:i-1,1)+error(1:i-1,2))
xlabel('#iterations')
ylabel('Error Criterion (J)')
title('(i)Error Criterion Function vs #Iterations')

%%
figure
plot3(mu1_matrix(1:i-1,1),mu1_matrix(1:i-1,2),mu1_matrix(1:i-1,3),'s','Color',[136.393011118675 91.4699846204271 93.6968711393490]/255)
hold on
plot3(mu2_matrix(1:i-1,1),mu2_matrix(1:i-1,2),mu2_matrix(1:i-1,3),'x','Color',[162.156070034399 196.664244579291 216.105206199513]/255)
hold off
xlabel('R')
ylabel('G')
zlabel('B')
title('(ii) Cluster Means Through Iterations')
legend('mu1', 'mu2')

%%
figure
plot3(X_labeled_copy(find(X_labeled_copy(:,4)==0),1),X_labeled_copy(find(X_labeled_copy(:,4)==0),2),X_labeled_copy(find(X_labeled_copy(:,4)==0),3),'.','Color',[136.393011118675 91.4699846204271 93.6968711393490]/255)
hold on
plot3(X_labeled_copy(find(X_labeled_copy(:,4)==1),1),X_labeled_copy(find(X_labeled_copy(:,4)==1),2),X_labeled_copy(find(X_labeled_copy(:,4)==1),3),'.','Color',[162.156070034399 196.664244579291 216.105206199513]/255)
hold off
xlabel('R')
ylabel('G')
zlabel('B')
title('(iii) Labeled Data Samples in RGB space')
legend('Cluster #1', 'Cluster #2')

