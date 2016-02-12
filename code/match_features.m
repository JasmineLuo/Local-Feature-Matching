% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features 1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2,xc1,yc1,xc2,yc2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Placeholder that you can delete. Random matches and confidences
num_features1 = size(features1, 1);
num_features2 = size(features2, 1);
% matches = zeros(num_features1, 2);
% matches(:,1) = randperm(num_features1);
% matches(:,2) = randperm(num_features2);

%% calculate mutual distance between every two point
Distance=zeros(num_features1,num_features2);
for i=1:1:num_features1
    for j=1:1:num_features2
        temp=sum((features1(i,:)-features2(j,:)).^2);
        Distance(i,j)=temp.^(1/2);
    end
end

MaxBound=max(max(Distance));
%% --------------------change to make difference between each point bigger let them between 0.01-1
% MinBound=min(min(Distance));
% Distance=(Distance-MinBound).*(1-0.01)./(MaxBound-MinBound)+0.01;
%% -------------------fen ge xian---------------
NNDR_thresh=0.95; %%uper is closest, down is second closest, if value is smaller than cetain value,
               %%condier good match
%space_thresh=1000; % unit: pixel; can be changed to other scale
%%the search scheme is always find the smallest of the total, and extract
%%the collum and row once consider mateched
ind1=(1:1:num_features1);
ind2=(1:1:num_features2);
%current search range;
% if num_features1>=num_features2
%     master=2; %search limited by colum
% else
%     master=1; %search limited by 
% end

match_feature1=[];
match_feature2=[];
NNDR=[];
Conficence_invert=[];
flag=1; %to ensure stop when it comes to find filtered pair again
        %stop when flag==0

while ~isempty(ind1) && ~isempty(ind2) && flag~=0
    %% find min point
    minv=min(min(Distance(ind1,ind2)));
    [ypos,xpos]=find(Distance(ind1,ind2)==min(min(Distance(ind1,ind2))));
    min1=minv(1);
    if min1>MaxBound %%---------------change 'MaxBound' to 1
        break;
    else
    end
    xpos1=xpos(1);
    ypos1=ypos(1);
%     if mod(pos(1),length(ind2))~=0
%         xpos1=mod(pos(1),length(ind2));
%         ypos1=floor(pos(1)/length(ind2))+1;
%     else
%         xpos1=length(ind2);
%         ypos1=pos(1)/length(ind2);
%     end
        %%possition of current scope!!!
    %% look for second min point in both row and colum
    [temp1,pos21]=sort(Distance(ind1(ypos1),ind2));%search in row
    [temp2,pos22]=sort(Distance(ind1,ind2(xpos1)));
    if length(temp1)>1 && length(temp2)>1
        if temp1(2)>=temp2(2)
            min2=temp2(2)';
            xpos2=xpos1;
            ypos2=ind1(pos22(2)');
            % two in same coloum ie, find second close match in 2 to 1
        else
            min2=temp1(2);
            xpos2=ind2(pos21(2));
            ypos2=ypos1;
        end
    else
        if length(temp1)==1
            min2=temp2(2)';
            xpos2=xpos1;
            ypos2=ind1(pos22(2)');
        else
            min2=temp1(2);
            xpos2=ind2(pos21(2));
            ypos2=ypos1;
        end     
    end
    
        %% filter the pairs with significnatly different location, to avoid mismatch of periodic or symetric structures
        %radius = ((xc1(ind1(ypos1))-xc2(ind2(xpos1)))^2+(yc1(ind1(ypos1))-yc2(ind2(xpos1)))^2)^(1/2);
    if  min1>min2*NNDR_thresh %|| radius >= space_thresh
        %% filtter the ones with NNDR greater than threshold
        Distance(ind1(ypos1),ind2(xpos1))= MaxBound*10; %%------------change from MaxBound*10
        % avoid to find them again
    else
        %% record and extract xpos1, ypos1
        match_feature1=[match_feature1,ind1(ypos1)];
        match_feature2=[match_feature2,ind2(xpos1)];
        NNDR=[NNDR,min1/min2];
        Conficence_invert=[Conficence_invert,min1];
        ind1(ypos1)=[];
        ind2(xpos1)=[];
        %extract the matched pair
    end
    
    % detect the filtered pair, still have problem to jumpout during the
    % loo
    if min1>=MaxBound
        flag=0;
    else
    end
    
end

Total=length(match_feature1)-1;
matches=[match_feature1(1:1:Total)',match_feature2(1:1:Total)'];
confidences=1./Conficence_invert(1:1:Total);

% confidences = rand(num_features1,1);
%You will implement the "ratio test" or "nearest neighbor distance ratio test"
%method of matching local features as described in the lecture materials and Szeliski 4.1.3. 
%See equation 4.18 in particular. Simply compute all pairs of distances between all features.
% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
% [confidences, ind] = sort(confidences, 'descend');
% matches = matches(ind,:);