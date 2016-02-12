% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features,xxc,yyc] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

% Placeholder that you can delete. Empty features.
%features = zeros(size(x,1), 128);

%% drop the pixels that doesn't have so much surroundings(too close to the bondary)
% let the SIFT be satndard size
xran=size(image,2);
yran=size(image,1);
xeff_min=10;
yeff_min=10;
xeff_max=xran-10;
yeff_max=yran-10;

xmin_t=x-xeff_min; ind1=find(xmin_t<0); 
if ~isempty(ind1)
    x(ind1)=[];
    y(ind1)=[];
else
end
xmax_t=xeff_max-x; ind2=find(xmax_t<0);
if ~isempty(ind2)
    x(ind2)=[];
    y(ind2)=[];
else
end
ymin_t=y-yeff_min; ind3=find(ymin_t<0); 
if ~isempty(ind3)
    x(ind3)=[];
    y(ind3)=[];
else
end
ymax_t=yeff_max-y; ind4=find(ymax_t<0); 
if ~isempty(ind4)
    x(ind4)=[];
    y(ind4)=[];
else
end        

xxc=x;
yyc=y;
%% for each pixel establish 4*4 histogram, each historam has 8directions
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)
Number=length(x);
features=zeros(Number,128); %l-r u-d 16 segments, each length is 8

%% form oriented gaussian filters 
angle=(0:45:315);
theta=angle/180*pi;
sigma=1;
L=ceil(3*sigma);
F =(-L:1:L);
G = exp(-(F.^2)/(2*sigma^2));
GD = -(F./sigma).*exp(-(F.^2)/(2*sigma^2));

%% calculate the histogram of each cell individually
% features_raw1=zeros(Number,128); %pixel contribute to every direction
features_raw2=zeros(Number,128); %pixel contribute to two signifinat direction

%%---------smoothing------------
%%---------gaussian weighing--------
sigma2=8;
xa=(1:1:16)-8.5;
ya=(1:1:16)-8.5;
[YY,XX]= meshgrid(ya,xa);
Gweight= exp(-((XX).^2+(YY).^2)/(2*sigma2^2));
%construct weight function

for k=1:1:Number
    xc=x(k);
    yc=y(k);
    slice=image((yc-7):(yc+8),(xc-7):(xc+8));
    
    I_x=zeros(16,16);
    I_y=zeros(16,16);
    
    for i=1:4:13
        for j=1:4:13
            I_x(i:(i+3),j:(j+3))=imfilter(imfilter(slice(i:(i+3),j:(j+3)),GD,'same'),(G'),'same');
            I_y(i:(i+3),j:(j+3))=imfilter(imfilter(slice(i:(i+3),j:(j+3)),G,'same'),(GD'),'same');
        end
    end
        
    gradient_raw=zeros(16,16,8);
    for n=1:1:8
        gradient_raw(:,:,n)=Gweight.*(cos(theta(n))*I_x+sin(theta(n))*I_y);
    end
       
%     % since each gradient contribute to two orientations  
%     for n=1:1:8
%         for m=1:1:4
%             for l=1:1:4
%                 features_raw1(k,((m-1)*4+l-1)*8+n)=sum(sum(gradient_raw( (m*4-3):(m*4),(l*4-3):(l*4),n )));
%             end
%         end
%     end
%     %raw 1 complete
       
    %for raw2:
    for m=1:1:16
        for l=1:1:16
            temp1=gradient_raw(m,l,:);
            [temp2,ind1]=sort(temp1);
            temp1=zeros(1,8);
            temp1(ind1(7))=temp2(7);
            temp1(ind1(8))=temp2(8);
            gradient_raw(m,l,:)=temp1;
        end
    end
    %espetially remember gradient_raw is changed
 
    % interpolate using convolution to operate
    
    %
    
    for n=1:1:8
        for m=1:1:4
            for l=1:1:4
                features_raw2(k,((m-1)*4+l-1)*8+n)=3/4*sum(sum(gradient_raw( (m*4-3):(m*4),(l*4-3):(l*4),n )))...
                    + 1/4*sum(sum(gradient_raw( (max(m*4-3,1)):(min(m*4,16)),(max(l*4-3,1)):(min(l*4,16)),n )));
            end
        end
    end
    %raw2 complete
end
    %%complete feature_raw for everypoint

%% like 'trilinear interpolation' (contribution to 8cabins)
%%since feature_raw2 have already selected the two orientations for each
%%just use convolution to make contribution to 4 space cabin
%%-----------but still wonder if the contribution is identical

%         filter=1/4*ones(2,2);
%         for k=1:1:Number
%             for n=1:1:8
%                 temp3=imfilter( reshape(features_raw2(k,(n:8:end)),4,4)' , filter ,'same'); %remeber to ranspose
%                 features_raw2(k,(n:8:end))=reshape(temp3',1,16);
%             end
%         end
    
%% normalization
%%in normalizing 128D vector

Vthresh=0.2; %all value elevated by 0.2
for k=1:1:Number
    %%normalize the max to be 1
     Vmin=min(features_raw2(k,:));
     Vmax=max(features_raw2(k,:));
     features(k,:)= (features_raw2(k,:)-Vmin).* (1-Vthresh)./(Vmax-Vmin) +Vthresh;
    %%normalize the distance to be 1
%      distance=(sum(features_raw2(k,:).^2)).^(1/2);
%      features(k,:)= features_raw2(k,:)./distance+0.2;
end

end








