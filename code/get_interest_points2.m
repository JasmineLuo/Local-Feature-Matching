% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points2(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete. 20 random points
% x = ceil(rand(20,1) * size(image,2));
% y = ceil(rand(20,1) * size(image,1));

%% step1, first derive dirivitive of Guassian-----------
L=5;%size of Laplatian of Gaussian;
devi=1;%standard deviation; ----follow the setting suggested

sobel_filter_x = [-1 0 1; -1 0 1; -1 0 1];
sobel_filter_y = sobel_filter_x';

c=(L+1)/2;
log_filter_x=zeros(L,L);
log_filter_y=zeros(L,L);
for i=1:1:L
    for j=1:1:L
        common=exp(-((j-c)^2+(i-c)^2)/(2*devi^2));
        log_filter_x(i,j)= -1*(j-c)/(devi^2)*common;
        log_filter_y(i,j)= -1*(i-c)/(devi^2)*common;
    end
end
%calculate individually

%log_filter_x=log_filter_x./(1.1267 + 1.4033 + 1.0209);
%!!!!!normalization 
%!!!!!normalization could be added if interest point found on flat plane


I_x=imfilter(image,log_filter_x,'same'); %get Ix
I_y=imfilter(image,log_filter_y,'same'); %get Iy

%!!!use outerproduct as <A,B>=B'*A, or should use kron (A,B)?
A11=I_x.*I_x; 
A12=I_y.*I_x;
A22=I_y.*I_y;

%biger Gaussian for window (wighing) function
w=fspecial('gaussian',6,1); 

A11_w=imfilter(A11,w,'same');
A12_w=imfilter(A12,w,'same');
A22_w=imfilter(A22,w,'same');

%A=[A11,A12;A21,A22]; %% so what is the use for this Gaussian filter?

%% Compute Scaler interest measure
% Corner Response function to identify
%alpha=0.04;
% divide sencondary momentA11
%R = A11_w.*A22_w-A12_w.^2-alpha*(A11_w+A22_w).^2;
threshold=0.01;
% divide sencondary momentA11
R = ((A11_w.*A22_w-A12_w.^2)./(A11_w+A22_w+eps));
M = ordfilt2(R,feature_width^2,ones(feature_width));
R = (R==M)& M > threshold; 
[y,x] = find(R);

R_result=zeros(1,length(x));
for m=1:1:length(x)
    R_result(m)=R(y(m),x(m));
end

figure; imshow(image); hold on;
scatter(x,y); hold off;
figure; imshow(R); hold on;
scatter(x,y); hold off;

% % % %% ---USE origin to see 
% % %  Rmax=max(max(R));
% % %  Rmin=min(min(R));
% % %  R=(R-Rmin)./(Rmax-Rmin)*100;
% % % % to define the number of local patching
% % % xran=size(R,2);
% % % yran=size(R,1);
% % % 
% % % nx=floor(xran/feature_width);
% % % ny=floor(yran/feature_width); %the rest pixels will be counted in last patches;
% % % xlast=xran-(nx-1)*feature_width;
% % % ylast=yran-(ny-1)*feature_width;
% % % 
% % % % find local maximum pixel restore location and R value
% % % x1=zeros(ny,nx);
% % % y1=zeros(ny,nx);
% % % R1=zeros(ny,nx);

% % % %% careful that the x1 y1 stands for local coordinate
% % % for i=1:1:(nx-1) % j in y direction
% % %     for j=1:1:(ny-1) % i in x
% % %         temp2=R( ((j-1)*feature_width+1):j*feature_width , ((i-1)*feature_width+1):i*feature_width );
% % %         ind=find(temp2==max(max(temp2)));
% % %         [xt,yt]=ind2sub([feature_width,feature_width],ind);
% % %     % for situation that many points share the same value
% % %     % take the first of them and it will be fine due to later constraints
% % %         x1(j,i)=yt(1)+(i-1)*feature_width;
% % %         y1(j,i)=xt(1)+(j-1)*feature_width;
% % %         R1(j,i)=max(max(temp2));
% % %     end
% % % end
% % % 
% % % for k=1:1:(ny-1) % search last coloum
% % %     temp2=R( ((k-1)*feature_width+1):k*feature_width , (xran-xlast)+1:xran );
% % %     ind=find(temp2==max(max(temp2)));
% % %     [xt,yt]=ind2sub([feature_width,feature_width],ind);
% % %     % for situation that many points share the same value
% % %     % take the first of them and it will be fine due to later constraints
% % %     x1(k,nx)=yt(1)+(nx-1)*feature_width;
% % %     y1(k,nx)=xt(1)+(k-1)*feature_width;
% % %     R1(k,nx)=max(max(temp2));
% % % end
% % % 
% % % for k=1:1:(nx-1) % search last raw
% % %     temp2=R( (yran-ylast)+1:yran ,((k-1)*feature_width+1):k*feature_width);
% % %     ind=find(temp2==max(max(temp2)));
% % %     [xt,yt]=ind2sub([feature_width,feature_width],ind);
% % %     % for situation that many points share the same value
% % %     % take the first of them and it will be fine due to later constraints
% % %     x1(ny,k)=yt(1)+(k-1)*feature_width;
% % %     y1(ny,k)=xt(1)+(ny-1)*feature_width;
% % %     R1(ny,k)=max(max(temp2));
% % % end
% % % 
% % % temp2=R( (yran-ylast)+1:yran ,(xran-xlast)+1:xran);
% % % ind=find(temp2==max(max(temp2)));
% % % [xt,yt]=ind2sub([feature_width,feature_width],ind);
% % % x1(ny,nx)=yt(1)+(nx-1)*feature_width;
% % % y1(ny,nx)=xt(1)+(ny-1)*feature_width;
% % % R1(ny,nx)=max(max(temp2)); % search last patch
% % % 
% % % % calculate basic even findings
% % % x=[];
% % % y=[];
% % % R_result=[];
% % % Radius=[];
% % % 
% % % for k=1:1:ny
% % %     x=[x,x1(k,:)];
% % %     y=[y,y1(k,:)];
% % %     R_result=[R_result,R1(k,:)];
% % % end
% % % 
% % % figure; imshow(I_x); hold on; 
% % % scatter(x,y); hold off;
% % % figure; imshow(R); hold on;
% % % scatter(x,y); hold off;
% % % %% -- EBD for origin
%% calculate the second list to extinguish the flat 
%Radius=floor(feature_width/2); %change to a variable Radius
%for each larger value find the nearest larger value
% Radius=[];
% for k=1:1:length(x)
%     temp5=R-R(y(k),x(k));
%     [r,c]=find(temp5);
%     if ~isempty(r)
%         xres=x(c);
%         yres=y(r);
%         dis=((xres-x(k)).^2+(yres-y(k)).^2).^(1/2);
%         %ind4=find(dis==min(min(dis)));
%         Radius(k)=ceil(2^(-1/2)*min(min(dis)));
%     else
%     	Radius(k)=ceil(2^(-1/2)*min(xran,yran));
%     end
% end
% figure; plot((1:1:length(x)),Radius);

xran=size(image,2);
yran=size(image,1);
Number=800; 
thresholdh=0.8; %%-------ORIGIN 0.8
Rind=1; %the rate R increases
%thresholdl=0.8;

while length(x)>Number
k=1;
while k<=length(x)
    xcor=x(k);
    ycor=y(k);
    xmin=max(1,xcor-Rind);
    ymin=max(1,ycor-Rind);
    xmax=min(xran,xcor+Rind);
    ymax=min(yran,ycor+Rind); %% change Radious to Rind for every point
    temp3=R(ymin:ymax,xmin:xmax);
    temp3=temp3-thresholdh*R_result(k);
    ind=find(temp3>0); %%this ind coordinate is local 
    if length(ind)>1    
%         if R_result(k)<Rmax*0.95
        if k~=1 && k~=length(x)
            x=[x(1:(k-1));x((k+1):end)]; %%change to ';', before is ','!!!
            y=[y(1:(k-1));y((k+1):end)];
            R_result=[R_result(1:k-1),R_result((k+1):end)];
        else
            if k==1
            x=x(2:end);
            y=y(2:end);
            R_result=R_result(2:end);
            else
            x=x(1:end-1);
            y=y(1:end-1);
            R_result=R_result(1:end-1);
            end
         end
%         else
%          k=k+1;  
%         end
    else 
%         if R_result <= Rmax*thresholdl; % to moveout noise
%         if k~=1
%             x=[x(1:k-1),x((k+1):end)];
%             y=[y(1:k-1),y((k+1):end)];
%             R_result=[R_result(1:k-1),R_result((k+1):end)];
%         else
%             x=x(2:end);
%             y=y(2:end);
%             R_result=R_result(2:end);
%         end
%         else
%         k=k+1;
%         end
          k=k+1;
    end
    
%     if k<length(x)&& (mod(x(k),feature_width)==1 || mod(y(k),feature_width) ==1 ...
%             || x(k)==xran || y(k)==yran)
%         
%         if k~=1 && k~=length(x)
%             x=[x(1:k-1),x((k+1):end)];
%             y=[y(1:k-1),y((k+1):end)];
%             R_result=[R_result(1:k-1),R_result((k+1):end)];
%         else
%             if k==1
%             x=x(2:end);
%             y=y(2:end);
%             R_result=R_result(2:end);
%             else
%             x=x(1:end-1);
%             y=y(1:end-1);
%             R_result=R_result(1:end-1);
%             end
%         end
%     else
%         k=k+1;
%     end % to rid of the arrays on the right            
end

%%to extinguish the arrays in flat area, use number restriction
%%renew the list everytime
%     for n=1:1:length(x)
%         temp5=R_result-R_result(n);
%         ind3=find(temp5>0);
%     if ~isempty(ind3)
%         xres=x(ind3);
%         yres=y(ind3);
%         dis=((xres-x(n)).^2+(yres-y(n)).^2).^(1/2);
%     %ind4=find(dis==min(min(dis)));
%     	  Radius(n)=ceil(Rind*2^(-1/2)*min(min(dis)));
%     else
%         Radius(n)=ceil(Rind*2^(-1/2)*min(xran,yran));
%     end
%     end
%         figure; plot((1:1:length(Radius)),Radius);
     Rind=Rind+1; %---------attention in using this
    
end

figure; imshow(image); hold on;
scatter(x,y); hold off;
figure; imshow(R); hold on;
scatter(x,y); hold off;

figure;
end

