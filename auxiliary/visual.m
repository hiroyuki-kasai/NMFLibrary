function visual( W, mag, cols, ysize )
% visual - display a basis for image patches
%
% W        the basis, with patches as column vectors
% mag      magnification factor
% cols     number of columns (x-dimension of map)
% ysize    [optional] height of each subimage
%
    
% Is the basis non-negative    
if min(W(:))>=0,
    % Make zeros white and positive values darker, as in earlier NMF papers
    W = -W;
    maxi=0;
    mini=min(W(:));
    bgval = mini/2;
else
    % Make zero gray, positive values white, and negative values black
    maxi = max(max(abs(W)));
    mini = -maxi;
    bgval = maxi;
end
    
% Get maximum absolute value (it represents white or black; zero is gray)

% This is the side of the window
if ~exist('ysize'), ysize = sqrt(size(W,1)); end
xsize = size(W,1)/ysize;

% Helpful quantities
xsizem = xsize-1;
xsizep = xsize+1;
ysizem = ysize-1;
ysizep = ysize+1;
rows = ceil(size(W,2)/cols);

% Initialization of the image
I = bgval*ones(2+ysize*rows+rows-1,2+xsize*cols+cols-1);

for i=0:rows-1
  for j=0:cols-1
    
    if i*cols+j+1>size(W,2)
      % This leaves it at background color
      
    else
      % This sets the patch
      I(i*ysizep+2:i*ysizep+ysize+1, ...
	j*xsizep+2:j*xsizep+xsize+1) = ...
          reshape(W(:,i*cols+j+1),[ysize xsize]);
    end
    
  end
end

% Make a black border
I(1,:) = mini;
I(:,1) = mini;
I(end,:) = mini;
I(:,end) = mini;

I = imresize(I,mag);

colormap(gray(256));
iptsetpref('ImshowBorder','tight'); 
subplot('position',[0,0,1,1]);
imshow(I,[mini maxi]);
truesize;  
drawnow
