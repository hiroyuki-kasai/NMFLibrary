function [tfr,t,f] = tfrstft(x,t,N,h,trace);
%TFRSTFT Short time Fourier transform.
%	[TFR,T,F]=TFRSTFT(X,T,N,H,TRACE) computes the short-time Fourier 
%	transform of a discrete-time signal X. 
% 
%	X     : signal.
%	T     : time instant(s)          (default : 1:length(X)).
%	N     : number of frequency bins (default : length(X)).
%	H     : frequency smoothing window, H being normalized so as to
%	        be  of unit energy.      (default : Hamming(N/4)). 
%	TRACE : if nonzero, the progression of the algorithm is shown
%	                                 (default : 0).
%	TFR   : time-frequency decomposition (complex values). The
%	        frequency axis is graduated from -0.5 to 0.5.
%	F     : vector of normalized frequencies.
%
%	Example :
%	 sig=[fmconst(128,0.2);fmconst(128,0.4)]; tfr=tfrstft(sig);
%	 subplot(211); imagesc(abs(tfr));
%	 subplot(212); imagesc(angle(tfr));
%



[xrow,xcol] = size(x);
if (nargin < 1),
 error('At least 1 parameter is required');
elseif (nargin <= 2),
 N=xrow;
end;

hlength=floor(N/4);
hlength=hlength+1-rem(hlength,2);

if (nargin == 1),
 t=1:xrow; h = tftb_window(hlength); trace=0;
elseif (nargin == 2) | (nargin == 3),
 h = tftb_window(hlength); trace = 0;
elseif (nargin == 4),
 trace = 0;
end;

if (N<0),
 error('N must be greater than zero');
end;
[trow,tcol] = size(t);
if (xcol~=1),
 error('X must have one column');
elseif (trow~=1),
 error('T must only have one row'); 
end; 

[hrow,hcol]=size(h); Lh=(hrow-1)/2; 
if (hcol~=1)|(rem(hrow,2)==0),
 error('H must be a smoothing window with odd length');
end;

h=h/norm(h);

tfr= zeros (N,tcol) ;  

for icol=1:tcol,
 ti= t(icol); tau=-min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,xrow-ti]);
 indices= rem(N+tau,N)+1; 
 tfr(indices,icol)=x(ti+tau,1).*conj(h(Lh+1+tau));
end;
tfr=fft(tfr); 

if (nargout==0),
 tfrqview(abs(tfr).^2,x,t,'tfrstft',h);
elseif (nargout==3),
 if rem(N,2)==0, 
  f=[0:N/2-1 -N/2:-1]'/N;
 else
  f=[0:(N-1)/2 -(N-1)/2:-1]'/N;  
 end;
end;