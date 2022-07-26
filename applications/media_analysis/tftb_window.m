function h=tftb_window(N,name,param,param2);
%tftb_window	Window generation.
%	H=tftb_window(N,NAME,PARAM,PARAM2)
%	yields a window of length N with a given shape.
%
%	N      : length of the window
%	NAME   : name of the window shape (default : Hamming)
%	PARAM  : optional parameter
%	PARAM2 : second optional parameters
%
%	Possible names are :
%	'Hamming', 'Hanning', 'Nuttall',  'Papoulis', 'Harris',
%	'Rect',    'Triang',  'Bartlett', 'BartHann', 'Blackman'
%	'Gauss',   'Parzen',  'Kaiser',   'Dolph',    'Hanna'.
%	'Nutbess', 'spline',  'Flattop'
%
%	For the gaussian window, an optionnal parameter K
%	sets the value at both extremities. The default value is 0.005
%
%	For the Kaiser-Bessel window, an optionnal parameter
%	sets the scale. The default value is 3*pi.
%
%	For the Spline windows, h=tftb_window(N,'spline',nfreq,p)
%	yields a spline weighting function of order p and frequency
%	bandwidth proportional to nfreq.
%
%       Example: 
%        h=tftb_window(256,'Gauss',0.005); 
%        plot(0:255, h); axis([0,255,-0.1,1.1]); grid
%


if (nargin==0), error ( 'at least 1 parameter is required' ); end;
if (N<=0), error('N should be strictly positive.'); end;
if (nargin==1), name= 'Hamming'; end ;
name=upper(name);
if strcmp(name,'RECTANG') | strcmp(name,'RECT'), 
 h=ones(N,1);
elseif strcmp(name,'HAMMING'),
 h=0.54 - 0.46*cos(2.0*pi*(1:N)'/(N+1));
elseif strcmp(name,'HANNING') | strcmp(name,'HANN'),
 h=0.50 - 0.50*cos(2.0*pi*(1:N)'/(N+1));
elseif strcmp(name,'KAISER'),
 if (nargin==3), beta=param; else beta=3.0*pi; end;
 ind=(-(N-1)/2:(N-1)/2)' *2/N; beta=3.0*pi;
 h=besselj(0,j*beta*sqrt(1.0-ind.^2))/real(besselj(0,j*beta));
elseif strcmp(name,'NUTTALL'),
 ind=(-(N-1)/2:(N-1)/2)' *2.0*pi/N;
 h=+0.3635819 ...
   +0.4891775*cos(    ind) ...
   +0.1363995*cos(2.0*ind) ...
   +0.0106411*cos(3.0*ind) ;
elseif strcmp(name,'BLACKMAN'),
 ind=(-(N-1)/2:(N-1)/2)' *2.0*pi/N;
 h= +0.42 + 0.50*cos(ind) + 0.08*cos(2.0*ind) ;
elseif strcmp(name,'HARRIS'),
 ind=(1:N)' *2.0*pi/(N+1);
 h=+0.35875 ...
   -0.48829 *cos(    ind) ...
   +0.14128 *cos(2.0*ind) ...
   -0.01168 *cos(3.0*ind);
elseif strcmp(name,'BARTLETT') | strcmp(name,'TRIANG'),
 h=2.0*min((1:N),(N:-1:1))'/(N+1);
elseif strcmp(name,'BARTHANN'),
 h=  0.38 * (1.0-cos(2.0*pi*(1:N)/(N+1))') ...
   + 0.48 * min((1:N),(N:-1:1))'/(N+1);
elseif strcmp(name,'PAPOULIS'),
 ind=(1:N)'*pi/(N+1); h=sin(ind);
elseif strcmp(name,'GAUSS'),
 if (nargin==3), K=param; else K=0.005; end;
 h= exp(log(K) * linspace(-1,1,N)'.^2 );
elseif strcmp(name,'PARZEN'),
 ind=abs(-(N-1)/2:(N-1)/2)'*2/N; temp=2*(1.0-ind).^3;
 h= min(temp-(1-2.0*ind).^3,temp);
elseif strcmp(name,'HANNA'),
 if (nargin==3), L=param; else L=1; end;
 ind=(0:N-1)';h=sin((2*ind+1)*pi/(2*N)).^(2*L);
elseif strcmp(name,'DOLPH') | strcmp(name,'DOLF'),
 if (rem(N,2)==0), oddN=1; N=2*N+1; else oddN=0; end;
 if (nargin==3), A=10^(param/20); else A=1e-3; end;
 K=N-1; Z0=cosh(acosh(1.0/A)/K); x0=acos(1/Z0)/pi; x=(0:K)/N; 
 indices1=find((x<x0)|(x>1-x0));
 indices2=find((x>=x0)&(x<=1-x0));
 h(indices1)= cosh(K*acosh(Z0*cos(pi*x(indices1))));
 h(indices2)= cos(K*acos(Z0*cos(pi*x(indices2))));
 h=fftshift(real(ifft(A*real(h))));h=h'/h(K/2+1);
 if oddN, h=h(2:2:K); end;
elseif strcmp(name,'NUTBESS'),
 if (nargin==3), beta=param; nu=0.5; 
 elseif (nargin==4), beta=param; nu=param2;
 else beta=3*pi; nu=0.5;
 end;
 ind=(-(N-1)/2:(N-1)/2)' *2/N; 
 h=sqrt(1-ind.^2).^nu .* ...
   real(besselj(nu,j*beta*sqrt(1.0-ind.^2)))/real(besselj(nu,j*beta));
elseif strcmp(name,'SPLINE'),
 if (nargin < 3),
  error('Three or four parameters required for spline windows');
 elseif (nargin==3),
  nfreq=param; p=pi*N*nfreq/10.0;
 else nfreq=param; p=param2;
 end;
  ind=(-(N-1)/2:(N-1)/2)'; 
  h=sinc((0.5*nfreq/p)*ind) .^ p;
elseif strcmp(name,'FLATTOP'),
 ind=(-(N-1)/2:(N-1)/2)' *2.0*pi/(N-1);
 h=+0.2810639 ...
   +0.5208972*cos(    ind) ...
   +0.1980399*cos(2.0*ind) ;
else error('unknown window name');
end;