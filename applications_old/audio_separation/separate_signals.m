function [recMusic, recSpeech] = separate_signals(V,Bm,Bs,niter)

for i=1:niter
    W=pinv([Bm,Bs])*V;
    Wm=W(1:size(Bm,2),:);
    Ws=W((size(Bm,2)+1):end,:);
    
    recMusic=Bm*Wm;
    recSpeech=Bs*Ws;
end
S= (recMusic.^2) ./ (recMusic.^2+recSpeech.^2);
recMusic=S.*V;
recSpeech=(1-S).*V;
% Signal Seperation algorithm from
% https://www.researchgate.net/publication/326914780_Supervised_sound_source_separation_using_Nonnegative_Matrix_Factorization
% page 15-16
end