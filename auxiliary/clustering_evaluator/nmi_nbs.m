function v = nmi_nbs(label, result)
% Nomalized mutual information
% Written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
assert(length(label) == length(result));

label = label(:);
result = result(:);

n = length(label);

label_unique = unique(label);
result_unique = unique(result);

% check the integrity of result
if length(label_unique) ~= length(result_unique)
    error('The clustering result is not consistent with label.');
end;

c = length(label_unique);

% distribution of result and label
Ml = double(repmat(label,1,c) == repmat(label_unique',n,1));
Mr = double(repmat(result,1,c) == repmat(result_unique',n,1));
Pl = sum(Ml)/n;
Pr = sum(Mr)/n;

% entropy of Pr and Pl
Hl = -sum( Pl .* log2( Pl + eps ) );
Hr = -sum( Pr .* log2( Pr + eps ) );


% joint entropy of Pr and Pl
% M = zeros(c);
% for I = 1:c
% 	for J = 1:c
% 		M(I,J) = sum(result==result_unique(I)&label==label_unique(J));
% 	end;
% end;
% M = M / n;
M = Ml'*Mr/n;
Hlr = -sum( M(:) .* log2( M(:) + eps ) );

% mutual information
MI = Hl + Hr - Hlr;

% normalized mutual information
v = sqrt((MI/Hl)*(MI/Hr)) ;