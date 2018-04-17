function [ Vo ] = normalization( Vo_org, max_gray_level )
%NOMALIZATION この関数の概要をここに記述
%   詳細説明をここに記述

    Vo_max = max(max(Vo_org));
    
    Vo = Vo_org * (max_gray_level / Vo_max);
end

