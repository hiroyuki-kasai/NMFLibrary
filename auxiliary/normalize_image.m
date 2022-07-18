function Vo = normalize_image( Vo_org, max_gray_level )

    Vo_max = max(max(Vo_org));
    Vo = Vo_org * (max_gray_level / Vo_max);
end