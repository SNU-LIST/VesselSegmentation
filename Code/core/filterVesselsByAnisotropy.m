function out = filterVesselsByAnisotropy(vessel_mask, homogeneityMeasure, thred)
    % This function removes falsely included vessels based on the homogeneity criteria.
    % It uses Hessian-based measures to determine the homogeneity of vessel regions
    % and suppresses regions that do not meet the criteria.
    
    % Initialization of output mask
    out = vessel_mask;
    
    CC = bwconncomp(vessel_mask, 6);
    for k = 1:CC.NumObjects
        
        homogeneityValue = mean(mink(homogeneityMeasure(CC.PixelIdxList{k}), round(length(CC.PixelIdxList{k}) * 0.75)));
        if homogeneityValue < thred
            out(CC.PixelIdxList{k}) = 0;
        end
        
    end

end