function [norm_data] = normaliseData(aData)
    
    headrs = fieldnames(aData);
    
    if isstruct(aData) && ismember('time',headrs)
        
        strt = min(aData.time);
        stp = max(aData.time);
        d_rate = (stp-strt)/(length(aData.time)-1);
        
        for iter = 2:length(headrs)
            norm_data.(headrs{iter}) = interp1(strt:d_rate:stp,aData.(headrs{iter}),...
                strt:((stp-strt)/100):stp,'spline');
        end
    else
        disp('Data not normalised. Input must be of type struct')
    end

end