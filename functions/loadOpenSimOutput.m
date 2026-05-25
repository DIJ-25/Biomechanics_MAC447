function [Angles] = loadOpenSimOutput(aFolder)
    
    %Retrieve .mot file in selected directory
    reqFiles = dir(fullfile(aFolder,'*.mot'));
    if isempty(reqFiles)
        reqFiles = dir(fullfile(aFolder,'*.sto'));
    end
    
    if ~isempty(reqFiles)
        try
            %load the OpenSim generated joint angles
            Data = importdata(fullfile(reqFiles(1).folder,reqFiles(1).name),'\t');
            d_headers = Data.colheaders;
            
            for iter = 1:length(d_headers)
                Angles.(d_headers{iter}) = Data.data(:,iter);
            end

        catch
            disp('IK Data file could not be loaded');
        end

    else
        disp('Selected IK folder does not contain required .mot file');
    end
end