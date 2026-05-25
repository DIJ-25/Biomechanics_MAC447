function [Angles] = loadPiG(aFolder)

    %Retrieve .mat file in selected directory
    reqFiles = dir(fullfile(aFolder,'*.mat'));
    if ~isempty(reqFiles)
        try
            %load the Plugin Gait generated joint angles
            Data = load(fullfile(reqFiles(1).folder,reqFiles(1).name));
            Angles = Data.Angles;
        catch
            disp('Data file could not be loaded');
        end
        
    else
        disp('Selected folder does not contain required file');
    end
end