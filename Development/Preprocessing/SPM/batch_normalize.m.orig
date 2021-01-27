% The files should have their origin set to the anterior commissure
% of the brain.
imageSourceFolder = './spm_indir/**/';

imageSourceFiles = dir(string(imageSourceFolder) + '*.nii');

nrun = length(imageSourceFiles);
jobfile = {'./batch_normalize_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(2, nrun);

for crun = 1:nrun
    
    imagePath = fullfile(imageSourceFiles(crun).folder, imageSourceFiles(crun).name);

    imagePath = strcat(imagePath, ',1');
    
    inputs{1, crun} = cellstr(imagePath);
    inputs{2, crun} = cellstr(imagePath);

end

spm('defaults', 'PET');
spm_jobman('run', jobs, inputs{:});
