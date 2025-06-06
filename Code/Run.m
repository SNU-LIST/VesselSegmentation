%%% Vessel segmentation algorithm for chi-separation
% 
% Input
%  - r2star: Hz unit
%  - x_para, x_dia: ppm unit (generated by chi-separation algorithm - https://github.com/SNU-LIST/chi-separation)
%  - brainMask 
%  - CSF_mask (optional)

%  - params
%    (1) params for vessel enhancement filter (MFAT, Default; https://github.com/Haifafh/MFAT)
%    (2) params for Seed generation (Step 1)
%       - alpha: Threshold for large vessel
%       - beta: Threshold for small vessel
%       - mipSlice: # of slices covering 16 mm (for 1 mm isotropic data, 16 slices)
%       - overlap: # of overlapped slices (for 1 mm isotropic data, 8 slices)
%    (3) params for Region growing (Step 2)
%       - Gamma 1, 2: Region growing limit -> limit = [Gamma 1, Gamma 2]
%    (4) params for non-vessel structure removal (Step 3)
%       - Aniso_Thresh: Anisotropy Threshold

% Output
%  - vesselMask_para, vesselMask_dia: vessel masks for x_para and x_dia

% T. Kim, S. Ji, K. Min, M. Kim, J. Youn, C. Oh, J. Kim, and J. Lee
% Vessel segmentation for X-separation (chi-separation)
% arXiv, 2025
% https://arxiv.org/abs/2502.01023

% Laboratory for Imaging Science and Technology
% Seoul National University
% email: sakkar2@snu.ac.kr
% Last modified 25.02.05

clear all;
close all;

%% Load data
load('Your Data path')

%% Set parameters
% (1) params for vessel enhancement filter (MFAT, Default)
params.tau = 0.02; params.tau2 = 0.35; params.D = 0.3;
params.spacing = voxelSize;
params.scales = 4; params.sigmas = [0.25,0.5,0.75,1];
params.whiteondark = true;

% (2) params for Seed Generation (Step 1)
params.alpha = 2; % Threshold for large vessel seeds
params.beta = 1; % Threshold for small vessel seeds
params.mipSlice = round(16 / params.spacing(3) / 2) * 2;
params.overlap = params.mipSlice / 2;

% (3) params for Region Growing and non-vessel structure removal (Step 2 & 3)
params.limit = [0.5, -0.5]; %% gamma1 and gamma2
params.Aniso_Thresh = 0.0012;
params.similarity = 0.5; % see (Eq. 3)

%% Run
r2star = r2star .* brainMask; r2star(r2star < 0) = 0;
x_para = x_para .* brainMask; x_para(x_para < 0) = 0;
x_dia = x_dia .* brainMask; x_dia(x_dia < 0) = 0;

seedInput.img1 = r2star; seedInput.img2 = x_para .* x_dia;
baseInput.img1 = x_para; baseInput.img2 = x_dia;

% If there is a CSF mask
[paraMask_init, diaMask_init, homogeneityMeasure_p, homogeneityMeasure_d] = ...
    vesselSegmentation_Chiseparation(seedInput, baseInput, brainMask, min(brainMask, 1 - CSF_mask), params);

% else
% [paraMask_init, diaMask_init, homogeneityMeasure_p, homogeneityMeasure_d] = ...
%     vesselSegmentation_Chiseparation(seedInput, baseInput, brainMask, brainMask, params);

vesselMask_para = filterVesselsByAnisotropy(paraMask_init, homogeneityMeasure_p, params.Aniso_Thresh);
vesselMask_dia  = filterVesselsByAnisotropy(diaMask_init, homogeneityMeasure_d, params.Aniso_Thresh);

