function [paraMask, diaMask, homo_p, homo_d] = vesselSegmentation_Chiseparation(seedInput,baseInput, mask, CSF_mask, options)
    %% Step1. Seed Generation
    totalSeed = seedGeneration(seedInput,mask,CSF_mask,options);
    
    %% Step2. Vessel geometric characteristics-guided region growing
    [paraMask, diaMask, homo_p, homo_d] = regionGrowing_vesselGeometryCharacteristics(baseInput, totalSeed, mask, [options.limit(1:2), options.similarity], options);
    paraMask = removeCluster(paraMask, 2);
    diaMask = removeCluster(diaMask, 2); 
    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step1. Seed Generation main functions
function totalSeed = seedGeneration(input,mask,cmask,options)
    
    %% Seed Generation for large Vessels
    
    % 0. Pre-process image boundaries to avoid high frequency components between tissue and non-tissue regions.
    preprocessedImg = fill_boundary(input.img1, mask, cmask);

    % 1. Suppress the background noise and normalize the image.
    img_bgSuppress = backgroundSuppress(preprocessedImg) .* imdilate(mask, strel('sphere',3)); %imdilate(mask, strel('sphere',3));
       
    % 2. Calculate MFAT to estimate vessel probability across the image.
    [~, MFAT] = FractionalAnisotropicTensor3D(img_bgSuppress, cmask, options.scales, options);
    
    % 3. Threshold the MFAT values to identify likely vessel regions.
    seed1 = zeros(size(input.img1));
    vesselThreshold = mean(MFAT(cmask == 1)) + options.alpha * std(MFAT(cmask == 1));
    seed1(MFAT > vesselThreshold) = 1;
    
    
    %% Seed Generation for small Vessels
    
    if isfield(input, 'img2')
        preprocessedImg = fill_boundary(input.img2, mask, cmask);
    end
    
    [xres, yres, zres] = size(input.img1);
    cur_bottom = 1; cur_top = cur_bottom + options.mipSlice - 1;

    totalSeed = seed1;
    
    % Process slices with overlapping MIPs
    while cur_top ~= zres
        
        if cur_top > zres
            cur_top = zres;
        end
        slices = cur_bottom:cur_top;
        
        if options.whiteondark
            [mip, idx] = max(preprocessedImg(:,:,slices), [], 3);
        else
            [mip, idx] = min(preprocessedImg(:,:,slices), [], 3);
        end
        seedMIP = zeros(size(mip));
        brainMIP = zeros(size(mip));
        totalMIP = zeros(size(mip));
        for x = 1:size(idx,1)
            for y = 1:size(idx,2)
                seedMIP(x,y) = seed1(x,y,slices(1) + idx(x,y) - 1);
                brainMIP(x,y) = mask(x,y,slices(1) + idx(x,y) - 1);
                totalMIP(x,y) = cmask(x,y,slices(1) + idx(x,y) - 1);
            end
        end
              
        mip = normalize(mip) .* (1 - seedMIP);
        mip(~isfinite(mip)) = 0;
        mip = inpaintCoherent(mip, logical(max(seedMIP, 1 - brainMIP)));
        MFAT_2D = FractionalAnisotropicTensor2D(mip, 1, options);
        
        thred = mean(MFAT_2D(totalMIP == 1 & seedMIP == 0)) + options.beta * std(MFAT_2D(totalMIP == 1 & seedMIP == 0));
        MFAT_2D(MFAT_2D > thred) = 1;
        MFAT_2D(MFAT_2D ~= 1) = 0;
        MFAT_2D = removeCluster(MFAT_2D, 2) .* brainMIP;
        
        temp = zeros(xres,yres,(cur_top - cur_bottom + 1));

        for i = 1:xres
            for j = 1:yres

                if MFAT_2D(i,j) == 1
                    temp(i,j,idx(i,j)) = 1;
                end

            end
        end

        totalSeed(:,:,cur_bottom:cur_top) = max(totalSeed(:,:,cur_bottom:cur_top),temp);
             
        if cur_top == zres
            break;
        end
        
        cur_bottom = cur_bottom + options.overlap;
        cur_top = cur_top + options.overlap;
    end
    
    %%
    totalSeed = removeCluster(totalSeed, 1) .* cmask;
    
end

function out = fill_boundary(img, boundaryMask, exclusionMask)
    % FILL_BOUNDARY Fill boundaries in the given image.
    % img: Input image.
    % boundaryMask: Mask marking regions to be preserved.
    % exclusionMask: Mask marking regions to be filled.
    %
    % Returns:
    % out: Image with boundaries filled.
    
    out = zeros(size(img));
    for sliceIdx = 1:size(img,3)
        out(:,:,sliceIdx) = inpaintCoherent(img(:,:,sliceIdx) .* boundaryMask(:,:,sliceIdx), ~exclusionMask(:,:,sliceIdx));
    end
    out(isnan(out)) = 0; % Ensure no NaN values in the output.
end

function out = backgroundSuppress(img)

    % Transform the image to the frequency domain using custom FFT function
    kspace = fft3c(img, 11);
    
    % Create an inverse hamming filter
    filter = inverseHammingFilter(80, 80, 80, size(img));
    
    % Apply the filter in the frequency domain
    filtered = kspace .* filter;
    
    % Transform the filtered k-space back to spatial domain
    bgSuppressedImg = real(ifft3c(filtered, 11));
    
    % Normalize the background suppressed image
    bgSuppressedImg_ = normalize(bgSuppressedImg);
    out = bgSuppressedImg_;
    
end

function [out, vesselness] = FractionalAnisotropicTensor3D(I, mask, scales, options)
% Calculates the vesselness Fractional anisotropy tensor of a 3D input image
%   - Fractional anisotropy tensor equation

% inputs,
%   I : 3D image
%   scales : vector of scales on which the vesselness is computed
%   options: struct containing the following fields
%      - tau, tau2: cutoff thresholding related to eigenvalues
%      - D: the step size of solution evolution
%      - whiteondark: parameter for Hessian features computation
% outputs,
%   out: final vesselness response over scales

% example:
%   out = FractionalAnisotropicTensor3D(I, [.5:.5:3], options);

% Function written by Haifa F. Alhasson, Durham University (Dec 2017)
% Modified for MFAT by Taechang Kim, Seoul National University, (~~ 2023)
% Based on code by T. Jerman, University of Ljubljana (October 2014)

%% Parameter Setting
tau = options.tau;
tau2 = options.tau2;
D = options.D;
whiteondark = options.whiteondark;

I(~isfinite(I)) = 0;

%% Preprocessing 
I = single(I);

%% Enhancement 
for j = 1:scales
    %% Compute Eigen-values
    I_ = imgaussian(I, options.sigmas(j));
    [~, Lambda2, Lambda3, ~] = computeHessianFeatures(I_, mask, whiteondark);
    
    % Adjust Lambda3 based on thresholds
    Lambda3M = adjustLambda(Lambda3, tau);
    Lambda4 = adjustLambda(Lambda3, tau2);
    
    %% Compute Fractional Anisotropy Tensor equation
    response = computeFractionalAnisotropy(Lambda2, Lambda3M, Lambda4);
    
    %% Apply restrictions
    response = applyRestrictions(response, Lambda2, Lambda3M);
    
    %% Update vesselness
    if j == 1
        vesselness = response;
    else   
        vesselness = vesselness + D .* tanh(abs(response) - D);
        vesselness = max(vesselness, response);     
    end
    vesselness = min(max(vesselness, 0), 1);
    
    % Assign response to output for current scale
    out(:,:,:,j) = response;
end

% Normalize final vesselness
vesselness = normalize(vesselness);

end

function out = FractionalAnisotropicTensor2D(I,scales,options)
% calculates the vesselness Fractional anisotropy tensor of a 2D
% input image
%   -Fractional anisotropy tensor equation:
%    Reference :
%     Hansen, Charles D., and Chris R. Johnson. Visualization handbook. Academic Press, 2011.�? APA    
% inputs,
%   I : 2D image
%   sigmas : vector of scales on which the vesselness is computed
%   spacing : input image spacing resolution - during hessian matrix 
%       computation, the gaussian filter kernel size in each dimension can 
%       be adjusted to account for different image spacing for different
%       dimensions            
%   tau,tau 2: cutoff thresholding related to eignvlaues.
%   D  : the step size of soultion evolution.
%
% outputs,
%   out: final vesselness response over scales sigmas
%
% Example:

% out = FractionalIstropicTensor(I, sigmas,spacing,tau ,tau2,D )
% sigmas = [1:1:3];
% out = FractionalIstropicTensor(I, sigmas, 1,0.03,0.3,0.27)
%
% Function written by Haifa F. Alhasson , Durham University (Dec 2017)
% Based on code by T. Jerman, University of Ljubljana (October 2014)
%%
tau = options.tau;
tau2 = options.tau2;
D = options.D;
whiteondark = options.whiteondark;

% spacing = [spacing spacing]; 
% verbose = 1;
%% preprocessing 
I = single(I);
%% Enhancement 
vesselness = zeros(size(I));
for j = 1:scales
%     if verbose
%         %disp(['Current filter scale (sigma): ' num2str(sigmas(j)) ]);
%     end
    %% (1) Eigen-values
    [~, Lambda2] = imageEigenvalues(I(:,:,:,j),whiteondark);  
    %% filter response at current scale from RVR
    Lambda3 = Lambda2;
    Lambda3(Lambda3<0 & Lambda3 >= tau .* min(Lambda3(:)))=  tau.* min(Lambda3(:));
    %% New filter response
    Lambda4 = Lambda2;
    Lambda4(Lambda4<0 & Lambda4 >= tau2 .* min(Lambda4(:)))= tau2.* min(Lambda4(:));
    %% (2) Fractional Anisotropy Tensor equation: 
    % Mean Eigen-value (LambdaMD):
    LambdaMD = abs(abs(Lambda2)+ abs(Lambda3) +abs(Lambda4))./3;
    % response at current scale 
    response = sqrt((((abs(Lambda2))-abs(LambdaMD)).^2+(abs((Lambda3))-abs(LambdaMD)).^2+(abs(Lambda4)-abs(LambdaMD)).^2)) ./sqrt(((abs(Lambda2))).^2+((abs(Lambda3))).^2+(abs(Lambda4)).^2);    
    response  = imcomplement(sqrt(3./2).*response);
    %% (3) Post-processing: targeting gaussian noise in the background
    x = Lambda3 - Lambda2;
    response(x == min(x(:))) = 1;
    response(x < max(x(:))) = 0; 
    response(Lambda2 > x) = 0;
%     response(~isfinite(response)) = 0;
%     response(Lambda3 > x) = 0;
%     response(~isfinite(response)) = 0;
%     response(x <= max(x(:))) = 1; 
    response(Lambda3 > x) = 0;
    response(Lambda2>=0) = 0;
    response(Lambda3>=0) = 0;   
    response(~isfinite(response)) = 0;   

    %% (4) Update vesselness & I
    if(j==1)
        vesselness = response;
    else  
        vesselness = vesselness + D .* tanh( response - D);
        vesselness = max(vesselness,response);
    end
    % Normalize vessleness
     vesselness = min(max(vesselness, 0), 1);
    
    clear Lambda2 Lambda3 Lambda4 LambdaMD
end
out = vesselness ./ max(vesselness(:));  
out(out < 1e-2) = 0;

end

function [out1, out2, homogeneityMeasure1, homogeneityMeasure2] = regionGrowing_vesselGeometryCharacteristics(baseInput, seed, mask, limit, options)
    
    % Compute Hessian features from the image
    [~, lambda2, lambda3, orientedFeatures] = computeHessianFeatures(baseInput.img1, mask, options.whiteondark);    
    homogeneityMeasure1 = normalize((abs(lambda2) .* abs(lambda3)));
    
    % Perform region growing based on orientation and intensity.
    if options.whiteondark
        out1_ = regionGrowing_byOrientedIntensity_white(baseInput.img1, seed, mask, orientedFeatures, homogeneityMeasure1, limit, options);
    else
        out1_ = regionGrowing_byOrientedIntensity_black(baseInput.img1, seed, mask, orientedFeatures, homogeneityMeasure1, limit, options);
    end
    
    [~, lambda2, lambda3, orientedFeatures] = computeHessianFeatures(baseInput.img2, mask, options.whiteondark);    
    homogeneityMeasure2 = normalize((abs(lambda2) .* abs(lambda3)));
    
    % Perform region growing based on orientation and intensity.
    if options.whiteondark
        out2_ = regionGrowing_byOrientedIntensity_white(baseInput.img2, seed, mask, orientedFeatures, homogeneityMeasure2, limit, options);
    else
        out2_ = regionGrowing_byOrientedIntensity_black(baseInput.img2, seed, mask, orientedFeatures, homogeneityMeasure2, limit, options);
    end

    out1 = out1_ .* mask;
    out2 = out2_ .* mask;
end

function out = regionGrowing_byOrientedIntensity_white(img, seed, mask, orientedFeatures, homogeneityMeasure, limit, options)
    
    connectedComponents = bwconncomp(seed,26); 
    numPixels = cellfun(@numel, connectedComponents.PixelIdxList);
    [~,idxs] = sort(numPixels,2,'descend');
    [~, MFAT] = FractionalAnisotropicTensor3D(img, mask, options.scales, options);    
    
    global_mean = mean(img(seed == 1));
    global_std = std(img(seed == 1));
    [xres, yres, zres] = size(seed);
    
    x_diff = [1,-1,0,0,0,0];
    y_diff = [0,0,1,-1,0,0];
    z_diff = [0,0,0,0,1,-1];    
    
    num = 1;
    out = zeros(size(img));
    
    % Iteratively grow the region
    while length(numPixels) > num
       
        currentRegion = zeros(size(img));
        currentRegion(connectedComponents.PixelIdxList{idxs(num)}) = 1;
        out(currentRegion == 1) = 1;
        
        [px, py, pz] = ind2sub(size(img),connectedComponents.PixelIdxList{idxs(num)});
        queue = [px, py, pz];
        
        % Use queue to search for voxels to grow
        while ~isempty(queue)
            
            xv = queue(1,1);
            yv = queue(1,2);
            zv = queue(1,3);
            queue(1,:) = [];   
            cur_dir = squeeze(orientedFeatures(xv,yv,zv,:));
            
            for n = 1:6
                
                xv_ = xv + x_diff(n);
                yv_ = yv + y_diff(n);
                zv_ = zv + z_diff(n);
                
                if xv_ ~= 0 && yv_ ~= 0 && zv_ ~= 0 && xv_ ~= xres + 1 && yv_ ~= yres + 1 && zv_ ~= zres + 1
                    if  ~out(xv_,yv_,zv_) && mask(xv_,yv_,zv_)
                         if img(xv_,yv_,zv_) > global_mean + limit(1) * global_std
                             queue(end+1,:) = [xv_,yv_,zv_];
                             out(xv_,yv_,zv_) = 1;
                         elseif img(xv_,yv_,zv_) > global_mean + limit(2) * global_std
                             target_dir = [orientedFeatures(xv_,yv_,zv_,1), orientedFeatures(xv_,yv_,zv_,2), orientedFeatures(xv_,yv_,zv_,3)];
                             similarity = dot(cur_dir, target_dir)/(norm(cur_dir) * norm(target_dir));


                             if img(xv_,yv_,zv_) > img(xv,yv,zv)
                                 ratio = img(xv,yv,zv)/img(xv_,yv_,zv_);
                             else
                                 ratio = img(xv_,yv_,zv_)/img(xv,yv,zv);
                             end

                             if ratio ~= 0 && MFAT(xv_,yv_,zv_) >=  limit(3)  * (1 - abs(similarity)) / ratio / (1 - exp((-1) * 10 * homogeneityMeasure(xv_,yv_,zv_)))
                                 queue(end+1,:) = [xv_,yv_,zv_];
                                 out(xv_,yv_,zv_) = 1;
                             end
                        end
                    end
                end
                
            end
            
        end
        num = num + 1;
        
    end
end

function out = removeCluster(img, n)
    
    CC = bwconncomp(img,26);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    out = img;
    
    for  i = 1:length(numPixels)
        if numPixels(i) <= n
            out(CC.PixelIdxList{i}) = 0;
        end
    end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Utils %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Lambda1, Lambda2, Lambda3, eigDirection] = computeHessianFeatures(V, mask, whiteOnDark)
    % This function computes Hessian-based features, including eigenvalues and eigenvectors.
    % Inputs:
    %   V          : Input 3D image
    %   whiteOnDark: Boolean flag to invert image (true if bright structures on dark background)

    % Calculate 3D hessian
    [Hxx, Hyy, Hzz, Hxy, Hxz, Hyz] = calculateHessian3D(V);

    % If dark structures on bright background, invert hessian values
    if ~whiteOnDark
        Hxx = -Hxx; Hxy = -Hxy; Hxz = -Hxz;
        Hyy = -Hyy; Hyz = -Hyz; Hzz = -Hzz;    
    end

    % Compute parameters to reduce computation
%     B1 = -(Hxx + Hyy + Hzz);
%     B2 = Hxx .* Hyy + Hxx .* Hzz + Hyy .* Hzz - Hxy .* Hxy - Hxz .* Hxz - Hyz .* Hyz;
%     B3 = Hxx .* Hyz .* Hyz + Hxy .* Hxy .* Hzz + Hxz .* Hyy .* Hxz - Hxx .* Hyy .* Hzz - Hxy .* Hyz .* Hxz - Hxz .* Hxy .* Hyz;
% 
%     T = getMaskForReducedComputation(B1, B2, B3);
%    clear B1 B2 B3;

    indices = find(mask == 1);    
    [Lambda1, Lambda2, Lambda3, eigDirection] = computeEigenValuesVectors(Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, indices);
    
end

function LambdaAdj = adjustLambda(Lambda, tau)
    LambdaAdj = Lambda;
    LambdaAdj(LambdaAdj < 0 & LambdaAdj >= tau .* min(LambdaAdj(:))) = tau .* min(LambdaAdj(:));
    LambdaAdj(LambdaAdj > 0) = 0;
end

function response = computeFractionalAnisotropy(Lambda2, Lambda3M, Lambda4)
    LambdaMD = (abs(Lambda2) + abs(Lambda3M) + abs(Lambda4)) ./ 3;
    response = sqrt((((abs(Lambda2))-abs(LambdaMD)).^2 + (abs(Lambda3M)-abs(LambdaMD)).^2 + (abs(Lambda4)-abs(LambdaMD)).^2) ./ (abs(Lambda2).^2 + abs(Lambda3M).^2 + abs(Lambda4).^2));    
    response = sqrt(3./2) .* response;
    response = imcomplement(response);
end

function response = applyRestrictions(response, Lambda2, Lambda3M)
    x = Lambda3M - Lambda2;
    response(x == max(x(:,:,:))) = 1; 
    response(Lambda3M > x) = 0;
    response(~isfinite(response)) = 0;
    response(Lambda2 >= 0) = 0;
    response(~isfinite(response)) = 0;
    response(Lambda3M >= 0) = 0;   
    response(~isfinite(response)) = 0;   
end

function T = getMaskForReducedComputation(B1, B2, B3)
    T = ones(size(B1));
    T(B1 <= 0) = 0;
    T(B2 <= 0 & B3 == 0) = 0;
    T(B1 > 0 & B2 > 0 & B1 .* B2 < B3) = 0;
end

function [Lambda1, Lambda2, Lambda3, eigDirection] = computeEigenValuesVectors(Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, indices)
    % Computes the eigenvalues and eigenvectors for Hessian values

    matrix_size = size(Hxx);
    
    % Initializing
    Lambda1 = zeros(matrix_size);
    Lambda2 = zeros(matrix_size);
    Lambda3 = zeros(matrix_size);
    Vx = zeros(matrix_size);
    Vy = zeros(matrix_size);
    Vz = zeros(matrix_size);
      
    Hxx = Hxx(indices);
    Hyy = Hyy(indices);
    Hzz = Hzz(indices);
    Hxz = Hxz(indices);
    Hyz = Hyz(indices);
    Hxy = Hxy(indices);   
      
    [Lambda1_, Lambda2_, Lambda3_, Vx_, Vy_, Vz_] =eig3volume(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz);
    
    Lambda1(indices) = Lambda1_;
    Lambda2(indices) = Lambda2_;
    Lambda3(indices) = Lambda3_;
    Vx(indices) = Vx_;
    Vy(indices) = Vy_;
    Vz(indices) = Vz_;
    eigDirection = cat(4, Vx, Vy, Vz);
    
    % Clean up noise
%    Lambda1(~isfinite(Lambda1) | abs(Lambda1) < 1e-4) = 0;
%    Lambda2(~isfinite(Lambda2) | abs(Lambda2) < 1e-4) = 0;
%    Lambda3(~isfinite(Lambda3) | abs(Lambda3) < 1e-4) = 0;
end

function [Lambda1, Lambda2] = imageEigenvalues(I,whiteondark)
% calculates the two eigenvalues for each voxel in a volume

% Calculate the 2D hessian
[Hxx, Hyy, Hxy] = Hessian2D(I);

% Correct for scaling
% c=sigma.^2;
% Hxx = c*Hxx; 
% Hxy = c*Hxy;
% Hyy = c*Hyy;

% correct sign based on brightness of structuress
if whiteondark == false
    c=-1;
    Hxx = c*Hxx; 
    Hxy = c*Hxy;
    Hyy = c*Hyy;   
end

% reduce computation by computing vesselness only where needed
% S.-F. Yang and C.-H. Cheng, �Fast computation of Hessian-based
% enhancement filters for medical images,� Comput. Meth. Prog. Bio., vol.
% 116, no. 3, pp. 215�225, 2014.
B1 = - (Hxx+Hyy);
B2 = Hxx .* Hyy - Hxy.^2;


T = ones(size(B1));
T(B1<0) = 0;
T(B2==0 & B1 == 0) = 0;

clear B1 B2;

indeces = find(T==1);

Hxx = Hxx(indeces);
Hyy = Hyy(indeces);
Hxy = Hxy(indeces);

% Calculate eigen values
[Lambda1i,Lambda2i]=eigvalOfHessian2D(Hxx,Hxy,Hyy);

clear Hxx Hyy Hxy;

Lambda1 = zeros(size(T));
Lambda2 = zeros(size(T));

Lambda1(indeces) = Lambda1i;
Lambda2(indeces) = Lambda2i;

% some noise removal
Lambda1(~isfinite(Lambda1)) = 0;
Lambda2(~isfinite(Lambda2)) = 0;

Lambda1(abs(Lambda1) < 1e-4) = 0;
Lambda2(abs(Lambda2) < 1e-4) = 0;

end

function [Lambda1,Lambda2]=eigvalOfHessian2D(Dxx,Dxy,Dyy)
% This function calculates the eigen values from the
% hessian matrix, sorted by abs value

% Compute the eigenvectors of J, v1 and v2
tmp = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);

% Compute the eigenvalues
mu1 = 0.5*(Dxx + Dyy + tmp);
mu2 = 0.5*(Dxx + Dyy - tmp);

% Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
check=abs(mu1)>abs(mu2);

Lambda1=mu1; Lambda1(check)=mu2(check);
Lambda2=mu2; Lambda2(check)=mu1(check);

end

function [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = calculateHessian3D(Volume)
%  This function Hessian3D filters the image with an Gaussian kernel
%  followed by calculation of 2nd order gradients, which aprroximates the
%  2nd order derivatives of the image.
% 
% [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = Hessian3D(Volume,Sigma,spacing)
% 
% inputs,
%   I : The image volume, class preferable double or single
%   Sigma : The sigma of the gaussian kernel used. If sigma is zero
%           no gaussian filtering.
%   spacing : input image spacing
%
% outputs,
%   Dxx, Dyy, Dzz, Dxy, Dxz, Dyz: The 2nd derivatives
%
% Function is written by D.Kroon University of Twente (June 2009)

F = Volume;

% Create first and second order diferentiations
Dz=gradient3(F,'z');
Dzz=(gradient3(Dz,'z'));
clear Dz;

Dy=gradient3(F,'y');
Dyy=(gradient3(Dy,'y'));
Dyz=(gradient3(Dy,'z'));
clear Dy;

Dx=gradient3(F,'x');
Dxx=(gradient3(Dx,'x'));
Dxy=(gradient3(Dx,'y'));
Dxz=(gradient3(Dx,'z'));
clear Dx;

end

function [Dxx, Dyy, Dxy] = Hessian2D(I)
%  filters the image with an Gaussian kernel
%  followed by calculation of 2nd order gradients, which aprroximates the
%  2nd order derivatives of the image.
% 
% [Dxx, Dyy, Dxy] = Hessian2D(I,Sigma,spacing)
% 
% inputs,
%   I : The image, class preferable double or single
%   Sigma : The sigma of the gaussian kernel used. If sigma is zero
%           no gaussian filtering.
%   spacing : input image spacing
%
% outputs,
%   Dxx, Dyy, Dxy: The 2nd derivatives

% if nargin < 3, Sigma = 1; end
% 
% if(Sigma>0)
%     F=imgaussian(I,Sigma,spacing);
% else
%     F=I;
% end

F = I;

% Create first and second order diferentiations
Dy=gradient2(F,'y');
Dyy=(gradient2(Dy,'y'));
clear Dy;

Dx=gradient2(F,'x');
Dxx=(gradient2(Dx,'x'));
Dxy=(gradient2(Dx,'y'));
clear Dx;

end

function D = gradient3(F,option)
% This function does the same as the default matlab "gradient" function
% but with one direction at the time, less cpu and less memory usage.
%
% Example:
%
% Fx = gradient3(F,'x');

[k,l,m] = size(F);
D  = zeros(size(F),class(F)); 

switch lower(option)
case 'x'
    % Take forward differences on left and right edges
    D(1,:,:) = (F(2,:,:) - F(1,:,:));
    D(k,:,:) = (F(k,:,:) - F(k-1,:,:));
    % Take centered differences on interior points
    D(2:k-1,:,:) = (F(3:k,:,:)-F(1:k-2,:,:))/2;
case 'y'
    D(:,1,:) = (F(:,2,:) - F(:,1,:));
    D(:,l,:) = (F(:,l,:) - F(:,l-1,:));
    D(:,2:l-1,:) = (F(:,3:l,:)-F(:,1:l-2,:))/2;
case 'z'
    D(:,:,1) = (F(:,:,2) - F(:,:,1));
    D(:,:,m) = (F(:,:,m) - F(:,:,m-1));
    D(:,:,2:m-1) = (F(:,:,3:m)-F(:,:,1:m-2))/2;
otherwise
    disp('Unknown option')
end

end

function D = gradient2(F,option)
% Example:
%
% Fx = gradient2(F,'x');

[k,l] = size(F);
D  = zeros(size(F),class(F)); 

switch lower(option)
case 'x'
    % Take forward differences on left and right edges
    D(1,:) = (F(2,:) - F(1,:));
    D(k,:) = (F(k,:) - F(k-1,:));
    % Take centered differences on interior points
    D(2:k-1,:) = (F(3:k,:)-F(1:k-2,:))/2;
case 'y'
    D(:,1) = (F(:,2) - F(:,1));
    D(:,l) = (F(:,l) - F(:,l-1));
    D(:,2:l-1) = (F(:,3:l)-F(:,1:l-2))/2;
otherwise
    disp('Unknown option')
end
        
end

function filter = inverseHammingFilter(Hx, Hy, Hz, matrix_size)
    % Create an inverse hamming filter based on the provided dimensions and matrix size
    
    filter = ones(matrix_size);
    x = repmat([-round(matrix_size(1)/2):-1, 1:round(matrix_size(1))/2]', 1, matrix_size(2), matrix_size(3));
    y = repmat([-round(matrix_size(2)/2):-1, 1:round(matrix_size(2))/2], matrix_size(1), 1, matrix_size(3));
    z = repmat(reshape([-round(matrix_size(3)/2):-1, 1:round(matrix_size(3))/2], 1, 1, []), matrix_size(1), matrix_size(2), 1);

    coef = x.^2/Hx^2 + y.^2/Hy^2 + z.^2/Hz^2;
    filter(coef <= 1) = 0.6 * (1 - cos(pi * sqrt(coef(coef <= 1))));
end

function img_ = normalize(img)
    % Normalize the image values to be between 0 and 1

    img_ = img - min(img(:));
    img_ = img_ ./ max(img_(:));

end

function im = fft3c(d,option)
% USAGE : im = fft3c(d,option)
%
% fft3c performs a centered fft3
%
% option :
%     0 -> all dir
%     1 -> y dir
%     2 -> x dir
%     8 -> z dir
%     3 -> y,x dir
%     9 -> y,z dir
%     10 -> x,z dir
%     11 -> x,y,z dir
%     
% coded by Sang-Young Zho
% last modified at 2009.05.27

if nargin==1
    option = 11;
end

switch option
    case 0 %     0 -> all dir
        im = ifftshift(fftn(fftshift(d)));
        
    case 1 %     1 -> y dir
        im = ifftshift(fft(fftshift(d),[],1));
    case 2 %     2 -> x dir
        im = ifftshift(fft(fftshift(d),[],2));
    case 8 %     8 -> z dir
        im = ifftshift(fft(fftshift(d),[],3));
        
    case 3 %     3 -> y,x dir
        im = fftshift(d);
        clear d;
        im = fft(im,[],1);
        im = fft(im,[],2);
        im = ifftshift(im);        
    case 9 %     9 -> y,z dir
        im = fftshift(d);
        clear d;
        im = fft(im,[],1);
        im = fft(im,[],3);
        im = ifftshift(im);
    case 10 %     10 -> x,z dir
        im = fftshift(d);
        clear d;
        im = fft(im,[],2);
        im = fft(im,[],3);
        im = ifftshift(im);
        
    case 11 %     11 -> x,y,z dir
        im = fftshift(d);
        clear d;
        im = fft(im,[],1);
        im = fft(im,[],2);
        im = fft(im,[],3);
        im = ifftshift(im);
        
    otherwise
        disp('Error using "fft3c"...')
        disp('Invalid option.')
        im = [];
        return;
end
end

function im = ifft3c(d,option)
% USAGE : im = ifft3c(d,option)
%
% ifft3c performs a centered ifft3
%
% option :
%     0 -> all dir
%     1 -> y dir
%     2 -> x dir
%     8 -> z dir
%     3 -> y,x dir
%     9 -> y,z dir
%     10 -> x,z dir
%     11 -> x,y,z dir
%     
% coded by Sang-Young Zho
% last modified at 2009.05.27

if nargin==1
    option = 0;
end

switch option
    case 0 %     0 -> all dir
        im = ifftshift(ifftn(fftshift(d)));
        
    case 1 %     1 -> y dir
        im = ifftshift(ifft(fftshift(d),[],1));
    case 2 %     2 -> x dir
        im = ifftshift(ifft(fftshift(d),[],2));
    case 8 %     8 -> z dir
        im = ifftshift(ifft(fftshift(d),[],3));
        
    case 3 %     3 -> y,x dir
        im = fftshift(d);
        clear d;
        im = ifft(im,[],1);
        im = ifft(im,[],2);
        im = ifftshift(im);
    case 9 %     9 -> y,z dir
        im = fftshift(d);
        clear d;
        im = ifft(im,[],1);
        im = ifft(im,[],3);
        im = ifftshift(im);
    case 10 %     10 -> x,z dir
        im = fftshift(d);
        clear d;
        im = ifft(im,[],2);
        im = ifft(im,[],3);
        im = ifftshift(im);
        
    case 11 %     11 -> x,y,z dir
        im = fftshift(d);
        clear d;
        im = ifft(im,[],1);
        im = ifft(im,[],2);
        im = ifft(im,[],3);
        im = ifftshift(im);

    otherwise
        disp('Error using "ifft3c"...')
        disp('Invalid option.')
        im = [];
        return;
end
end

function I=imgaussian(I,sigma,siz)
% IMGAUSSIAN filters an 1D, 2D color/greyscale or 3D image with an 
% Gaussian filter. This function uses for filtering IMFILTER or if 
% compiled the fast  mex code imgaussian.c . Instead of using a 
% multidimensional gaussian kernel, it uses the fact that a Gaussian 
% filter can be separated in 1D gaussian kernels.
%
% J=IMGAUSSIAN(I,SIGMA,SIZE)
%
% inputs,
%   I: The 1D, 2D greyscale/color, or 3D input image with 
%           data type Single or Double
%   SIGMA: The sigma used for the Gaussian kernel
%   SIZE: Kernel size (single value) (default: sigma*6)
% 
% outputs,
%   J: The gaussian filtered image
%
% note, compile the code with: mex imgaussian.c -v
%
% example,
%   I = im2double(imread('peppers.png'));
%   figure, imshow(imgaussian(I,10));
% 
% Function is written by D.Kroon University of Twente (September 2009)

if(~exist('siz','var')), siz=sigma*6; end

if(sigma>0)
    % Make 1D Gaussian kernel
    x=-ceil(siz/2):ceil(siz/2);
    H = exp(-(x.^2/(2*sigma^2)));
    H = H/sum(H(:));

    % Filter each dimension with the 1D Gaussian kernels\
    if(ndims(I)==1)
        I=imfilter(I,H, 'same' ,'replicate');
    elseif(ndims(I)==2)
        Hx=reshape(H,[length(H) 1]);
        Hy=reshape(H,[1 length(H)]);
        I=imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
    elseif(ndims(I)==3)
        if(size(I,3)<4) % Detect if 3D or color image
            Hx=reshape(H,[length(H) 1]);
            Hy=reshape(H,[1 length(H)]);
            for k=1:size(I,3)
                I(:,:,k)=imfilter(imfilter(I(:,:,k),Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
            end
        else
            Hx=reshape(H,[length(H) 1 1]);
            Hy=reshape(H,[1 length(H) 1]);
            Hz=reshape(H,[1 1 length(H)]);
            I=imfilter(imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate'),Hz, 'same' ,'replicate');
        end
    else
        error('imgaussian:input','unsupported input dimension');
    end
end
end
