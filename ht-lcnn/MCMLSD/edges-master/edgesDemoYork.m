% Demo for Structured Edge Detector (please see readme.txt first).

%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

load('ECCV_TrainingAndTestImageNumbers.mat')
load('Manhattan_Image_DB_Names.mat')


type = 'train';


%please set the the YorkUrbanDB directory here
dsetPath = fullfile('/Users/yimingqian/Dropbox/PhD/MCMLSD/YorkUrbanDB');
% dsetPath = fullfile('E:\Dropbox\PhD\MCMLSD\YorkUrbanDB');

if strcmp(type, 'train')
    setIndexes = trainingSetIndex;
else
    setIndexes = testSetIndex;
end

Nsetsamples = size(setIndexes,1);


idxTest = 1;
idxImgs = 0;
listing = dir(dsetPath);
ttlData = size(listing,1);
alpha = 0; %Threshold for line probabilities.
% setDebug(0);

for i=1:length(setIndexes)
    filename=Manhattan_Image_DB_Names{setIndexes(idxTest)}(1:end-1)
    idxTest = idxTest + 1;
    %     load([dataFolder '/' filename '.mat']);
    img=imread([dsetPath '/' filename '/' filename '.jpg']);
    E=edgesDetect(img,model);
    save([filename 'edge.mat'],'E');
end
