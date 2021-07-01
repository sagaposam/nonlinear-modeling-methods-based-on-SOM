function [bestW,bestWO,bestNetDef,bestRP,bestIn,Mincost,tElapsed]=optimization(TP,TT,VP,VT,N,int_max,PopSize)
% Train of the network using the O-ELM framework使用O-ELM框架进行网络训练
% 
% Usage        - [bestW,bestWO,bestNetDef,bestRP,bestIn,tElapsed]=OELM_Train(TP,TT,VP,VT,N,Method,int_max,PopSize)
%
% Input:
% TP           - Training input matrix (Number of inputs x Number of samples)
% TT           - Training output vector (1 x Number os samples)
% VP           - Validation input matrix (Number of inputs x Number of samples)
% VT           - Validation output vector (1 x Number os samples)
% N            - Maximum number of hidden nodes
% Method       - Optimization method:
%                       'GA': Genetic Algorithm
%                       'SA': Simulated Annealing
%                       'DE': Differential Evolutio
%                       'BBO': Biogeography-based optimization
% W1          - Matrix of input weights and bias (Number of hidden neurons x Number of Input +1)
% W2          - Vector of output weights
% int_max     - Maximum number of iterations
% PopSize     - Polulation size (is only used in GA and SA)
%
% Output: 
% bestW       - Optimized matrix of input weights and bias
% bestWO      - Optimized matrix of output weights
% bestNetDef  - Optimized structure of the network
% bestRP      - Tikhonov's regularization parameter
% bestIn      - Optimized set of input variables
% tElepsed    - Time elapsed
%                          
%
 %Chaotic_map_no=1; %Chebyshev
%Chaotic_map_no=2; %Circle
% Chaotic_map_no=3; %Gauss/mouse
%Chaotic_map_no=4; %Iterative
%Chaotic_map_no=5; %Logistic
%Chaotic_map_no=6; %Piecewise
%Chaotic_map_no=7; %Sine
%Chaotic_map_no=8; %Singer
%Chaotic_map_no=9; %Sinusoidal
%Chaotic_map_no=10; %Tent

%You can define the number of search agents and iterations in the Init.m file
%Max_iterations=10000;% This should be equal or greater than OPTIONS.Maxgen in Init.m file

%ChaosVec=zeros(10,Max_iterations);
%Calculate chaos vector
%for i=1:10
  %  ChaosVec(i,:)=chaos(i,Max_iterations,1);
%end

global ind_IL ind_HL ind_W ind_RP NN NI NNmax NNmin Wmax Wmin RPmax RPmin P T VV PS
P=TP;
T=TT;
VV.P=VP;
VV.T=VT;
% PS=PS2;

NI=size(P,1); % Number of Inputs
NN=N;         % Maximum number of hidden nodes


% ---  Initialize Population ---
I=rand(PopSize,NI); 
N=rand(PopSize,NN);
Miw=rand(PopSize,NN*(NI+1));
TR=rand(PopSize,1);
Mt=[I N Miw TR];
Mi=Mt(1,:);
ind_IL=1:NI;
ind_HL=NI+1:NI+NN;
ind_W=NI+NN+1:NI+NN+NN*(NI+1);
ind_RP=NI+NN+NN*(NI+1)+1;

% ---  Bounds of regularization paramenter (Real variable) ---
RPmin=0;
RPmax=100;

% ---  Bounds of auxiliar variable that defines each activation function of hidden nodes (Integer variable) (only 2 types of activation function were used) ---
NNmin=0;
NNmax=2;

% ---  Bounds of input weights matrix ---
Wmin=-1;
Wmax=1;


% --- Train of the network ---
tStart=tic;
[M,Mincost]=differentialEvolution(@loss,zeros(size(Mi))',ones(size(Mi))',Mi,PopSize,[],int_max,0);
tElapsed=toc(tStart);
% --- Process de obtained soluction ---
iw=reshape(M(ind_W),NN,NI+1);
auxI=M(ind_IL);
auxN=M(ind_HL);
auxN=floor((NNmax-NNmin+1)*auxN)+NNmin;
auxI=round(auxI);
bestIn=logical(auxI);
Neu=(auxN==0);
NeuT=(auxN==1);
NeuL=(auxN==2);

bestW=(Wmax-Wmin)*iw+Wmin;
bestW(:,~bestIn)=[];
bestW(Neu,:)=[];

H=zeros(1,NN);
H(NeuL) = char('L');
H(NeuT) = char('S');
H(Neu)=[];
O=char([76 ones(1,length(H)-1)*'-']);
bestNetDef = [H;O];

bestRP= (RPmax-RPmin)*M(ind_RP)+RPmin;
bestWO=OELMCF(P(bestIn,:),T,bestW,bestNetDef,bestRP);

function p=loss(x) %Fitness function
global ind_IL ind_HL ind_W ind_RP NN NI NNmax NNmin Wmax Wmin RPmax RPmin P T VV PS

iw=reshape(x(ind_W),NN,NI+1);
auxI=x(ind_IL);
auxN=x(ind_HL);
auxN=floor((NNmax-NNmin+1)*auxN)+NNmin;
auxI=round(auxI);
if sum(auxI)~=0 && sum(auxN)~=0
    In=logical(auxI);
    Neu=(auxN==0);
    NeuT=(auxN==1);
    NeuL=(auxN==2);
    
    W=(Wmax-Wmin)*iw+Wmin;
    W(:,~In)=[];
    W(Neu,:)=[];
    
    H=zeros(1,NN);
    H(NeuL) = char('L');
    H(NeuT) = char('S');
    H(Neu)=[];
    O=char([76 ones(1,length(H)-1)*'-']);
    netDef = [H;O];
    
    regp= (RPmax-RPmin)*x(ind_RP)+RPmin;
    wo=OELMCF(P(In,:),T,W,netDef,regp);
    [~,p]=OELM_Predict(netDef,W,wo,VV.P(In,:),VV.T);
    
else
    p=inf;
end




