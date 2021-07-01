clear; clc; close all; 
 
t1=clock;
%--------------- Organize the training data ------------ 
load P1_Apr.mat
a=P1_Apr(:,[4,7,8,9,10,11,12,13]);
X=a(1:1440,1:end-1)'; 
Y=a(1:1440,end)'; 
TX=a(1441:2159,1:end-1)'; 
TY=a(1441:2159,end)'; 

[x,inputps]=mapminmax(X,0,1); 
tx=mapminmax('apply',TX,inputps);
y=Y;
ty=TY;

Dw1_1=x';
Dw1_2=y';
Dw2_1=tx';
Dw2_2=ty';
Dw1=[Dw1_1,Dw1_2];
Dw2=[Dw2_1,Dw2_2];
Dw=[Dw1;Dw2];


[LEN_DATA DIM_INPUT]=size(Dw1_1); 


Mx = 14;           % Number of neurons in the X-dimension 
My = 14;            % Number of neurons in the Y-dimension 
MAP_SIZE = [Mx My];   % Size of 2-D SOM map 
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Create a CL network structure  %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
sMap = som_map_struct(DIM_INPUT,'msize',MAP_SIZE,'rect','sheet'); 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Different weights initialization methods %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% sMap  = som_randinit(Dw, sMap);   % Random weight initialization 
% sMap  = som_lininit(Dw, sMap);    % Linear weight initialization 
I=randperm(LEN_DATA); sMap.codebook=Dw1_1(I(1:Mx*My),:);  % Select Mx*My data vectors at random 
 
Co=som_unit_coords(sMap); % Coordinates of neurons in the map 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Specification of some training parameters %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
si=round(max(Mx,My)/2);  % Initial neighborhood 
sf=0.001;                % Final neighborhood 
ei=1;                  % Initial learning rate 
ef=0.001;              % Final learning rate 
Nep=10;                % Number of epochs 
Tmax=LEN_DATA*Nep;     % Maximum number of iterations 
T=0:Tmax;              % Time index for training iteration 
eta=ei*power(ef/ei,T/Tmax);  % Learning rate vector 
sig=si*power(sf/si,T/Tmax);  % Neighborhood width vector 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Train Kohonen Map (TKM)  %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
counter=zeros(1,Mx*My);  % Counter for the number of victories 
for t=1:Nep,  % loop for the epochs 
     
    epoch=t, % Show current epoch 
    for tt=1:LEN_DATA, 
         % Compute distances of all prototype vectors to current input 
         Di=sqrt(som_eucdist2(sMap,Dw1_1(tt,:))); 
 
         % Find the BMU (i.e. the one with minimum value for Di) 
         [Di_min win] = min(Di); 
         counter(win)=counter(win)+1;   % Increment the number of victories of the winner 
      
         % Update the weights of the winner and its neighbors 
         T=(t-1)*LEN_DATA+tt;    % iteration throughout the epochs 
         for i=1:Mx*My, 
             % Squared distance (in map coordinates) between winner and neuron i 
             D2=power(norm(Co(win,:)-Co(i,:)),2); 
              
             % Compute corresponding value of the neighborhood function 
             H=exp(-0.5*D2/(sig(T)*sig(T))); 
              
             % Update the weights of neuron i 
             sMap.codebook(i,:)=sMap.codebook(i,:) + eta(T)*H*(Dw1_1(tt,:)-sMap.codebook(i,:)); 
         end 
    end 
    % Quantization error per training epoch 
    Qerr(t) = som_quality(sMap, Dw1_1); 
end 


% Divide the training vectors into clusters based on learned prototypes 
[V I] = som_divide(sMap,Dw1_1); 




 
% Build local MR models from data 
WW=[]; 
alpha=0.8;
for i=1:Mx*My 
     y=Dw1_2(I{i}(:,1));      % Prediction vector 
     X=[ones(size(y)) V{i}(:,1:end)];  % Regression matrix 
     
     w=inv(X'*X+alpha*eye(size(X,2)))*X'*y;            % Estimated coefficient vector
     WW=[WW; w'];      % Store the coefficient vectors 
end 
t2=clock;
traintime=etime(t2,t1);

t3=clock;
%--------------- Organize the testing data ------------ 

 
Ytrue=Dw2_2;       % Desired output values 
%------------------------------------------------------ 
 
[LEN_DATA DIM_INPUT]=size(Dw2);  % Data matrix size (1 input vector per row) 
 

for t=1:LEN_DATA, 
    win= som_bmus(sMap,Dw2_1(t,:),1);    % Find the winning neuron (bmu) 
    Yhat(t) = dot(WW(win,:),[1 Dw2_1(t,1:end)]);  % Predicted value of neuron 'win' 
    %error(t)=Ytrue(t)-Yhat(t);          % Corresponding prediction error 
   
    
      
end

t4=clock;
testtime=etime(t4,t3);

wholetime=etime(t4,t1);
T_Y=Yhat';
err_jd=Ytrue-T_Y;
  %err_xd=abs(err_jd)./ty;
  mae=mean(abs(err_jd));
  %mape=100*mean(err_xd);
  %nmse=mean(err_jd.^2)/var(ty);
  rmse=mean(err_jd.^2).^0.5;
 
Ytrue_mean=mean(Ytrue);

sum=0;
for i=1:size(Ytrue,1)
    sum1=(Ytrue(i)-Ytrue_mean)^2;
    sum=sum+sum1;
    
end

R_s=1-(rmse^2/(sum/size(Ytrue,1)));