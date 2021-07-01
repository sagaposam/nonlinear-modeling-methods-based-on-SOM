clear; 
clc; 
 

t1=clock; 
%--------------- Organize the training data ------------ 
load P1_Apr.mat
a=P1_Apr(:,[4,7,8,9,10,11,12,13]);
%数据预备
%数据预处理
X=a(1:1439,1:end-1)'; %size：15x1096
Y=a(1:1439,end)'; %size：1x1096
TX=a(1440:2159,1:end-1)'; %size：15x200
TY=a(1440:2159,end)'; %size：1x200
[x,inputps]=mapminmax(X,0,1); %inputps允许对值进行一致处理的过程设置
tx=mapminmax('apply',TX,inputps);
%[y,outputps]=mapminmax(Y,0,1);
%ty=mapminmax('apply',TY,outputps);
y=Y;
ty=TY;

Dw1_1=x';
Dw1_2=y';
Dw2_1=tx';
Dw2_2=ty';
Dw1=[Dw1_1,Dw1_2];
Dw2=[Dw2_1,Dw2_2];
Dw=[Dw1;Dw2];
Ytrue1=Dw1_2;
 

[LEN_DATA1 DIM_INPUT1]=size(Dw1_1);  % Data matrix size (1 input vector per row) 
 
%------------------------------------------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Create the network structure  %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
Mx =14;   % Number of neurons in the X-dimension 
My =14;    % Number of neurons in the Y-dimension 
MAP_SIZE = [Mx My];        % Size of SOM map 
sMap = som_map_struct(2*DIM_INPUT1,'msize',MAP_SIZE,'rect','sheet'); 

 
sMap.codebook = rand(size(sMap.codebook));   % Random weight initialization 
 
Co=som_unit_coords(sMap); % Coordinates of neurons in the map 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Specification of some training parameters %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
si=round(max(Mx,My)/2);  % Initial neighborhood 
sf=0.001;                % Final neighborhood 
ei=0.1;                  % Initial learning rate 
ef=0.001;              % Final learning rate 
Nep=10;                % Number of epochs 
Tmax=LEN_DATA1*Nep;     % Maximum number of iterations 
T=0:Tmax;              % Time index for training iteration 
eta=ei*power(ef/ei,T/Tmax);  % Learning rate vector 
sig=si*power(sf/si,T/Tmax);  % Neighborhood width vector 
 
%%%%%%%%%%%%%%%%%%%%%%%% 
% Train Kohonen Map   %% 
%%%%%%%%%%%%%%%%%%%%%%%%



for t=1:Nep  % loop for the epochs 
     
    epoch=t; % Show current epoch 
    for tt=1:LEN_DATA1
         
         T=(t-1)*LEN_DATA1+tt;    % iteration throughout the epochs 
          
         % Compute distances of all prototype vectors to current input 
         Di=sqrt(som_eucdist2(sMap.codebook(:,1:DIM_INPUT1),Dw1_1(tt,:))); %Di:144x1
 
         [Di_min win] = min(Di);         % Find the winner (BMU)  Di_min记录了每列的最小值，win记录了每列最小值的行索引
          
         % Prediction procedure 
         c=sMap.codebook(win,DIM_INPUT1+1:end);  % Coefficient vector of the winner 
         Yhat1(tt) = dot(c,Dw1_1(tt,:));            % Estimated output 
         error_win(tt)= Ytrue1(tt)-Yhat1(tt);     % Prediction error 
      
         % Update the clustering weight vector and the coefficient vector 
         % of the winner and of its neighbors 
         T=(t-1)*LEN_DATA1+tt;    % iteration throughout the epochs 
         for i=1:Mx*My
             % Squared distance (in map coordinates) between winner and neuron i 
             D2=power(norm(Co(win,:)-Co(i,:)),2); 
              
             % Compute corresponding value of the neighborhood function 
             H=exp(-0.5*D2/(sig(T)*sig(T))); 
              
             % Update the clustering weights of neuron i 
             sMap.codebook(i,1:DIM_INPUT1)=sMap.codebook(i,1:DIM_INPUT1) + eta(T)*H*(Dw1_1(tt,:)-sMap.codebook(i,1:DIM_INPUT1)); 
              
             % Update the coefficient vector of the i-th neuron (LMS-like rule) 
             c=sMap.codebook(i,DIM_INPUT1+1:end);  % Coefficient vector of the i-th neuron  
             Yhat1(i) = dot(c,Dw1_1(tt,:)); 
             error(i) = Ytrue1(tt)-Yhat1(i); 
             sMap.codebook(i,DIM_INPUT1+1:end)=sMap.codebook(i,DIM_INPUT1+1:end)+eta(T)*H*error(i)*Dw1_1(tt,:); 
         end 
    end 
    SSE(t)=mean(sum(error_win.^2)); 
end 
 
% figure; plot(SSE);      % Plot the learning curve 
t2=clock;
traintime=etime(t2,t1);

t3=clock; 
%--------------- Organize the testing data ------------ 
%clear Yhat; 

 
Ytrue=Dw2_2;       % Desired output values 
%------------------------------------------------------ 
 

[LEN_DATA DIM_INPUT]=size(Dw2_1);  % Data matrix size (1 input vector per row) 
 
for tt=1:LEN_DATA      
         % Compute distances of all prototype vectors to current input 
         Di=sqrt(som_eucdist2(sMap.codebook(:,1:DIM_INPUT),Dw2_1(tt,:))); 
 
         [Di_min win] = min(Di);         % Find the winning neuron  
          
         % Prediction procedure 
         c=sMap.codebook(win,DIM_INPUT+1:end);  % Coefficient vector of the winner 
         Yhat(tt) = dot(c,Dw2_1(tt,:));            % Estimated output 
         
         %error_win(tt)= Ytrue(tt)-Yhat(tt);     % Prediction error 
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

