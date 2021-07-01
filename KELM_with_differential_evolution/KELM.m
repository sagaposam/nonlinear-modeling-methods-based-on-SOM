load P1_Apr.mat
a=P1_Apr(:,[4,7,8,9,10,11,12,13]);
X=a(1:1439,1:end-1)'; 
Y=a(1:1439,end)'; 
TX=a(1440:2159,1:end-1)'; 
TY=a(1440:2159,end)'; 
[x,inputps]=mapminmax(X,0,1); 
tx=mapminmax('apply',TX,inputps);
y=Y;
ty=TY;





Ytrue=ty';



[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY] = elm_kernel1(x,y,tx,ty,0, 40.508162009807690, 'RBF_kernel',1.251706184240190);