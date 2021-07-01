function [centers,p,ind,U,maxU] = FCM(D,m)

%D£ºTrained SOM output,m is the maximum number of clusters

centers={};
U={};
maxU={};
coodbook_cluster={};


for i=1:m
    [centers{i,1},U{1,i}] = fcm(D,i);
    maxU{1,i}=max(U{1,i});
end
    



% we=size(sMap.codebook,1);
coodbook_cluster=[];
P_select={};
for i=2:m
    for j=1:size(D,1)
    coodbook_cluster(i,j)=find(U{1,i}(:,j) == maxU{1,i}(j));
        for k=1:i
        P_select{i,k}=find(coodbook_cluster(i,:)==k);
        
        
        
            
        end
   
 
    end
    
    
    
    
    
end 

sm=[];
sm1=zeros(size(D,1),1);
p={};
 for i=2:m
     for j=1:i
         sm=P_select{i,j};
         sm1(sm)=j;
         
     end
     p{i,1}=sm1;
     p{1,1}=ones(size(D,1),1);
 
 end 
ind=[];
for i=1:m

 ind(i) = db_index(D,p{i}, centers{i}, 2);


end












return;