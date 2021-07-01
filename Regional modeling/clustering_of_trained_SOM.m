
[c,p,ind,U,maxU] = FCM_with_DBindex(sMap.codebook,max_cluster);

ind1=sort(ind);
N_optimal_cluster=0;
for i=1:max_cluster
    if ind(i)==ind1(1)
        
        N_optimal_cluster=i;
    end   
end
P_select=p{N_optimal_cluster};
c_select=c{N_optimal_cluster};
V_weights_matrix=zeros(N_optimal_cluster,Mx*My);
for i=1:N_optimal_cluster
    for j=1:Mx*My
        if P_select(j)==i
           V_weights_matrix(i,j)=j; 
        else
           V_weights_matrix(i,j)=0; 
        end
    end
            
           
            
    
end
cluster={};
for i=1:N_optimal_cluster
       A=V_weights_matrix(i,:);
       A(find(A==0))=[];
       cluster{i}=A;
end


P1=[];
T1=[];
cluster_data={};

WW=[];
W2={};
for i=1:N_optimal_cluster
    [len dim]=size(cluster{1,i});
    
       for j=1:dim
           P2=V{cluster{1,i}(j)}(:,1:end);
           T2=Dw1_2(I{cluster{1,i}(j)});
           P1=[P1;P2];
           T1=[T1;T2];
           D1=[P1,T1];
           
       end
       cluster_data{1,i}=D1;
       P=cluster_data{i}(:,1:end-1)';
       T=cluster_data{i}(:,end)';
       W2{i}=OutputWeight;
       ww=[TrainingAccuracy];
       WW=[WW;ww];
    
       P1=[];
       T1=[];
       D1=[];
     
end







