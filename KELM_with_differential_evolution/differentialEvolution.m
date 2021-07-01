% Differential Evolution: a method for finding the global optimum of the fitFunction
% The program requires 4 objects
% 1. A handle to the cost function to be minimized. The function must return a positive scalar.
% 2. Lower bound L.
% 3. Upper bound H.
% The optional parameters are 
% 1. initial condition on the parameter px0
% 2. Number of random parameter vectors to be evaluated (i.e., the population) NP
% 3. Tolerance tol (default 1e-5) : need not be a small number
% 4. Number of iterations maxGEN (default 1e+6)
% 5. flag=1 to display Emin after every iteration. 
% Relative tolerance for convergence : function will quit under the following 3 scenarios
% 1. |Emin-tol| < 1e-5 or
% 2. The iteration number has reached maxGEN or
% 3. The population has converged |Emin-mean(E)|<1e-5
% Note: For fast convergence use log parameters if their ranges are vastly different
% under that case L=log(lb), and H=log(ub). This also helps if parameters are required to be positive.
% Examples:
%   x=differentialEvolution(@(x) myfun(x,arg1,arg2,...,argn),L,H);
%   x=differentialEvolution(@(x) myfun(x,arg1,arg2,...,argn),L,H,x0);
%   x=differentialEvolution(@(x) myfun(x,arg1,arg2,...,argn),L,H,x0,NP);
%   x=differentialEvolution(@(x) myfun(x,arg1,arg2,...,argn),L,H,[],[],tol); leave undefined objects as []
%   x=differentialEvolution(@(x) myfun(x,arg1,arg2,...,argn),L,H,x0,NP,tol,maxGEN);
%   x=differentialEvolution(@(x) myfun(x,arg1,arg2,...,argn),L,H,x0,NP,tol,maxGEN,flag);
% Note: flag=1 by default and prints out the best E after every iteration,
% to make it 0 you will have to explicitly write
%   x=differentialEvolution(@(x) myfun(x,arg1,arg2,...,argn),L,H,[],[],[],[],0);
% where the other options may be left empty '[]', or have values.
% The program saves intermediate parameter values when the number of iterations reach
% 5000 as a convinience.
% Remarks: This DE is a variant of the types of DE discussed in 
% Differential Evolution: A Survey of the State-of-the-Art, IEEE Trans.
% Evol. Comput. vol 15, no 1, Feb 2011
function [px,MinCost]=differentialEvolution(fitFunc,L,H,varargin)
    % set defaults
    D=length(L);
    NP=10*D;% population is 10 times # of parameters
    tol=1e-5;
    maxGEN=1e+6;
    inargs=nargin;   
    flag=1;
    switch(inargs)
        case 4,        
            if(~isempty(varargin{1}))
                    px0=varargin{1};
            end            
        case 5,            
            if(~isempty(varargin{1}))
                    px0=varargin{1};
            end   
            if(~isempty(varargin{2}))
                NP=varargin{2};
            end
        case 6,    
            if(~isempty(varargin{1}))
                    px0=varargin{1};
            end                        
            if(~isempty(varargin{2}))
                NP=varargin{2};
            end
            if(~isempty(varargin{3}))
                tol=varargin{3};
            end
        case 7,    
            if(~isempty(varargin{1}))
                    px0=varargin{1};
            end                        
            if(~isempty(varargin{2}))
                NP=varargin{2};
            end
            if(~isempty(varargin{3}))
                tol=varargin{3};
            end            
            if(~isempty(varargin{4}))
                maxGEN=varargin{4};
            end          
        case 8,    
            if(~isempty(varargin{1}))
                    px0=varargin{1};
            end                        
            if(~isempty(varargin{2}))
                NP=varargin{2};
            end
            if(~isempty(varargin{3}))
                tol=varargin{3};
            end            
            if(~isempty(varargin{4}))
                maxGEN=varargin{4};
            end     
            if(~isempty(varargin{5}))
                flag=varargin{5};
            end   
    end
   
   Pop=zeros(D,NP); % Population
   Epop=zeros(1,NP); % cost function of population
   iBest=1; % best index

   r=zeros(1,2); % random index

   % initialize random number generator
   rand('state',sum(100*clock));
   fnew=@(x) fitFunc(x); 
   if(exist('px0','var'))
    Pop(:,1)=px0;
    Epop(1)= feval(fnew,Pop(:,1));
    if(isinf(Epop(1)))
        for j=1:NP            
            Pop(:,j)=L + (H-L).*rand(D,1);
            Epop(j)= feval(fnew,Pop(:,j));
            while(isinf(Epop(j)))
                Pop(:,j)=L + (H-L).*rand(D,1);
                Epop(j)= feval(fnew,Pop(:,j));                
            end
        end        
    else
        for j=2:NP
            Pop(:,j)=L + (H-L).*rand(D,1);
            Epop(j)= feval(fnew,Pop(:,j));
            while(isinf(Epop(j)))
                Pop(:,j)=L + (H-L).*rand(D,1);
                Epop(j)= feval(fnew,Pop(:,j));                
            end        
        end
    end
   else
    for j=1:NP
        Pop(:,j)=L + (H-L).*rand(D,1);
        Epop(j)= feval(fnew,Pop(:,j));
        while(isinf(Epop(j)))
            Pop(:,j)=L + (H-L).*rand(D,1);
            Epop(j)= feval(fnew,Pop(:,j));                
        end        
    end
   end
       
   [Ebest,iBest]=min(Epop);GEN=1;
   E0=Ebest;
   
   while(abs(Ebest-tol)>1e-5 && (GEN<=maxGEN) && (abs(Ebest-mean(Epop))>1e-5))   
     for j=1:NP
       % Pick indexes for random difference vector
       r(1) = floor(rand()* NP) + 1;
       while (r(1)==j)
	     r(1) = floor(rand()* NP) + 1;
       end
       r(2) = floor(rand()* NP) + 1;
       while ((r(2)==r(1))||(r(2)==j))
         r(2) = floor(rand()* NP) + 1;
       end
       r(3) = floor(rand()* NP) + 1;
       while ((r(3)==r(2))||(r(3)==r(1))||(r(3)==j))
         r(3) = floor(rand()* NP) + 1;
       end
      
       % create trial vector
       % crossover probability (1-CR) is very high initially 
       % and tends to 0 as population converges
       % so CR is inversely proportional to Ebest
       % if Ebest is bad CR is small, else CR is high
       CR = unifrnd(0.2,0.9);
       % Factor F has to be high when Ebest is high, and low else
       % F is proportional to Ebest
       F=unifrnd(0.5,2);
       % mutant vector
       v=Pop(:,r(3))+F*(Pop(:,r(1))-Pop(:,r(2)));
       id1=find(v<L);id2=find(v>H);  
       id=union(id1,id2);
       v(id)=L(id)+(H(id)-L(id)).*rand(length(id),1);     
       u=Pop(:,j);% trial vector
       rv=rand(D,1);
       id=find(rv<CR);
       u(id)=v(id);
       
       Eu=feval(fnew,u);
       if(Eu <= Epop(j))
         Pop(:,j)=u;
         Epop(j)=Eu;
       end
     end
     [Ebest,iBest]=min(Epop);
     MinCost(GEN)=Ebest;
     disp(['The best of Generation # ', num2str(GEN), ' are ',num2str(Ebest)]);
     if(flag==1)
        disp(Ebest);
     end
     GEN=GEN+1;
     if(GEN==5000)
         px=Pop(:,iBest);
         strt=strcat('intermediate_',num2str(GEN));
         save(strt,'px');
     end
   end
   px=Pop(:,iBest);
