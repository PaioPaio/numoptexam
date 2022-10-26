clc; clear; close all;

%% BFGS vs Newton comparison

% steps
n = 40;

% convergence
nsol = 0;

% tolerance
tol = 1e-8;

%% Symbolic and numeric trial

% symbols
syms x y

% circle lol
%fsym(x,y) = x^2 + y^2;

% Himmelblau's function (benchmark)
%fsym(x,y) = (x^2 + y-11)^2 + (x+y^2-7)^2;

% Three-hump camel function
fsym(x,y) = 2*x^2-1.05*x^4+x^6/6 +x*y+y^2;
 
f = matlabFunction(fsym);

% gradient
gradfsym(x,y) = gradient(fsym,[x,y]);
gradf = matlabFunction(gradfsym);
gradcomputed = zeros(2,n);

% hessians
Hsym(x,y) = hessian(fsym,[x,y]);
H = matlabFunction(Hsym);
Hcomputed = zeros(2,2,n);

% solution steps
solsym = zeros(2,n+1);
solcomputed = solsym;
sol = solsym;
sol(:,1) = randn(2,1);

% initialize BFGS matrices and variables
Hb = zeros(2,2,n+1);
%Hb(:,:,1) = H(sol(1,1),sol(2,1));
Hb(:,:,1) = 0.5 * eye(2);
sb = zeros(2,n);
yb = zeros(2,n);
alpha = zeros(1,n);

% fvalue for debug

fsol = [];
fsol(:,1)=sol(:,1);

%% Iterations
% Need to check for both computed gradient (with euler) and symbolic one


% BFGS + symbolic gradient
for i = 1:n 
    
    % LineSearch with simbolic gradient, get both xk+1 and alphak+1
    [xk1, ak1] = linsearch_computed(f,gradf,Hb(:,:,i),sol(:,i));
    sol(:,i+1) = xk1;
    alpha(i) = ak1;
    fsol(:,i+1) = f(sol(1,i+1),sol(2,i+1));

    nsol = i+1;

    if norm(gradf(xk1(1),xk1(2))) <= tol
        break
    end
    
    yb(:,i) = gradf(sol(1,i+1),sol(2,i+1)) - gradf(sol(1,i),sol(2,i));
    sb(:,i) = sol(:,i+1) - sol(:,i);

    Hb(:,:,i+1) = BFGSiteration(Hb(:,:,i),sb(:,i),yb(:,i));
end

% BFGS + computed gradient
for i = 1:n 
    
    % LineSearch with simbolic gradient, get both xk+1 and alphak+1
    [xk1, ak1] = linsearch_computed(f,gradf,Hb(:,:,i),sol(:,i));
    sol(:,i+1) = xk1;
    alpha(i) = ak1;
    fsol(:,i+1) = f(sol(1,i+1),sol(2,i+1));

    nsol = i+1;

    if norm(gradf(xk1(1),xk1(2))) <= tol
        break
    end
    
    yb(:,i) = gradf(sol(1,i+1),sol(2,i+1)) - gradf(sol(1,i),sol(2,i));
    sb(:,i) = sol(:,i+1) - sol(:,i);

    Hb(:,:,i+1) = BFGSiteration(Hb(:,:,i),sb(:,i),yb(:,i));
end

% Newton + computed gradient
for i = 1:n 
    
    % LineSearch with simbolic gradient, get both xk+1 and alphak+1
    [xk1, ak1] = linsearch_computed(f,gradf,Hb(:,:,i),sol(:,i));
    sol(:,i+1) = xk1;
    alpha(i) = ak1;
    fsol(:,i+1) = f(sol(1,i+1),sol(2,i+1));

    nsol = i+1;

    if norm(gradf(xk1(1),xk1(2))) <= tol
        break
    end
    
    yb(:,i) = gradf(sol(1,i+1),sol(2,i+1)) - gradf(sol(1,i),sol(2,i));
    sb(:,i) = sol(:,i+1) - sol(:,i);

    Hb(:,:,i+1) = BFGSiteration(Hb(:,:,i),sb(:,i),yb(:,i));
end

% Newton + symbolic gradient
for i = 1:n 
    
    % LineSearch with simbolic gradient, get both xk+1 and alphak+1
    [xk1, ak1] = linsearch_computed(f,gradf,Hb(:,:,i),sol(:,i));
    sol(:,i+1) = xk1;
    alpha(i) = ak1;
    fsol(:,i+1) = f(sol(1,i+1),sol(2,i+1));

    nsol = i+1;

    if norm(gradf(xk1(1),xk1(2))) <= tol
        break
    end
    
    yb(:,i) = gradf(sol(1,i+1),sol(2,i+1)) - gradf(sol(1,i),sol(2,i));
    sb(:,i) = sol(:,i+1) - sol(:,i);

    Hb(:,:,i+1) = BFGSiteration(Hb(:,:,i),sb(:,i),yb(:,i));
end

%% Check solution 

fsol


%% functions


function Hk1 = BFGSiteration(Hk,sk,yk)
% approximate hessian inverse iteratively through secant equation and PD initial point

    Hk1 = Hk + (sk'*yk + yk'*Hk*yk)/(sk'*yk)^2 * (sk*sk') - (Hk*yk*sk' + sk*yk'*Hk)/(sk'*yk);

end

function [xk1, alpha] = linsearch_computed(f,gradf,H,xk)

    % method constants
    c1 = 1e-4;
    c2 = 0.9;
    rho = 0.8;
    alpha = 0.9;
    iterlimit = 20;

    % search direction and next iteration
    pk = - H * gradf(xk(1),xk(2));
    xk1 = xk + alpha * pk;

    % Armijo and Curvature = Wolfe
    while iterlimit > 0 &&...
          (f(xk1(1), xk1(2))> f(xk(1),xk(2)) + c1 * alpha * gradf(xk(1),xk(2))' * pk ||...
          gradf(xk1(1), xk1(2))' * pk < c2 * gradf(xk(1),xk(2))' * pk)
        alpha = rho * alpha;
        xk1 = xk + alpha * pk;
        iterlimit = iterlimit -1;
    end
    
end
