clear all; clc; close all;

global T1Max; global T1Min;
global T2Max; global T2Min;
global m; global transDrag; global rotDrag;
global L; global Iyy; global grav;

T1Max = 36.7875/2; T2Max = 36.7875/2;
T1Min = 0; T2Min = 0;
m = 1.25; transDrag = 0.25; rotDrag = 0.02255;
L = 0.5; Ixx = 0.03; grav = 9.81;

% set goal region: Vy Vz Theta Wp;
r1 = 0.1; % goal radius of Vy
c1 = 0.0; % goal center of Vy

r2 = 0.1; % goal radius of Vz
c2 = 1.0; % goal center of Vz (for takeoff)

r3 = pi/18; % goal radius of Theta
c3 = 0.0; % goal center of Theta

r4 = pi/10; % goal radius of Wp
c4 = 0.0;  % goal center of Wp

epsilon = 1e-6;

% set entire state region: Vx Vz Theta Wt;
dim = 4;
Min = zeros(dim,1);
Max = zeros(dim,1);
Min(1) = -5.0;
Min(2) = -5.0;
% angle region should be periodic
Min(3) = -pi/2;
Min(4) = -10*pi;

Max(1) = 5.0;
Max(2) = 5.0;
% angle region should be periodic
Max(3) = pi/2;
Max(4) = 10*pi;

dx = [0.1; 0.1; pi/18; pi/10];
% dx = [0.1; 0.1; pi/40; pi/12];
% dx = single(dx);
% dx = [0.1; 0.1; 2*pi/100];
% dx = [0.5; 0.5; 2*pi/100];
% dx = [26/50; 20/39; 2*pi/49];
% dx = [26/100; 20/77; 2*pi/100];

Max(3) = Max(3) - dx(3);
[xs, N] = gridGeneration(dim, Min, Max, dx);
% xs = single(xs);

% initialization
phi = 100 * ones(N(1),N(2),N(3),N(4));
% phi = single(phi);
% phi = 100*ones(N(1),N(2),N(3));

% Target
flag = abs(c1 - xs(:,:,:,:,1)) <= r1 & abs(c2 - xs(:,:,:,:,2)) <= r2 & abs(c3 - xs(:,:,:,:,3)) <= r3 & abs(c4 - xs(:,:,:,:,4)) <= r4;
phi(flag) = 0;
% phi((xs(:,:,:,1).^2 + xs(:,:,:,2).^2) <= r^2) = 0;

%LF sweeping
mex mexLFsweep.cpp; disp('mexing done!');

% numIter = 10000;
numIter = 50;
%TOL = eps;
TOL = 0.1;

startTime = cputime;
tic;
% mexLFsweep(phi,xs,dx,alpha,beta,V1,V2,numIter,TOL);
mexLFsweep(phi,xs,dx,T1Max,T2Max, T1Min, T2Min,m,transDrag,rotDrag,L,Ixx,grav,numIter,TOL);
toc;


endTime = cputime;
fprintf('Total execution time %g seconds', endTime - startTime);

