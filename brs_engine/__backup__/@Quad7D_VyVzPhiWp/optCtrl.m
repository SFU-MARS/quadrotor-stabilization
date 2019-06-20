function uOpt = optCtrl(obj, ~, x, deriv, uMode)

%% Input processing
if nargin < 5
    uMode = 'max';
end

if ~iscell(deriv)
    deriv = num2cell(deriv);
end
%% Optimal control
uOpt = cell(obj.nu, 1);

det = cell(obj.nu, 1);

%% (1) Vy = ... 
%% (2) Vz = ... 
%% (3) \phi = ... 
%% (4) w_p = ... 

%% Two det elements for two controls: T1, T2 in Y-Z plane. Note that here is same as what is in 'BaRC' paper cause the gazebo has same routine on "roll" angle with that paper.

det{1} = deriv{1}.*(-1/obj.m).*sin(x{3}) + deriv{2}.*(1/obj.m).*cos(x{3}) + deriv{4}.*(-obj.l/obj.Ixx);

det{2} = deriv{1}.*(-1/obj.m).*sin(x{3}) + deriv{2}.*(1/obj.m).*cos(x{3}) + deriv{4}.*(obj.l/obj.Ixx);

uMin = [obj.T1Min; obj.T2Min];
uMax = [obj.T1Max; obj.T2Max];

if strcmp(uMode, 'max')
    for i = 1:obj.nu
        uOpt{i} = (det{i} >= 0)*uMax(i) + (det{i} < 0)*uMin(i);
    end

elseif strcmp(uMode, 'min')
    for i = 1:obj.nu
        uOpt{i} = (det{i} >= 0)*uMin(i) + (det{i} < 0)*uMax(i);
    end
end

end


