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

%% (2) Vz = ... 
%% (3) \phi = ... 

%% Two det elements for two controls: T1, T2 in Y-Z plane. Note that here is same as what is in 'BaRC' paper cause the gazebo has same routine on "roll" angle with that paper.

det{1} = deriv{1} .* cos(x{2}) / obj.m;
det{2} = deriv{1} .* cos(x{2}) / obj.m;
det{3} = deriv{2};

uMin = [obj.T1Min; obj.T2Min; obj.wtRange(1)];
uMax = [obj.T1Max; obj.T2Max; obj.wtRange(2)]; 

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


