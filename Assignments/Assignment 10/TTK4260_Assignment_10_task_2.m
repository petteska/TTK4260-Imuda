%% TTK 4260 Multivariat - Task 2
close all
clear
clc

%% Load data
load twotankdata
z = iddata(y, u, 0.2, 'Name', 'Two tank system');

%% Split data
z1 = z(1:1000);
z2 = z(1001:2000);
z3 = z(2001:3000);
plot(z1,z2,z3)
legend('Estimation','Validation 1', 'Validation 2')

%% Define regressor

%NN = [2 3 4]; %[#past outputs, #past inputs, #Input delay]

V = arxstruc(z1,z2,struc(1:5, 1:5, 1:5));
% select best order by Akaike's information criterion (AIC)
nn = selstruc(V,'aic');

%% Nonlinear ARX models
mw1 = nlarx(z1,[5 1 3], wavenet);
mw2 = nlarx(z1,[5 1 3], wavenet('Number', 8 ))
compare(z1,mw1, mw2);

%% View regressors:
getreg(mw2)


%% Mapping:

RegUseTable = mw2.RegressorUsage


RegNames = RegUseTable.Properties.RowNames;
I = contains(RegNames,'u1');
RegUseTable.("y1:NonlinearFcn")(~I) = false;

%% Evaluating new model:
mlin = arx(z1,[5 1 3]);  % linear ARX model

%% Training
compare(z1,mlin,mw2)

%% Val 1:
compare(z2,mlin,mw2)

%% Val 2:
compare(z3,mlin,mw2)


%% Evaluate residuals
resid(z2,mlin,mw2)
legend show

plot(mw2) % plot nonlinearity response as a function of selected regressors



%% Sigmoidnet function

ms1 = nlarx(z1, [5 1 3], sigmoidnet('Number', 8));
compare(z1,ms1)

%% Val2:
compare(z2,ms1);

%% Val3:
compare(z3,ms1);

%% Treepartion:
mt1 = nlarx(z1,[5 1 3], treepartition);

compare(z1, ms1, mt1)

%% Hammerstein-Wiener Model
mhw1 = nlhw(z1, [1 5 3], pwlinear, pwlinear);
compare(z1,mhw1)


plot(mhw1)

%% Dead Zone and saturation

mhw2 = nlhw(z1, [1 5 3], deadzone, saturation);
compare(z1,mhw2)

%%
mhw3 = nlhw(z1, [1 5 3], 'deadzone', unitgain); % no output nonlinearity
mhw4 = nlhw(z1, [1 5 3], [],'saturation'); % no input nonlinearity

compare(z1,mhw3,mhw4)
mhw2.InputNonlinearity.ZeroInterval
mhw2.OutputNonlinearity.LinearInterval


mhw5 = nlhw(z1, [1 5 3], deadzone([0.1 0.2]), saturation([-1 1]));

%% Specific properties

opt = nlhwOptions();
opt.SearchMethod = 'gna';
opt.SearchOptions.MaxIterations = 7;
mhw6 = nlhw(z1, [1 5 3], deadzone, saturation, opt);

opt.SearchOptions.MaxIterations = 30;
mhw7 = nlhw(z1, mhw6, opt);
compare(z1, mhw6, mhw7)