load Generation_data.mat;                                                           % load data from hard drive to workspace
IrradiationData=readtable('Irradiation data.xlsx');
TrainIp=table2array(IrradiationData(1028:2927,4));                          % read data from workspace
TestIp=table2array(generationdata_table_dt(1005:2904,4));
TestIp(isnan(TrainIp)) = [];                                                % remove NAN from DATA
TrainIp(isnan(TrainIp)) = [];                                               % remove NAN from DATA
TrainIp(TestIp>50)=[];                                                      % remove noise (more than 50) from DATA
TestIp(TestIp>50)=[];                                                       % remove noise (more than 50) from DATA
TrainIp(TestIp<0)=[];                                                       % remove noise (less than 0) from DATA
TestIp(TestIp<0)=[];                                                        % remove noise (less than 0) from DATA

TestIp(TrainIp<=0)=[];                                                      % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=[];                                                     % remove noise (less than 0) from DATA

TrainIp=TrainIp';                                                           % convert row vs column
TestIp=TestIp'; 
mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

input = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
target = (TestIp - mn2) / (mx2-mn2);


numTimeStepsTrain = floor(0.8*numel(TrainIp));                               % 80 and 20 percent training and testing points

figure
plot(input(1:50))
hold on
plot(target(1:50),'.-')
legend(["Training" "Testing"])
xlabel("Time")
ylabel("kWh")
title(" Unit Generation")
close Figure 1;

XTrainIp = input(1:numTimeStepsTrain+1);                                     % training input data points
XTestIp = target(1:numTimeStepsTrain+1);                                     % training target data points

YTrainIp = input(numTimeStepsTrain+1:end);                                  % testing input data points
YTestIp = target(numTimeStepsTrain+1:end);

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

rmsepred=[];
rmseupdat=[];
maepred=[];
maeupdat=[];
mapepred=[];
mapeupdat=[];

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer]; 

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options
%     'ValidationData',{XTestIp,YTestIp},...
%     'ValidationFrequency',30, ...



for j=1:1:4
XTestIp = target(1:numTimeStepsTrain+1);
net = trainNetwork([XTrainIp(1:end-j);XTestIp(1:end-j)],XTestIp(j+1:end),layers,options); % LSTM training
[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-(j-1):end);XTestIp(end-(j-1):end)]);
numTimeStepsTest = numel(YTestIp);
for i = j+1:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-j);YPred(:,i-j)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmsepred(j) = (sqrt(mean((YPred-YTest).^2)))*100/(max(YTest));
maepred(j) = mean(abs(YPred-YTest))
mapepred(j) = mean(abs((YPred(YTest~=0)-YTest(YTest~=0)))./YTest(YTest~=0))*100                                                                           % error of network
mape1=((YPred(YTest~=0)-YTest(YTest~=0))./YTest(YTest~=0));
XTestIp = (mx2-mn2)*XTestIp + mn2;  
figure
plot(XTestIp(1:end))
hold on
idx = numTimeStepsTrain+1:(numTimeStepsTrain+1+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain+1) YPred],'.-')
hold off
xlabel("15 mins index")
ylabel("Generation data in kwh")
title("FORECAST WITHOUT UPDATES")
legend(["Observed" "Predicted"])


figure
subplot(2,2,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
xlabel("15 mins index")
ylabel("Generation data in kwh")
title("FORECAST FOR 2 DAYS WITHOUT UPDATES")

subplot(2,2,2)
stem(YPred - YTest)
xlabel("15 mins index")
ylabel("Error")
title("RMSE% = " + rmsepred(j))

subplot(2,2,3)
stem(YPred - YTest)
xlabel("15 mins index")
ylabel("Error")
title("MAE = " + maepred(j))

subplot(2,2,4)
stem(mape1)
xlabel("15 mins index")
ylabel("Error")
title("MAPE% = " + mapepred(j))
%reset network and restart the training for forecasting
XTestIp = target(1:numTimeStepsTrain+1);
net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-j);XTestIp(1:end-j)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp)-j;
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;
YTestforc=YTest(j+1:end)                                                                            % denormlize the predicted data as per min and max of target
rmseupdat(j) = (sqrt(mean((YPred-YTestforc).^2)))*100/(max(YTestforc))
maeupdat(j) = mean(abs(YPred-YTest(j+1:end)))
y1=abs((YPred(YTestforc~=0)-YTestforc(YTestforc~=0)));
y2=YTestforc(YTestforc~=0);
mapeupdat(j) = mean(y1./y2)*100 

XTestIp = (mx2-mn2)*XTestIp + mn2;  
figure
plot(XTestIp(1:end))
hold on
idx = numTimeStepsTrain+1:(numTimeStepsTrain+numTimeStepsTest)+j;
plot(idx,[YTest(1:j) YPred],'.-')
hold off
xlabel("15 min index")
ylabel("gen data in kwh")
title("Forecast with updates")
legend(["Observed" "Forecast"])

figure
subplot(2,2,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
xlabel("15 min index")
ylabel("gen data in kwh")
title("Forecast for two days with updates")

subplot(2,2,2)
stem(YPred - YTestforc)
xlabel("15 min index")
ylabel("Error")
title("RMSE% = " + rmseupdat(j))

subplot(2,2,3)
stem(YPred - YTestforc)
xlabel("15 min index")
ylabel("Error")
title("MAE = " + maeupdat(j))

subplot(2,2,4)
stem(y1./y2)
xlabel("15 min index")
ylabel("Error")
title("MAPE% = " + mapeupdat(j))
end
%GRU NETWORK TRAINING
layers = [ ...
    sequenceInputLayer(numFeatures)
    gruLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer]; 
rmsepred1=[];
rmseupdat1=[];
maepred1=[];
maeupdat1=[];
mapepred1=[];
mapeupdat1=[];
for j=1:1:4
XTestIp = target(1:numTimeStepsTrain+1);
net = trainNetwork([XTrainIp(1:end-j);XTestIp(1:end-j)],XTestIp(j+1:end),layers,options); % LSTM training
[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-(j-1):end);XTestIp(end-(j-1):end)]);
numTimeStepsTest = numel(YTestIp);
for i = j+1:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-j);YPred(:,i-j)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx2-mn2)*YPred + mn2;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx2-mn2)*YTest + mn2;                                              % target data
rmsepred1(j) = (sqrt(mean((YPred-YTest).^2)))*100/(max(YTest))
maepred1(j) = mean(abs(YPred-YTest))
mapepred1(j) = mean(abs((YPred(YTest~=0)-YTest(YTest~=0)))./YTest(YTest~=0))*100                                                                           % error of network
mape1=((YPred(YTest~=0)-YTest(YTest~=0))./YTest(YTest~=0));
XTestIp = (mx2-mn2)*XTestIp + mn2;  
figure
plot(XTestIp(1:end))
hold on
idx = numTimeStepsTrain+1:(numTimeStepsTrain+1+numTimeStepsTest);
plot(idx,[XTestIp(numTimeStepsTrain+1) YPred],'.-')
hold off
xlabel("15 mins index")
ylabel("Generation data in kwh")
title("FORECAST WITHOUT UPDATES")
legend(["Observed" "Predicted"])


figure
subplot(2,2,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
xlabel("15 mins index")
ylabel("Generation data in kwh")
title("FORECAST FOR 2 DAYS WITHOUT UPDATES")

subplot(2,2,2)
stem(YPred - YTest)
xlabel("15 mins index")
ylabel("Error")
title("RMSE% = " + rmsepred1(j))

subplot(2,2,3)
stem(YPred - YTest)
xlabel("15 mins index")
ylabel("Error")
title("MAE = " + maepred1(j))

subplot(2,2,4)
stem(mape1)
xlabel("15 mins index")
ylabel("Error")
title("MAPE% = " + mapepred1(j))
%reset network and restart the training for forecasting
XTestIp = target(1:numTimeStepsTrain+1);
net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-j);XTestIp(1:end-j)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp)-j;
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx2-mn2)*YPred + mn2;
YTestforc=YTest(j+1:end)                                                                            % denormlize the predicted data as per min and max of target
rmseupdat1(j) = (sqrt(mean((YPred-YTestforc).^2)))*100/(max(YTestforc))
maeupdat1(j) = mean(abs(YPred-YTest(j+1:end)))
y1=abs((YPred(YTestforc~=0)-YTestforc(YTestforc~=0)));
y2=YTestforc(YTestforc~=0);
mapeupdat1(j) = mean(y1./y2)*100 

XTestIp = (mx2-mn2)*XTestIp + mn2;  
figure
plot(XTestIp(1:end))
hold on
idx = numTimeStepsTrain+1:(numTimeStepsTrain+numTimeStepsTest)+j;
plot(idx,[YTest(1:j) YPred],'.-')
hold off
xlabel("15 min index")
ylabel("gen data in kwh")
title("Forecast with updates")
legend(["Observed" "Forecast"])

figure
subplot(2,2,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
xlabel("15 min index")
ylabel("gen data in kwh")
title("Forecast for two days with updates")

subplot(2,2,2)
stem(YPred - YTestforc)
xlabel("15 min index")
ylabel("Error")
title("RMSE% = " + rmseupdat1(j))

subplot(2,2,3)
stem(YPred - YTestforc)
xlabel("15 min index")
ylabel("Error")
title("MAE = " + maeupdat1(j))

subplot(2,2,4)
stem(y1./y2)
xlabel("15 min index")
ylabel("Error")
title("MAPE% = " + mapeupdat1(j))
end
rmsenetpred=[rmsepred' rmsepred1'];
maenetpred=[maepred' maepred1'];
mapenetpred=[mapepred' mapepred1'];
rmsenetupdat=[rmseupdat' rmseupdat1'];
maenetupdat=[maeupdat' maeupdat1'];
mapenetupdat=[mapeupdat' mapeupdat1'];
table1=["LSTM RMSE","GRU RMSE","LSTM MAE","GRU MAE","LSTM MAPE","GRU MAPE"];
table2=["15 mins","30 mins","45 mins","60 mins"];
multitimesteperrors=array2table([rmsenetpred maenetpred mapenetpred],"VariableNames",table1,"RowNames",table2);
onetimesteperrors=array2table([rmsenetupdat maenetupdat mapenetupdat],"VariableNames",table1,"RowNames",table2);