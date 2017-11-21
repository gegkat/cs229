%% training error vs fraction of training data
figure()
title('Training Error vs. Fraction of Training Data Used')
semilogy([3,10,20,50,100],squeeze(cnn(15,1,:)),'DisplayName','Single Layer CNN')
hold on
semilogy([3,10,20,50,100],squeeze(complex(15,1,:)),'DisplayName','Deep CNN')
hold on
semilogy([3,10,20,50,100],squeeze(linear(15,1,:)),'DisplayName','Linear Regression')
hold on
semilogy([3,10,20,50,100],squeeze(NVIDIA(15,1,:)),'DisplayName','NVIDIA CNN')
hold on
xlabel('% of Training Data Used')
ylabel('Training Error')
legend('show')
title('Training Error vs. Training Set Fraction Used')

%% bar graphs of validation error comparison
figure()
title('Comparison of Validation Errors Across Models')
b=bar([linear(15,2,5),linear(15,1,5);...
    simple(15,2,5),simple(15,1,5);cnn(15,2,5),cnn(15,1,5);...
    complex(15,2,5),complex(15,1,5);NVIDIA(15,2,5),NVIDIA(15,1,5)]);
names={'Linear Regression';'Shallow NN';'Simple CNN';'Deep CNN';'NVIDIA network'};
set(gca,'xticklabel',names)
ylabel('Error in %')
legend([string('Validation Error');string('Training Error')],'show')
b(1).FaceColor=[1,0,0];
b(2).FaceColor=[0,0,1];

%% learning curves
figure()
name={'cnn','complex','linear','NVIDIA','simple'};
subplot(231)
semilogy([3,10,20,50,100],squeeze(cnn(15,1,:)),'DisplayName','Training Error')
hold on
semilogy([3,10,20,50,100],squeeze(cnn(15,2,:)),'DisplayName','Validation Error')
title('Learning Curves for Simple CNN')
xlabel('Percentage of Training Data')
ylabel('Error')
legend('show')
subplot(232)
semilogy([3,10,20,50,100],squeeze(complex(15,1,:)),'DisplayName','Training Error')
hold on
semilogy([3,10,20,50,100],squeeze(complex(15,2,:)),'DisplayName','Validation Error')
title('Learning Curves for a deep CNN')
xlabel('Percentage of Training Data')
ylabel('Error')
legend('show')
subplot(233)
semilogy([3,10,20,50,100],squeeze(linear(15,1,:)),'DisplayName','Training Error')
hold on
semilogy([3,10,20,50,100],squeeze(linear(15,2,:)),'DisplayName','Validation Error')
title('Learning Curves for linear regression')
xlabel('Percentage of Training Data')
ylabel('Error')
legend('show')
subplot(234)
semilogy([3,10,20,50,100],squeeze(NVIDIA(15,1,:)),'DisplayName','Training Error')
hold on
semilogy([3,10,20,50,100],squeeze(NVIDIA(15,2,:)),'DisplayName','Validation Error')
title('Learning Curves for NVIDIA network')
xlabel('Percentage of Training Data')
ylabel('Error')
legend('show')
subplot(235)
semilogy([3,10,20,50,100],squeeze(simple(15,1,:)),'DisplayName','Training Error')
hold on
semilogy([3,10,20,50,100],squeeze(simple(15,2,:)),'DisplayName','Validation Error')
title('Learning Curves for Simple CNN')
xlabel('Percentage of Training Data')
ylabel('Error')
legend('show')