

Files for regression testing.

Created with: DL4J 0.6.0 Release

This file provides the configurations and code used to generate these files.

/////////////////////////////////////////////////

# 060_ModelSerializer_Regression_MLP_1.zip

MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(12345)
    .iterations(1)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(0.15)
    .updater(Updater.NESTEROVS).momentum(0.9)
    .weightInit(WeightInit.XAVIER)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(3).nOut(4)
        .activation("relu")
        .build())
    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation("softmax")
        .nIn(4).nOut(5).build())
    .pretrain(false).backprop(true).build();

MultiLayerNetwork net = new MultiLayerNetwork(conf);
net.init();

int numParams = net.numParams();
INDArray params = Nd4j.linspace(1,numParams, numParams);
net.setParameters(params);

ModelSerializer.writeModel(net, new File("060_ModelSerializer_Regression_MLP_1.zip"), true);

////////////////////////////////////////////////////

# 060_ModelSerializer_Regression_MLP_2.zip

MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	.seed(12345)
	.iterations(1)
	.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
	.learningRate(0.15)
	.updater(Updater.RMSPROP).rmsDecay(0.96)
	.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.1,1.2))
	.regularization(true).dropOut(0.6).l1(0.1).l2(0.2)
	.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.5)
	.list()
	.layer(0, new DenseLayer.Builder().nIn(3).nOut(4)
		.activation("leakyrelu")
		.build())
	.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
		.activation("identity")
		.nIn(4).nOut(5).build())
	.pretrain(false).backprop(true).build();

MultiLayerNetwork net = new MultiLayerNetwork(conf);
net.init();

int numParams = net.numParams();
INDArray params = Nd4j.linspace(1,numParams, numParams);
net.setParameters(params);

org.deeplearning4j.nn.api.Updater u = net.getUpdater();
int viewStateSize = u.stateSizeForLayer(net);
u.setStateViewArray(net, Nd4j.linspace(1,viewStateSize,viewStateSize), false);

ModelSerializer.writeModel(net, new File("060_ModelSerializer_Regression_MLP_2.zip"), true);



////////////////////////////////////////////////////

# 060_ModelSerializer_Regression_CNN_1.zip

MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(12345)
    .iterations(1)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(0.15)
    .updater(Updater.RMSPROP).rmsDecay(0.96)
    .weightInit(WeightInit.RELU)
    .list()
    .layer(0, new ConvolutionLayer.Builder().kernelSize(2,2).stride(1,1).padding(0,0).nIn(3).nOut(3).activation("tanh").build())    //(28-2+0)/1+1 = 27
    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(1,1).build())   //(27-2+0)/1+1 = 26
    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation("sigmoid")
        .nIn(26*26*3).nOut(5).build())
	.setInputType(InputType.convolutional(28,28,1)
    .pretrain(false).backprop(true).build();

MultiLayerNetwork net = new MultiLayerNetwork(conf);
net.init();

int numParams = net.numParams();
INDArray params = Nd4j.linspace(1,numParams, numParams);
net.setParameters(params);

org.deeplearning4j.nn.api.Updater u = net.getUpdater();
int viewStateSize = u.stateSizeForLayer(net);
u.setStateViewArray(net, Nd4j.linspace(1,viewStateSize,viewStateSize), false);

ModelSerializer.writeModel(net, new File("060_ModelSerializer_Regression_CNN_1.zip"), true);


////////////////////////////////////////////////////

# 060_ModelSerializer_Regression_LSTM_1.zip

MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	.seed(12345)
	.learningRate(0.15)
	.updater(Updater.RMSPROP).rmsDecay(0.96)
	.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.1, 1.2))
	.regularization(true).dropOut(0.6).l1(0.1).l2(0.2)
	.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.5)
	.list()
	.layer(0, new GravesLSTM.Builder().nIn(3).nOut(4)
		.activation("tanh")
		.build())
	.layer(1, new GravesBidirectionalLSTM.Builder().nIn(4).nOut(4)
		.activation("softsign")
		.build())
	.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
		.activation("softmax")
		.nIn(4).nOut(5).build())
	.pretrain(false).backprop(true).build();

MultiLayerNetwork net = new MultiLayerNetwork(conf);
net.init();

int numParams = net.numParams();
INDArray params = Nd4j.linspace(1, numParams, numParams);
net.setParameters(params);

org.deeplearning4j.nn.api.Updater u = net.getUpdater();
int viewStateSize = u.stateSizeForLayer(net);
u.setStateViewArray(net, Nd4j.linspace(1, viewStateSize, viewStateSize), false);

ModelSerializer.writeModel(net, new File("060_ModelSerializer_Regression_LSTM_1.zip"), true);



////////////////////////////////////////////////////


# 060_ModelSerializer_Regression_CG_LSTM_1.zip

ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
	.seed(12345)
	.learningRate(0.15)
	.updater(Updater.RMSPROP).rmsDecay(0.96)
	.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.1, 1.2))
	.regularization(true).dropOut(0.6).l1(0.1).l2(0.2)
	.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.5)
	.graphBuilder()
	.addInputs("in")
	.addLayer("0", new GravesLSTM.Builder().nIn(3).nOut(4)
		.activation("tanh")
		.build(), "in")
	.addLayer("1", new GravesBidirectionalLSTM.Builder().nIn(4).nOut(4)
		.activation("softsign")
		.build(), "0")
	.addLayer("2", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
		.activation("softmax")
		.nIn(4).nOut(5).build(), "1")
	.setOutputs("2")
	.pretrain(false).backprop(true).build();

ComputationGraph net = new ComputationGraph(conf);
net.init();

int numParams = net.numParams();
INDArray params = Nd4j.linspace(1, numParams, numParams);
net.setParams(params);

ComputationGraphUpdater u = net.getUpdater();
int viewStateSize = u.getStateViewArray().length();
u.setStateViewArray(Nd4j.linspace(1, viewStateSize, viewStateSize));

ModelSerializer.writeModel(net, new File("060_ModelSerializer_Regression_CG_LSTM_1.zip"), true);
