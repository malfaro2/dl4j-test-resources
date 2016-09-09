

Files for regression testing.

//////////////////////////////////////////

# 050_ModelSerializer_Regression_MLP_1.zip

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

ModelSerializer.writeModel(net, new File("050_ModelSerializer_Regression_MLP_1.zip"), true);

////////////////////////////////////////////////////
