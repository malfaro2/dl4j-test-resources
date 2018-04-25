

Files for regression testing.

Created with: DL4J 1.0.0-alpha Release

This file provides the configurations and code used to generate these files.
These tests were pulled from the examples

/////////////////////////////////////////////////

# GravesLSTMCharModelingExample

```
public class GravesLSTMCharModellingExample {
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 1;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
		int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		int nCharactersToSample = 300;				//Length of each sample to generate
		String generationInitialization = null;		//Optional character initialization; a random character is used if null
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);

		//Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
		CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();

		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.seed(12345)
			.l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(new RmsProp(0.1))
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
					.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();


        net.save(new File("GravesLSTMCharModelingExample_100a.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{3, iter.inputColumns(), 10});
        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("GravesLSTMCharModelingExample_Input_100a.bin")))){
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("GravesLSTMCharModelingExample_Output_100a.bin")))){
            Nd4j.write(output, dos);
        }

        System.exit(1);
```

////////////////////////////////////////////////////

# CustomLayerExample

```
public class CustomLayerExample {

    static{
        //Double precision for the gradient checks. See comments in the doGradientCheck() method
        // See also http://nd4j.org/userguide.html#miscdatatype
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        runInitialTests();
        doGradientCheck();
    }

    private static void runInitialTests() throws IOException {
        /*
        This method shows the configuration and use of the custom layer.
        It also shows some basic sanity checks and tests for the layer.
        In practice, these tests should be implemented as unit tests; for simplicity, we are just printing the results
         */

        System.out.println("----- Starting Initial Tests -----");

        int nIn = 5;
        int nOut = 8;

        //Let's create a network with our custom layer

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()

            .updater( new RmsProp(0.95))
            .weightInit(WeightInit.XAVIER)
            .l2(0.03)
            .list()
            .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(6).build())     //Standard DenseLayer
            .layer(1, new CustomLayer.Builder()
                .activation(Activation.TANH)                                                    //Property inherited from FeedForwardLayer
                .secondActivationFunction(Activation.SIGMOID)                                   //Custom property we defined for our layer
                .nIn(6).nOut(7)                                                                 //nIn and nOut also inherited from FeedForwardLayer
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
                .activation(Activation.SOFTMAX).nIn(7).nOut(nOut).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        net.save(new File("CustomLayerExample_100a.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{3, nIn});
        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("CustomLayerExample_Input_100a.bin")))){
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("CustomLayerExample_Output_100a.bin")))){
            Nd4j.write(output, dos);
        }

        System.exit(1);
```

```
package org.deeplearning4j.examples.misc.customlayers.layer;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Layer configuration class for the custom layer example
 *
 * @author Alex Black
 */
public class CustomLayer extends FeedForwardLayer {

    private IActivation secondActivationFunction;

    public CustomLayer() {
        //We need a no-arg constructor so we can deserialize the configuration from JSON or YAML format
        // Without this, you will likely get an exception like the following:
        //com.fasterxml.jackson.databind.JsonMappingException: No suitable constructor found for type [simple type, class org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer]: can not instantiate from JSON object (missing default constructor or creator, or perhaps need to add/enable type information?)
    }

    private CustomLayer(Builder builder) {
        super(builder);
        this.secondActivationFunction = builder.secondActivationFunction;
    }

    public IActivation getSecondActivationFunction() {
        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
        return secondActivationFunction;
    }

    public void setSecondActivationFunction(IActivation secondActivationFunction) {
        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
        this.secondActivationFunction = secondActivationFunction;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class
        // (i.e., a CustomLayerImpl instance)
        //For the most part, it's the same for each type of layer

        CustomLayerImpl myCustomLayer = new CustomLayerImpl(conf);
        myCustomLayer.setListeners(iterationListeners);             //Set the iteration listeners, if any
        myCustomLayer.setIndex(layerIndex);                         //Integer index of the layer

        //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
        // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
        // (i.e., it's a "view" array in that it's a subset of a larger array)
        // This is a row vector, with length equal to the number of parameters in the layer
        myCustomLayer.setParamsViewArray(layerParamsView);

        //Initialize the layer parameters. For example,
        // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
        // are in turn a view of the 'layerParamsView' array.
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        myCustomLayer.setParamTable(paramTable);
        myCustomLayer.setConf(conf);
        return myCustomLayer;
    }

    @Override
    public ParamInitializer initializer() {
        //This method returns the parameter initializer for this type of layer
        //In this case, we can use the DefaultParamInitializer, which is the same one used for DenseLayer
        //For more complex layers, you may need to implement a custom parameter initializer
        //See the various parameter initializers here:
        //https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/params

        return DefaultParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Memory report is used to estimate how much memory is required for the layer, for different configurations
        //If you don't need this functionality for your custom layer, you can return a LayerMemoryReport
        // with all 0s, or

        //This implementation: based on DenseLayer implementation
        InputType outputType = getOutputType(-1, inputType);

        int numParams = initializer().numParams(this);
        int updaterStateSize = (int)getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if(getIDropout() != null){
            //Assume we dup the input for dropout
            trainSizeVariable += inputType.arrayElementsPerExample();
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, CustomLayer.class, inputType, outputType)
            .standardMemory(numParams, updaterStateSize)
            .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)     //No additional memory (beyond activations) for inference
            .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
            .build();
    }


    //Here's an implementation of a builder pattern, to allow us to easily configure the layer
    //Note that we are inheriting all of the FeedForwardLayer.Builder options: things like n
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        private IActivation secondActivationFunction;

        //This is an example of a custom property in the configuration

        /**
         * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
         *
         * @param secondActivationFunction Second activation function for the layer
         */
        public Builder secondActivationFunction(String secondActivationFunction) {
            return secondActivationFunction(Activation.fromString(secondActivationFunction));
        }

        /**
         * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
         *
         * @param secondActivationFunction Second activation function for the layer
         */
        public Builder secondActivationFunction(Activation secondActivationFunction){
            this.secondActivationFunction = secondActivationFunction.getActivationFunction();
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public CustomLayer build() {
            return new CustomLayer(this);
        }
    }

}

```

```
package org.deeplearning4j.examples.misc.customlayers.layer;

import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Layer (implementation) class for the custom layer example
 *
 * @author Alex Black
 */
public class CustomLayerImpl extends BaseLayer<CustomLayer> { //Generic parameter here: the configuration class type

    public CustomLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
    }


    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        /*
        The preOut method(s) calculate the activations (forward pass), before the activation function is applied.

        Because we aren't doing anything different to a standard dense layer, we can use the existing implementation
        for this. Other network types (RNNs, CNNs etc) will require you to implement this method.

        For custom layers, you may also have to implement methods such as calcL1, calcL2, numParams, etc.
         */

        return super.preOutput(x, training);
    }


    @Override
    public INDArray activate(boolean training) {
        /*
        The activate method is used for doing forward pass. Note that it relies on the pre-output method;
        essentially we are just applying the activation function (or, functions in this example).
        In this particular (contrived) example, we have TWO activation functions - one for the first half of the outputs
        and another for the second half.
         */

        INDArray output = preOutput(training);
        int columns = output.columns();

        INDArray firstHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = layerConf().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        //IActivation function instances modify the activation functions in-place
        activation1.getActivation(firstHalf, training);
        activation2.getActivation(secondHalf, training);

        return output;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        /*
        The baockprop gradient method here is very similar to the BaseLayer backprop gradient implementation
        The only major difference is the two activation functions we have added in this example.

        Note that epsilon is dL/da - i.e., the derivative of the loss function with respect to the activations.
        It has the exact same shape as the activation arrays (i.e., the output of preOut and activate methods)
        This is NOT the 'delta' commonly used in the neural network literature; the delta is obtained from the
        epsilon ("epsilon" is dl4j's notation) by doing an element-wise product with the activation function derivative.

        Note the following:
        1. Is it very important that you use the gradientViews arrays for the results.
           Note the gradientViews.get(...) and the in-place operations here.
           This is because DL4J uses a single large array for the gradients for efficiency. Subsets of this array (views)
           are distributed to each of the layers for efficient backprop and memory management.
        2. The method returns two things, as a Pair:
           (a) a Gradient object (essentially a Map<String,INDArray> of the gradients for each parameter (again, these
               are views of the full network gradient array)
           (b) an INDArray. This INDArray is the 'epsilon' to pass to the layer below. i.e., it is the gradient with
               respect to the input to this layer

        */

        INDArray activationDerivative = preOutput(true);
        int columns = activationDerivative.columns();

        INDArray firstHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        INDArray epsilonFirstHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray epsilonSecondHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = layerConf().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        //IActivation backprop method modifies the 'firstHalf' and 'secondHalf' arrays in-place, to contain dL/dz
        activation1.backprop(firstHalf, epsilonFirstHalf);
        activation2.backprop(secondHalf, epsilonSecondHalf);

        //The remaining code for this method: just copy & pasted from BaseLayer.backpropGradient
//        INDArray delta = epsilon.muli(activationDerivative);
        if (maskArray != null) {
            activationDerivative.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);    //f order
        Nd4j.gemm(input, activationDerivative, weightGrad, true, false, 1.0, 0.0);
        INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        biasGrad.assign(activationDerivative.sum(0));  //TODO: do this without the assign

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);

        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose()).transpose();

        return new Pair<>(ret, epsilonNext);
    }

}

```




////////////////////////////////////////////////////

# VaeMnistAnomaly

```
public class VaeMNISTAnomaly {

    public static void main(String[] args) throws IOException {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 5;                    //Total number of training epochs
        int reconstructionNumSamples = 16;  //Reconstruction probabilities are estimated using Monte-Carlo techniques; see An & Cho for details

        //MNIST data for training
        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(new Adam(0.05))
            .weightInit(WeightInit.XAVIER)
            .l2(1e-4)
            .list()
            .layer(0, new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(256, 256)                    //2 encoder layers, each of size 256
                .decoderLayerSizes(256, 256)                    //2 decoder layers, each of size 256
                .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                .nIn(28 * 28)                                   //Input size: 28x28
                .nOut(32)                                       //Size of the latent variable space: p(z|x) - 32 values
                .build())
            .pretrain(true).backprop(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.save(new File("VaeMNISTAnomaly_100a.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(3, 28*28);
        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("VaeMNISTAnomaly_Input_100a.bin")))){
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("VaeMNISTAnomaly_Output_100a.bin")))){
            Nd4j.write(output, dos);
        }

        System.exit(1);

```


////////////////////////////////////////////////////

# HouseNumberDetection

```
public class HouseNumberDetection {
    private static final Logger log = LoggerFactory.getLogger(HouseNumberDetection.class);

    public static void main(String[] args) throws java.lang.Exception {

        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // number classes (digits) for the SVHN datasets
        int nClasses = 10;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 1.0;
        double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
        double detectionThreshold = 0.5;

        // parameters for the training phase
        int batchSize = 10;
        int nEpochs = 20;
        double learningRate = 1e-4;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

//        SvhnDataFetcher fetcher = new SvhnDataFetcher();
//        File trainDir = fetcher.getDataSetPath(DataSetType.TRAIN);
//        File testDir = fetcher.getDataSetPath(DataSetType.TEST);


        log.info("Load data...");

//        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
//        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        ObjectDetectionRecordReader recordReaderTrain = null;   //new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, new SvhnLabelProvider(trainDir));
//        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = null;    //new ObjectDetectionRecordReader(height, width, nChannels,gridHeight, gridWidth, new SvhnLabelProvider(testDir));
//        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = null;   //new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
//        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = null;    //new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
//        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        ComputationGraph model;
        String modelFilename = "model.zip";

        if (new File(modelFilename).exists()) {
            log.info("Load model...");

            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Build model...");

            ComputationGraph pretrained = (ComputationGraph)new TinyYOLO().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                    .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();

            model = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("conv2d_9")
                    .addLayer("convolution2d_9",
                            new ConvolutionLayer.Builder(1,1)
                                    .nIn(1024)
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1,1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_8")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambbaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            "convolution2d_9")
                    .setOutputs("outputs")
                    .build();

            model.save(new File("HouseNumberDetection_100a.bin"));
            Nd4j.getRandom().setSeed(12345);
            INDArray input = Nd4j.rand(new int[]{3, 3, 416, 416});
            try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("HouseNumberDetection_Input_100a.bin")))){
                Nd4j.write(input, dos);
            }
            INDArray output = model.outputSingle(input);
            try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(new File("HouseNumberDetection_Output_100a.bin")))){
                Nd4j.write(output, dos);
            }

            System.exit(1);
```




////////////////////////////////////////////////////