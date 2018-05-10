


```
public class UpsamplingTest {

    public static void main(String[] args) throws Exception {
        Nd4j.getRandom().setSeed(12345);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .convolutionMode(ConvolutionMode.Same)
            .activation(Activation.TANH)
            .list()
            .layer(new ConvolutionLayer.Builder().kernelSize(2,2).stride(1,1).nOut(5).build())
            .layer(new Upsampling2D.Builder().size(2).build())
            .layer(new ConvolutionLayer.Builder().kernelSize(2,2).stride(1,1).nOut(3).build())
            .layer(new OutputLayer.Builder().nOut(10).lossFunction(LossFunctions.LossFunction.MSE).build())
            .setInputType(InputType.convolutional(28, 28, 1))
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.save(new File("net.bin"));

        INDArray input = Nd4j.rand(new int[]{3,1, 28, 28});
        INDArray labels = Nd4j.rand(3, 10);
        INDArray out = net.output(input, false);

        File fIn = new File("in.bin");
        File fLabels = new File("labels.bin");
        File fOut = new File("out.bin");
        File fGrad = new File("gradient.bin");

        net.setInput(input);
        net.setLabels(labels);
        net.computeGradientAndScore();
        INDArray grad = net.getFlattenedGradients();

        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(fIn))){
            Nd4j.write(input, dos);
        }

        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(fLabels))){
            Nd4j.write(labels, dos);
        }

        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(fOut))){
            Nd4j.write(out, dos);
        }

        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(fGrad))){
            Nd4j.write(grad, dos);
        }


    }

}
```