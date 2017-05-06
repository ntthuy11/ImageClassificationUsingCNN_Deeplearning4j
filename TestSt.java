package thuy.test.dl;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.datavec.image.transform.ColorConversion;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.NetSaverLoaderUtils;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2YCrCb;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class TestSt {

    public static void main(String[] args) {
        new TestSt().runBalancedPathFilter();
    }


    // ======================================


    private Logger  logger          = LoggerFactory.getLogger(TestSt.class);

    //private String  srcDir          = "/Users/thnguyen/Downloads/_LEARNING/deepLearning4j/dl4j-examples/dl4j-examples/src/main/resources/animals/";
    private String  srcDir          = "/Users/thnguyen/Downloads/_LEARNING/deepLearning4j/lettuce-learning/";
    private String  saveDir         = "/Users/thnguyen/Downloads/"; // FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
    private boolean isSaveModel     = false;

    private int     seed1           = 42;
    private int     seed2           = 123;
    private Random  randNumGen1     = new Random(seed1);
    private Random  randNumGen2     = new Random(seed2);

    private int     nImg            = 538;      // (default: 80)
    private int     nLabels         = 4;
    private int     maxPathPerLabel = 53;       // (default: 20) max # images per label/folder (need to increase this as many as we can)
    private int     batchSize       = 53;       // (default: 20) will affect how 'noisy' the gradient is (gradient is averaged over the minibatch)
                                                //               & will change the path the minimization follows. Big batchSize mean less noisy
                                                //               gradient. Noise actually can be useful as it may help escape local minima.
                                                //               Look like 54 is better than 20 -> faster & better accuracy (tried 10, 20, 54, 70)
    private int     ratioForTraining= 80;       // 80% of total # images are used for training (20% used for testing)
    private int     listenerFreq    = 1;
    private int     epochs          = 50;       // (default: 50) measure of # times all of training vectors are used once to update weights
                                                //               increasing this does not mean higher accuracy (tried 10, 20)
    private int     nCores          = 2;        // (default: queueSize = 2) tried 1,2,4,8 -> Look like 2 is fastest, all gave same accuracy
    private String  modelType       = "LeNet";  // (default: AlexNet) LeNet, AlexNet, or custom

    private int     imgHeightToLoad = 100;      // (default: 100) decreasing this (tried 50) seems lead to lower accuracy
    private int     imgWidthToLoad  = 100;      // (default: 100) increasing this (tried 150) does not mean higher accuracy
    private int     imgChannels     = 3;


    // ======================================


    public void runBalancedPathFilter() { // load images from path/directory with BALANCE between each label (the same # paths for each label)

        // ---------------- Data Setup ----------------
        printAndLogString("Load data....");

        // organize and limit data file paths
        FileSplit fileSplit = new FileSplit(new File(srcDir), NativeImageLoader.ALLOWED_FORMATS, randNumGen1); // define basic dataset split with limits on format
        ParentPathLabelGenerator pathLabelGen = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen1, pathLabelGen, nImg, nLabels, maxPathPerLabel);
                // randomizes the order of paths in an array and removes paths randomly to have the SAME # paths for each label.
                // Further interlaces the paths on output based on their labels, to obtain easily optimal batches for training

        // train test split
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, ratioForTraining, 100 - ratioForTraining);
        InputSplit trainData = inputSplit[0];                           // samples the locations based on the PathFilter & splits the result into
        InputSplit testData = inputSplit[1];                            // an array of InputSplit objects, with sizes proportional to the weights

        // transform to generate large dataset to train on
        ImageTransform flipTransform1 = new FlipImageTransform(randNumGen1);
        ImageTransform flipTransform2 = new FlipImageTransform(randNumGen2);
        ImageTransform warpTransform = new WarpImageTransform(randNumGen1, 42);
        //ImageTransform colorTransform = new ColorConversion(randNumGen1, COLOR_BGR2YCrCb);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});
        //List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{colorTransform});


        // ---------------- Build model ----------------
        printAndLogString("Build model....");

        // define the network
        MultiLayerNetwork network;
        switch (modelType) {
            case "LeNet"     :  network = lenetModel(seed1, nLabels, imgHeightToLoad, imgWidthToLoad, imgChannels);     break;
            //case "AlexNet"   :  network = alexnetModel();           break;
            default:        throw new InvalidInputTypeException("Incorrect model provided.");
        }
        network.init();
        network.setListeners(new ScoreIterationListener(listenerFreq));

        // define others
        ImageRecordReader imgRecordReader = new ImageRecordReader(imgHeightToLoad, imgWidthToLoad, imgChannels, pathLabelGen);
                                                        // an imageLoader to load images in specified Height, Width, Channels
        DataSetIterator dataIter; // to iterate loading images in batch
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1); // normalizer
        MultipleEpochsIterator trainIter; // uses MultipleEpochsIterator to ensure model runs through the data for all epochs


        // ---------------- Train model ----------------
        printAndLogString("Train model....");

        // train without transformations
        try {
            imgRecordReader.initialize(trainData, null);         // TRAINING data, WITHOUT transformations
        } catch (IOException e) { }
        dataIter = new RecordReaderDataSetIterator(imgRecordReader, batchSize, 1, nLabels); // only load 1 batch at a time into memory to save memory
        scaler.fit(dataIter); // normalize images and generate large dataset to train on
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
        network.fit(trainIter);

        // train with transformations
        for (ImageTransform transform : transforms) {
            printAndLogString("\nTraining on transformation: " + transform.getClass().toString() + "\n");
            try {
                imgRecordReader.initialize(trainData, transform); // TRAINING data, WITH transformations
            } catch (IOException e) { }
            dataIter = new RecordReaderDataSetIterator(imgRecordReader, batchSize, 1, nLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
            network.fit(trainIter);
        }


        // ---------------- Evaluate model ----------------
        printAndLogString("Evaluate model....");
        try {
            imgRecordReader.initialize(testData);                 // TESTING DATA, NO transformations
        } catch (IOException e) { }
        dataIter = new RecordReaderDataSetIterator(imgRecordReader, batchSize, 1, nLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        printAndLogString(eval.stats(true));


        // ---------------- Save model ----------------
        if (isSaveModel) {
            printAndLogString("Save model....");
            NetSaverLoaderUtils.saveNetworkAndParameters(network, saveDir);
            NetSaverLoaderUtils.saveUpdators(network, saveDir);
        }


        // ---------------- Example on how to get predict results with trained model ----------------
        /*dataIter.reset();
        DataSet testDataSet = dataIter.next();               // testDataSet has size = 4
        List<String> predict = network.predict(testDataSet); // predict.size = 5

        for(int i = 0; i < predict.size(); i++) {
            String expectedResult = testDataSet.getLabelName(i);
            String modelResult = predict.get(i);
            printAndLogString("\nFor a single example that is labeled [" + expectedResult + "], the model predicted [" + modelResult + "]\n");
        }*/
    }


    // ================================== MODELS ==================================


    // Revisde Lenet Model approach achieves slightly above random
    private MultiLayerNetwork lenetModel(int seed, int nClass, int imgHeightResized, int imgWidthResized, int nImgChannel) {
        int         nIter           = 1;        // 1 (default)
        boolean     regularization  = false;    // FALSE (default)
                                                // Used to solve overfitting, by penalizing the loss function L by adding a multiple of an L1 (LASSO)
                                                //     or an L2 (Ridge) norm of the weights vector w. Equation:    L(X,Y) + lambda * N(w),
                                                //     where N is L1 or L2 or other norm
        //double      l1              = 0.005;    // L1 (LASSO): tried 0.0001, 0.001, 0.005, 0.01
        double      l2              = 0.005;    // L2 (Ridge): 0.005 (default)
        String      activationFunc  = "relu";   // tried "hardtanh", "leakyrelu" (allow a small, non-zero gradient when the unit is not active)
                                                //       "relu" (default), "sigmoid", "softmax", "softsign", "softplus", "tanh"
                                                // Turn sum of inputs into an output using an activation function
                                                // ReLU is a "rectified linear unit" activation func:  f(x) = max(0,x)  where x is input to a neuron
                                                //     is the most popular activation func in 2015
                                                // https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/NeuralNetConfiguration.Builder.html
                                                // TODO: check to use "maxout" (error "UnsupportedOperationException")
        double      learningRate    = 0.0001;   // 0.0001 (default)
                                                // LEARNING RATE defines how much the weights will be updated,
                                                //               from   new_W = old_W + learningRate * (derivativeE/derivativeW)
                                                //               where (dervE/dervW) is calculated based on error and activation function
        WeightInit  weightInit      = WeightInit.XAVIER;// tried DISTRIBUTION (w/out Distr func), NORMALIZED, RELU, UNIFORM, VI, XAVIER (default), ZERO
                                                        // TODO: check to use WeightInit.DISTRIBUTION, WeightInit.SIZE
        OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                                                        // tried CONJUGATE_GRADIENT (good but slow), STOCHASTIC_GRADIENT_DESCENT (default),
                                                        //       LBFGS (bad and very slow), LINE_GRADIENT_DESCENT (good but slow)
                                                        // TODO: check to use HESSIAN_FREE (error "No optimizer found" if stands alone)
        Updater     updater         = Updater.RMSPROP;  // tried ADADELTA, ADAGRAD, ADAM (good and fast) (Adaptive Moment Estimation, update of RMSPROP),
                                                        //       NESTEROVS (good and fast),
                                                        //       NONE (of course bad), RMSPROP (default) (Root Mean Square Propagation), SGD
                                                        // TODO: check to use CUSTOM
        double      momentum        = 0.9;              // 0.9 (default)
        boolean     backprop        = true;     // tried TRUE (default), false (yielded very bad result)
                                                // BACK PROPAGATION is to update/propagate errors from the output layer (backward) to hidden layers.
                                                //                  The errors in hidden layers can be calculated based on fraction/proportion
                                                //                  of weights and output-layer errors
        boolean     pretrain        = false;    // tried TRUE, FALSE (default) // look like TRUE and FALSE yielded similar results but TRUE is slower
                                                // unsupervised learning

        // for layers
        int[] kernel2x2 = new int[]{2, 2};
        int[] kernel5x5 = new int[]{5, 5};

        int[] stride1x1 = new int[]{1, 1};
        int[] stride2x2 = new int[]{2, 2};
        int[] stride5x5 = new int[]{5, 5};

        int[] pad0x0    = new int[]{0, 0};
        int[] pad1x1    = new int[]{1, 1};

        // create model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder() // Ref: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
                .seed(seed)
                .iterations(nIter)
                .regularization(regularization).l2(l2)
                .activation(activationFunc)
                .learningRate(learningRate)
                .weightInit(weightInit)
                .optimizationAlgo(optimizationAlgo)
                .updater(updater).momentum(momentum)
                .list()
                .layer(0, new ConvolutionLayer
                        .Builder(kernel5x5, stride1x1, pad0x0)  // (imgSize - kernel + 2 * pad) / stride + 1
                        .name("cnn1")                           // in this experimental data:
                        .nIn(nImgChannel)                       // LAYER 0:
                        .nOut(50)                               //      (100 - 5 + 0) / 1 + 1 = 96
                        .biasInit(0) // 0 (default)             // => Output of Layer 0:
                        .build())                               //      96 x 96 x 50
                .layer(1, new SubsamplingLayer                  // Pooling layer with kernel3x3, stride2x2 is called overlapping pooling
                        .Builder(kernel2x2, stride2x2)          // Pooling sizes with larger receptive fields are too destructive
                        .name("maxpool1")                       // => Output of LAYER 1::
                        .build())                               //      48 x 48 x 50
                .layer(2, new ConvolutionLayer
                        .Builder(kernel5x5, stride5x5, pad1x1)  // LAYER 2:
                        .name("cnn2")                           //      (48 - 5 + 2*1) / 5 + 1 = 10
                        .nOut(50)                              // => Output of Layer 2:
                        .biasInit(0) // 0 (default)             //      10 x 10 x 100
                        .build())
                .layer(3, new SubsamplingLayer                  // There are also average pooling or even L2-norm pooling
                        .Builder(kernel2x2, stride2x2)          //
                        .name("maxpool2")                       // => Output of LAYER 3:
                        .build())                               //      5 x 5 x 100
                .layer(4, new DenseLayer
                        .Builder()
                        .nOut(500) // 500 (default)
                        .build())
                .layer(5, new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // NEGATIVELOGLIKELIHOOD (default)
                        .nOut(nClass)
                        .activation("softmax") // "softmax" (default)
                        .build())
                .backprop(backprop).pretrain(pretrain)
                .cnnInputSize(imgHeightResized, imgWidthResized, nImgChannel)
                .build();
        return new MultiLayerNetwork(conf);
    }

/*
    private MultiLayerNetwork alexnetModel() { // AlexNet model interpretation based on the original paper ImageNet Classification
        int[] kernel3x3   = new int[]{3, 3};
        int[] kernel5x5   = new int[]{5, 5};
        int[] kernel11x11 = new int[]{11, 11};

        int[] stride1x1   = new int[]{1, 1};
        int[] stride2x2   = new int[]{2, 2};
        int[] stride4x4   = new int[]{4, 4};

        int[] pad2x2      = new int[]{2, 2};
        int[] pad3x3      = new int[]{3, 3};

        double nonZeroBias = 1;
        double dropOut = 0.5;
        Distribution gaussianDist = new GaussianDistribution(0, 0.005);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed1)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, new ConvolutionLayer
                        .Builder(kernel11x11, stride4x4, pad3x3)
                        .name("cnn1")
                        .nIn(imgChannels)
                        .nOut(96)
                        .biasInit(0)
                        .build())
                .layer(1, new LocalResponseNormalization
                        .Builder()
                        .name("lrn1")
                        .build())
                .layer(2, new SubsamplingLayer
                        .Builder(kernel3x3, stride2x2)
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer
                        .Builder(kernel5x5, stride1x1, pad2x2)
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(nonZeroBias) // with Deep Convolutional Neural Networks and the imagenetExample code referenced
                        .build())              // http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
                .layer(4, new LocalResponseNormalization
                        .Builder()
                        .name("lrn2")
                        .build())
                .layer(5, new SubsamplingLayer
                        .Builder(kernel3x3, stride2x2)
                        .name("maxpool2")
                        .build())
                .layer(6, new ConvolutionLayer
                        .Builder(kernel3x3, stride1x1, pad2x2)
                        .name("cnn3")
                        .nOut(384)
                        .biasInit(0)
                        .build())
                .layer(7, new ConvolutionLayer
                        .Builder(kernel3x3, stride1x1, pad2x2)
                        .name("cnn4")
                        .nOut(384)
                        .biasInit(nonZeroBias) // <---
                        .build())
                .layer(8, new ConvolutionLayer
                        .Builder(kernel3x3, stride1x1, pad2x2)
                        .name("cnn5")
                        .nOut(256)
                        .biasInit(nonZeroBias) // <---
                        .build())
                .layer(9, new SubsamplingLayer
                        .Builder(kernel3x3, stride2x2)
                        .name("maxpool3")
                        .build())
                .layer(10, new DenseLayer
                        .Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .biasInit(nonZeroBias)  // <---
                        .dropOut(dropOut)       // <---
                        .dist(gaussianDist)     // <---
                        .build())
                .layer(11, new DenseLayer
                        .Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .biasInit(nonZeroBias)  // <---
                        .dropOut(dropOut)       // <---
                        .dist(gaussianDist)     // <---
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(nLabels)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .cnnInputSize(imgHeightToLoad, imgWidthToLoad, imgChannels)
                .build();
        return new MultiLayerNetwork(conf);
    }
*/


    // ================================== PRIVATE ==================================


    private void printAndLogString(String s) {
        System.out.println(s);
        //logger.info(s);
    }
}
