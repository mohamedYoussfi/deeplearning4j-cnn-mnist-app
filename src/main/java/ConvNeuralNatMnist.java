import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class ConvNeuralNatMnist {

    private static Logger logger= LoggerFactory.getLogger(ConvNeuralNatMnist.class);

    public static void main(String[] args) throws IOException {
        String basePath=System.getProperty("user.home")+"/mnist_png";
        System.out.println(basePath);
        int height=28;int width=28;
        int channels=1;// signe channel for graysacle image
        int outputNum=10;// 10 digits classification
        int batchSize=54;
        int epochCount=1;
        int seed =1234;
        Map<Integer,Double> learningRateByIterations=new HashMap<>();
        learningRateByIterations.put(0,0.06);
        learningRateByIterations.put(200,0.05);
        learningRateByIterations.put(600,0.028);
        learningRateByIterations.put(800,0.006);
        learningRateByIterations.put(1000,0.001);
        double quadraticError=0.0005;
        double momentum=0.9;
        Random randomGenNum=new Random(seed);
        logger.info("Data Load and vectorisation");

        File trainDataFile=new File(basePath+"/training");
        FileSplit trainFileSplit=new FileSplit(trainDataFile, NativeImageLoader.ALLOWED_FORMATS,randomGenNum);
        ParentPathLabelGenerator labelMarker=new ParentPathLabelGenerator();
        ImageRecordReader trainImageRecordReader=new ImageRecordReader(height,width,channels,labelMarker);
        trainImageRecordReader.initialize(trainFileSplit);
        int labelIndex=1;

        DataSetIterator trainDataSetIterator=new RecordReaderDataSetIterator(trainImageRecordReader,batchSize,labelIndex,outputNum);
        DataNormalization scaler=new ImagePreProcessingScaler(0,1);
        scaler.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(scaler);

        File testDataFile=new File(basePath+"/testing");
        FileSplit testFileSplit=new FileSplit(testDataFile, NativeImageLoader.ALLOWED_FORMATS,randomGenNum);
        ImageRecordReader testImageRecordReader=new ImageRecordReader(height,width,channels,labelMarker);
        testImageRecordReader.initialize(testFileSplit);
        DataSetIterator testDataSetIterator=new RecordReaderDataSetIterator(testImageRecordReader,batchSize,labelIndex,outputNum);
        trainDataSetIterator.setPreProcessor(scaler);
        logger.info("Neural Network Model Configuation");

        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(quadraticError)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION,learningRateByIterations),momentum))
                .weightInit(WeightInit.XAVIER)
                .list()
                    .layer(0,new ConvolutionLayer.Builder()
                            .kernelSize(3,3)
                            .nIn(channels)
                            .stride(1,1)
                            .nOut(20)
                            .activation(Activation.RELU).build())
                     .layer(1, new SubsamplingLayer.Builder()
                             .poolingType(SubsamplingLayer.PoolingType.MAX)
                             .kernelSize(2,2)
                             .stride(2,2)
                             .build())
                    .layer(2, new ConvolutionLayer.Builder(3,3)
                            .stride(1,1)
                            .nOut(50)
                            .activation(Activation.RELU)
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2,2)
                            .stride(2,2)
                            .build())
                    .layer(4, new DenseLayer.Builder()
                            .activation(Activation.RELU)
                            .nOut(500)
                            .build())
                    .layer(5,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .nOut(outputNum)
                            .build())
                    .setInputType(InputType.convolutionalFlat(height,width,channels))
                .backpropType(BackpropType.Standard)
                .build();
        System.out.println(configuration.toJson());
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        model.init();

        UIServer uiServer=UIServer.getInstance();
        StatsStorage statsStorage=new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        logger.info("Total params:"+model.numParams());

        for (int i = 0; i < epochCount; i++) {
            model.fit(trainDataSetIterator);
            logger.info("End of epoch "+i);
            Evaluation evaluation=model.evaluate(testDataSetIterator);
            logger.info(evaluation.stats());
            trainDataSetIterator.reset();
            testDataSetIterator.reset();
        }

        logger.info("Saving model ....");
        ModelSerializer.writeModel(model,new File(basePath+"/model.zip"),true);

    }
}
