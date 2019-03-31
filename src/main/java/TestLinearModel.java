import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
public class TestLinearModel {
    public static void main(String[] args) throws Exception{
        int seed=123;
        double learningRate=0.01;
        int batchSize=50;
        int nEpochs=100;
        int numIn=2;int numOut=1;
        int nHidden=20;
        String filePathTrain=new ClassPathResource("linear_data_train.csv").getFile().getPath();
        String filePathTest=new ClassPathResource("linear_data_eval.csv").getFile().getPath();

        RecordReader rr=new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filePathTrain)));
        DataSetIterator dataSetTrain=new RecordReaderDataSetIterator(rr,batchSize,0,1);

        RecordReader rrTest=new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filePathTest)));
        DataSetIterator dataSetTest=new RecordReaderDataSetIterator(rrTest,batchSize,0,1);
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .list()
                    .layer(0, new DenseLayer.Builder()
                        .nIn(numIn)
                        .nOut(nHidden)
                        .activation(Activation.RELU).build())
                    .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(nHidden)
                        .nOut(numOut)
                        .activation(Activation.SIGMOID).build())
                .build();

        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        model.init();

        UIServer uiServer=UIServer.getInstance();
        StatsStorage statsStorage=new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        //model.setListeners(new ScoreIterationListener(10));

        for (int i = 0; i <nEpochs ; i++) {
            model.fit(dataSetTrain);
        }

        System.out.println("Model Evaluation....");

        Evaluation evaluation=new Evaluation(numOut);
        while(dataSetTest.hasNext()){
            DataSet dataSet=dataSetTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray labels=dataSet.getLabels();
            INDArray predicted=model.output(features,false);
            evaluation.eval(labels,predicted);
        }
        System.out.println(evaluation.stats());
        System.out.println("************************");
        System.out.println("Prédiction :");
        INDArray xs= Nd4j.create(new double[][]{
                {0.766837548998774,0.486441995062381},
                {0.332894760145352,-0.0112936854155695},
                {0.377466773756814,0.155504538357614}
        });
        INDArray ys=model.output(xs);
        System.out.println(ys);
    }
}
