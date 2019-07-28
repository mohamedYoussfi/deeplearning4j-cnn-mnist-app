import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.scene.text.Font;
import javafx.stage.Stage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class MnistClassifierUI extends Application {

  private static final String basePath = System.getProperty("user.home") + "/mnist_png";
  private final int canvasWidth = 256;
  private final int canvasHeight = 256;
  private MultiLayerNetwork net; // trained model
  private INDArray currentImage;
  private Canvas canvas;

  private List<XYChart.Data> data;

    private BarChart<String,Double> barChart;

  public MnistClassifierUI() throws IOException {
    File model = new File(basePath + "/model.zip");
    if (!model.exists())      throw new IOException("Can't find the model");
    net = ModelSerializer.restoreMultiLayerNetwork(model);
  }

  public static void main(String[] args) throws Exception {
    launch();
  }

  public void drawChart(){
    CategoryAxis xAxis=new CategoryAxis();
    NumberAxis yAxis=new NumberAxis(0,1,0.1);
    barChart=new BarChart(xAxis,yAxis);
    barChart.setCategoryGap(10); barChart.setBarGap(10);
    barChart.setTitle("Digits Predictions");
    xAxis.setLabel("Digits");  yAxis.setLabel("Prediction");
    XYChart.Series series1=new XYChart.Series();
    series1.setName("Probability");
    double[]values=new double[]{0.1,0.1,0.2,0.1,0.1,0.05,0.05,0.45,0.25,0.1} ;
    final String[] categoriesNames = new String[] {"0", "1", "2","3", "4", "5","6", "7", "8","9"};
    xAxis.getCategories().setAll(categoriesNames);
    data=new ArrayList<>();
    for(int i=0;i<categoriesNames.length;i++){
      data.add(new XYChart.Data(categoriesNames[i],values[i]));
    }
    series1.getData().addAll(data);
    barChart.getData().addAll(series1);
  }

  @Override
  public void start(Stage stage) throws Exception {
      canvas = new Canvas(canvasWidth, canvasHeight);
      GraphicsContext ctx = canvas.getGraphicsContext2D();
      ImageView imgView = new ImageView(); imgView.setFitHeight(28); imgView.setFitWidth(28);
      ctx.setLineWidth(10);
      ctx.setLineCap(StrokeLineCap.SQUARE);
      Label labelMessage=new Label("Draw a digit and Tape Enter de predict"+System.lineSeparator()+" Right Mouse click to clear");
      Label lblResult = new Label(); lblResult.setFont(new Font(30));
      Button buttonLearn=new Button("Learn This Gigit to the Model");
      Button buttonSaveModel=new Button("Save the Current Model");
    GridPane gridPane=new GridPane();int index=-1;
    gridPane.setDisable(true);
    for(int i=0;i<2;i++)
      for(int j=0;j<5;j++){
        Button button=new Button(String.valueOf(++index));
        gridPane.add(button,i,j);
        button.setOnAction((evt)->{
          learnThisDigitToModel(evt);
        });
    }
      VBox vbBottom = new VBox(10, imgView, lblResult,buttonLearn,gridPane);
      vbBottom.setAlignment(Pos.TOP_LEFT);
      VBox vBoxCanvas=new VBox(10,labelMessage,canvas,buttonSaveModel);
      HBox hBox2=new HBox(5, vBoxCanvas, vbBottom);
      VBox root = new VBox(10,hBox2);
      root.setAlignment(Pos.CENTER); root.setPadding(new Insets(10));
      drawChart();
      root.getChildren().add(barChart);
      Scene scene = new Scene(root, 800, 600);
      stage.setScene(scene);
      stage.setTitle("Deep Learning Example with DL4J");
      stage.setResizable(false);
      stage.show();

      canvas.setOnMousePressed(e -> {
      ctx.setStroke(Color.WHITE);  ctx.beginPath();
      ctx.moveTo(e.getX(), e.getY());
      ctx.stroke();
      canvas.requestFocus();
    });
    canvas.setOnMouseDragged(e -> {
      ctx.setStroke(Color.WHITE);
      ctx.lineTo(e.getX(), e.getY());
      ctx.stroke();
      canvas.requestFocus();
    });
    canvas.setOnMouseClicked(e -> {
      if (e.getButton() == MouseButton.SECONDARY) {
        clear(ctx);
      }
      canvas.requestFocus();
    });
    canvas.setOnKeyReleased(e -> {
      if (e.getCode() == KeyCode.ENTER) {
        BufferedImage scaledImg = getScaledImage(canvas);
        imgView.setImage(SwingFXUtils.toFXImage(scaledImg, null));
        try {
          predictImage(scaledImg, lblResult);
        } catch (Exception e1) {
          e1.printStackTrace();
        }
      }
    });
    clear(ctx);
    canvas.requestFocus();


      buttonLearn.setOnAction(evt->{
          if(currentImage!=null) gridPane.setDisable(false);
      });

    buttonSaveModel.setOnAction(evt->{
      try {
        ModelSerializer.writeModel(net,new File(basePath+"/model.zip"),true);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });
  }


  private void learnThisDigitToModel(ActionEvent evt) {
    Button b=(Button) evt.getSource();
    int digit=Integer.parseInt(b.getText());
    Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
    alert.setTitle("Learn This Digit : "+digit);
    alert.setHeaderText("Learn This Digit : "+b.getText());
    alert.setContentText("Learn This Digit : "+b.getText());
    Optional<ButtonType> result = alert.showAndWait();
    double[] digits=new double[10];
    digits[digit]=1;
    if (result.get() == ButtonType.OK){
      INDArray ys= Nd4j.create(digits);
      net.fit(currentImage,ys);
    } else {
      // ... user chose CANCEL or closed the dialog
    }
    canvas.requestFocus();
  }

  private void clear(GraphicsContext ctx) {
    ctx.setFill(Color.BLACK);
    ctx.fillRect(0, 0, 300, 300);
  }

  private BufferedImage getScaledImage(Canvas canvas) {
    WritableImage writableImage = new WritableImage(canvasWidth, canvasHeight);
    canvas.snapshot(null, writableImage);
    Image tmp = SwingFXUtils.fromFXImage(writableImage, null).getScaledInstance(28, 28, Image.SCALE_SMOOTH);
    BufferedImage scaledImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
    Graphics graphics = scaledImg.getGraphics();
    graphics.drawImage(tmp, 0, 0, null);
    graphics.dispose();
    return scaledImg;
  }

  private void predictImage(BufferedImage img, Label lbl) throws IOException {
    NativeImageLoader loader = new NativeImageLoader(28, 28, 1, true);
    currentImage = loader.asRowVector(img);
    ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    scaler.transform(currentImage);
    INDArray output = net.output(currentImage);
    //lbl.setText("Prediction: " + net.predict(image)[0] + "\n " + output);
    lbl.setText("Prediction: " + net.predict(currentImage)[0]);
    double[] d=output.toDoubleVector();
    for(int i=0;i<data.size();i++){
        data.get(i).setYValue(d[i]);
    }
  }

}