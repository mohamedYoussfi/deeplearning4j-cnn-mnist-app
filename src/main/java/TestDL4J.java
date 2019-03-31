import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.omg.PortableInterceptor.INACTIVE;

public class TestDL4J {
    public static void main(String[] args) {
        int nRows = 2;
        int nColumns = 2;
// Create INDArray of all ones
        INDArray ones = Nd4j.ones(nRows, nColumns);
// pad the INDArray
        INDArray padded = Nd4j.pad(ones, new int[]{2,2}, Nd4j.PadMode.CONSTANT );
        System.out.println("### Padded ####");
        System.out.println(padded);
        System.out.println("-----------------------------");
        INDArray diag=Nd4j.diag(padded);
        System.out.println(diag);
        System.out.println("------------------------------");
        INDArray identity=Nd4j.eye(6);
        System.out.println(identity);
        System.out.println("-------------------------------");
        INDArray linSpace=Nd4j.linspace(5,15,11);
        System.out.println(linSpace);
        System.out.println("-------------------------------");
        INDArray linSpace2=Nd4j.linspace(1,25,25).reshape(5,5);
        System.out.println(linSpace2);
        System.out.println("-------------------");
        System.out.println(linSpace2.getDouble(3,3));
        System.out.println("--------------------");
        linSpace2.putScalar(new int[]{2,2},66);
        System.out.println(linSpace2);
        System.out.println("--------------------");

        INDArray res1=Nd4j.getExecutioner().execAndReturn(new Tanh(linSpace2));
        System.out.println(res1);
        System.out.println("------------------");
        double res2=Nd4j.getExecutioner().execAndReturn(new Sum(linSpace2)).getFinalResult().doubleValue();
        double res3=linSpace2.sumNumber().doubleValue();
        System.out.println(res2);
        System.out.println(res3);
        System.out.println("***************************");
        INDArray a=Nd4j.create(new double[]{1,2,3,4},new int[]{2,2});
        INDArray b=Nd4j.create(new double[][]{{1,2,6},{3,4,7}});
        //INDArray b=Nd4j.rand(new int[]{3,4});
        System.out.println(a);
        System.out.println(b);
        System.out.println(a.mmul(b));

        Nd4j.writeTxt(a,"f.txt");

        INDArray z=Nd4j.zeros(50,1);
        INDArray o=Nd4j.ones(50,1);
        INDArray ys=Nd4j.concat(0,z,o);
        System.out.println(ys);
    }
}
