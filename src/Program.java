import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class Program {
    private static int[] sizes = {1, 100, 500, 1000, 1500, 2000, 3000, 4000,
            5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000,
            60000};
    private static ANNClassifier ann;

    public static void main(String[] args){
        introLine();
        trainRun();
        closure();
    }

    private static void introLine(){
        System.out.println("Welcome to automated ANN tester. ");
        System.out.println("Learning Method is set to: Back-Propagation. ");
        System.out.println("Learning Rate is set to: " + ANNClassifier.lr);
        System.out.println("---------------------------------------------");
    }

    private static void trainRun(){

        String train_path = System.getProperty("user.dir") +
                "\\train_files\\mnist_train.csv";
        String test_path = System.getProperty("user.dir") +
                "\\test_files\\mnist_test.csv";
        String out_path = System.getProperty("user.dir") +
                "\\results33_19.csv";

        try {
            for(int index = 0; index < sizes.length; index++)
                for(int trial = 1; trial <= 3; trial++) {
                    System.out.println("Trial " + trial + " with training size " +
                            sizes[index] + ": ");
                    ann = new ANNClassifier(test_path, train_path, sizes[index]);
                    File out = new File(out_path);
                    out.createNewFile();
                    BufferedWriter output = new BufferedWriter(new FileWriter(out, true));
                    output.write(sizes[index] + "," + ann.test(1000) + "\n");
                    System.out.println("Finished Testing. ");
                    output.close();
                }
        }catch(Exception e){
            System.out.println("Error: " + e.getMessage());
        }

    }

    private static void closure(){
        System.out.println("Trials completed. Press Enter key to close...");
        try {
            System.in.read();
        } catch (Exception ioe) {
            System.out.println("Error in accepting enter key closure... " +
                    ioe.getMessage());
        }
    }
}
