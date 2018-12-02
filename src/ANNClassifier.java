import java.io.File;
import java.util.Random;
import java.util.Scanner;

public class ANNClassifier {
    public static double lr = 0.01;

    public Random rand;

    private String test_path = System.getProperty("user.dir") +
            "\\test_files\\mnist_test.csv", train_path = System.getProperty("user.dir") +
            "\\train_files\\mnist_train.csv";

    private boolean[] test_rows;

    private double[][] outputLayer, hiddenLayer;

    private int hiddenLayerNodeCount = 32, epochMax = 13;

    public ANNClassifier(String test_path, String train_path, int trainCount) {
        this.test_path = test_path;
        this.train_path = train_path;
        this.rand = new Random();

        //auto populate with best solution and default train/test file(s).
        train(trainCount, true);
    }

    public ANNClassifier(String test_path) {
        this.test_path = test_path;
        this.rand = new Random();

        //TODO: change the number to recommended.
        train(5000, true);
        try {
            Scanner check = new Scanner(new File(
                    test_path)).useDelimiter("\n");
            check.nextLine();
            int lines = 0;
            while (check.hasNextLine()) {
                lines++;
                check.nextLine();
            }
            test(lines);
        }catch(Exception e){
            System.out.println("Training read error. ");
        }
    }

    private void randomize_rows(int amount, int size) {
        boolean[] temp = new boolean[size];
        int count = 0;
        while (count != amount) {
            int current = rand.nextInt(size);
            if (!temp[current]) {
                temp[current] = true;
                count++;
            }
        }
        this.test_rows = temp;
    }

    private static double[][] random_weights(int row, int col, Random rand){
        double[][] weight = new double[row][col];
        for(int r = 0; r < row; r++)
            for (int c = 0; c < col; c++)
                weight[r][c] = (rand.nextDouble() - 0.5)/10;
        return weight;
    }

    private void train(int trainSize, boolean print) {
        //if all else fails un-staticize methods
        try {
            randomize_rows(trainSize, 60000);
            //input's results
            double[][] results = populate_results(false, trainSize);
            //input
            double[][] trainings = initial_data(false, trainSize);

            //w0
            hiddenLayer = random_weights(trainings[0].length, hiddenLayerNodeCount, this.rand);

            //w1
            outputLayer = random_weights(hiddenLayerNodeCount, 10, this.rand);


            for (int epochs = 0; epochs < epochMax; epochs++) {
                for(int row = 0; row < trainSize; row++) {
                    //forward propagation
                    double[][] currentHiddenLayer = sigmoid(dotProduct(trainings[row], this.hiddenLayer));


                    double[][] currentOutputLayer = sigmoid(dotProduct(currentHiddenLayer, this.outputLayer));

                    double[][] outputError = subtract(results[row], currentOutputLayer[0]);

                    //backpropagation
                    double[][] outputLayerChange = multiply(sigmoidDerivative(currentOutputLayer), outputError);
                    double[][] hiddenLayerError = dotProduct(outputLayerChange, transpose(outputLayer));
                    double[][] hiddenLayerChange = multiply(sigmoidDerivative(currentHiddenLayer), hiddenLayerError);

                    //scaled by learning rate to update bias
                    this.outputLayer = add(this.outputLayer, scale(dotProduct(transpose(currentHiddenLayer), outputLayerChange), lr));
                    this.hiddenLayer = add(this.hiddenLayer, scale(dotProduct(transposeSingleRow(trainings[row]), hiddenLayerChange), lr));
                    if(print)
                        System.out.println("Training: " + ((1 + row + epochs * trainSize) * 100.0 / (trainSize * epochMax)) + "%");
                }

            }
            if(print)
                System.out.println("Finished Training");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    private double[][] transposeSingleRow(double[] input) {
        double[][] output = new double[input.length][1];
        for(int index = 0; index < input.length; index++)
            output[index][0] = input[index];
        return output;
    }

    private static double[][] multiply(double[][] array1, double[][] array2) {
        double[][] output = new double[1][array1[0].length];
        for(int index = 0; index < output[0].length; index++)
            output[0][index] = array1[0][index] * array2[0][index];
        return output;
    }

    private static double[][] scale(double[][] array, double scalar) {
        for(int row = 0; row < array.length; row++)
            for(int column = 0; column < array[0].length; column++)
                array[row][column] *= scalar;
        return array;
    }

    private static double[][] add(double[][] array1, double[][] array2){
        double[][] output = new double[array1.length][array1[0].length];
        if(array1.length == array2.length && array1[0].length == array2[0].length){
            for(int row = 0; row < array1.length; row++)
                for(int column = 0; column < array1[0].length; column++)
                    output[row][column] = array1[row][column] + array2[row][column];
        }
        return output;
    }

    private static double[][] transpose(double[][] input){
        double[][] output = new double[input[0].length][input.length];
        for(int row = 0; row < input.length; row++)
            for(int column = 0; column < input[0].length; column++)
                output[column][row] = input[row][column];
        return output;
    }

    private static double[][] subtract(double[] array1, double[] array2) {
        double[][] output = new double[1][array2.length];
        if(array1.length == array2.length)
            for (int index = 0; index < array1.length; index++)
                    output[0][index] = array1[index] - array2[index];
        return output;
    }

    private double[][] populate_results(boolean isTest, int rowCount) {
        try {
            double[][] results = new double[rowCount][10];
            Scanner rows;
            if(isTest)
                rows = new Scanner(new File(
                        test_path)).useDelimiter("\n");
            else
                rows = new Scanner(new File(
                    train_path)).useDelimiter("\n");
            rows.nextLine();
            int count = 0, index = 0;
            while (rows.hasNext() && index != rowCount) {
                if(test_rows[count]){
                    String result = rows.next().split(",")[0];
                    results[index][Integer.parseInt(result)] = 1.0;
                    index++;
                }
                else
                    rows.next();
                count++;
            }
            return results;
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
        return new double[0][0];
    }

    private double[][] initial_data(boolean isTest, int max) {
        try {
            Scanner rows;
            if(isTest)
                rows = new Scanner(new File(
                        test_path)).useDelimiter("\n");
            else
                rows = new Scanner(new File(
                    train_path)).useDelimiter("\n");
            int length = rows.next().split(",").length - 1;
            double[][] trainings = new double[max][length];
            rows.nextLine();
            int count = 0, index = 0;
            while (rows.hasNext() && index != max) {
                if (test_rows[count]) {
                    String[] line = rows.next().split(",");
                    for (int column = 1; column < line.length; column++)
                        trainings[index][column - 1] =
                                Integer.parseInt(line[column].replace("\r", ""))/255.0;
                    index++;
                }else
                    rows.next();
                count++;
            }
            return trainings;
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
        return new double[0][0];
    }

    public double test(int testSize) {
        try {
            randomize_rows(testSize, 10000);
            //here
            double[][] results = populate_results(true, testSize);
            double[][] trainings = initial_data(true, testSize);

            int correct = 0;
            for(int index = 0; index < results.length; index++) {
                int output = calculate(trainings[index]);
                System.out.println("Image " + (index + 1) + ": \t" +
                        output + "\tAccurate output is: " + maxIndex(results[index]));
                if(output == maxIndex(results[index]))
                    correct++;
            }
            System.out.println("Average Correctness: " + (100.0 * correct /
                    trainings.length) + "%");
            return (100.0 * correct / trainings.length);
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
        return -1.0;
    }

    //still need to test this
    private int calculate(double[] line) {
        double[][] hiddenLayer = sigmoid(dotProduct(line, this.hiddenLayer));
        double[][] currentOutputLayer = sigmoid(dotProduct(hiddenLayer,
                outputLayer));
        return maxIndex(currentOutputLayer[0]);
    }

    private static int maxIndex(double[] input){
        int maxIndex = 0;
        for(int index = 0; index < input.length; index++)
            if(input[maxIndex] < input[index])
                maxIndex = index;
        return maxIndex;
    }

    public static double[][] dotProduct(double[][] array1, double[][] array2) {
        if (array1[0].length == array2.length) {
            double[][] calculation = new double[array1.length][array2[0].length];
            for (int row = 0; row < calculation.length; row++)
                for (int column = 0; column < calculation[0].length; column++)
                    calculation[row][column] = innerCalculation(array1, array2,
                            column, row);
            return calculation;
        } else {
            return new double[0][0];
        }
    }

    private static double[][] dotProduct(double[] array1, double[][] array2) {
        if (array1.length == array2.length) {
            double[][] calculation = new double[1][array2[0].length];
            for (int column = 0; column < calculation[0].length; column++)
                calculation[0][column] = innerCalculation(array1, array2, column);
            return calculation;
        } else
            return new double[0][0];
    }

    private static double innerCalculation(double[][] array1,
                                           double[][] array2, int column, int
                                                   row) {
        double calculation = 0.0;
            for (int index = 0; index < array1[0].length; index++)
                calculation += array1[row][index] * array2[index][column];
        return calculation;
    }

    private static double innerCalculation(double[] array1,
                                           double[][] array2, int column) {
        double calculation = 0.0;
        for (int index = 0; index < array1.length; index++)
            calculation += array1[index] * array2[index][column];
        return calculation;
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double[][] sigmoid(double[][] input) {
        double[][] calculation = new double[input.length][input[0].length];
        for (int row = 0; row < calculation.length; row++)
            for (int column = 0; column < calculation[0].length; column++)
                calculation[row][column] = sigmoid(input[row][column]);
        return calculation;
    }

    private static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private static double[][] sigmoidDerivative(double[][] input) {
        double[][] calculation = new double[input.length][input[0].length];
        for (int row = 0; row < calculation.length; row++)
            for (int column = 0; column < calculation[0].length; column++)
                calculation[row][column] = sigmoidDerivative(
                        input[row][column]);
        return calculation;
    }
}
