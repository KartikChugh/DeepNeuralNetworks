package kc.ml.dnn.tests;

import kc.ml.dnn.network.Layer;
import kc.ml.dnn.network.Network;

import java.util.Arrays;

import static kc.ml.dnn.math.ActivationFunction.IDENTITY;
import static kc.ml.dnn.math.ActivationFunction.RELU;
import static kc.ml.dnn.math.ActivationFunction.SIGMOID;

public class NonlinearModelTest {

    // Define training data - model will fit inputs -> targets
    private static final double[][] inputsArray = {
            {0,0},
            {0,1},
            {1,0},
            {1,1},
    };
    private static final double[][] targetsArray = {
            {0},
            {1},
            {1},
            {0},
    };

    public static void main(String[] args) {

        // Builds network with three hidden layers
        Network network = new Network();
        //network.setSeed(1000);
        Layer i = new Layer(2, true);
        Layer h = new Layer(10, IDENTITY, true);
        Layer h2 = new Layer(10, RELU, true);
        Layer h3 = new Layer(10, RELU , true);
        Layer o = new Layer(1, SIGMOID);
        network.addLayers(i, h, h2, h3, o);

        // Displays network state before training
        System.out.println("NETWORK:");
        System.out.println(network);

        System.out.println("PREDICTIONS:");
        for (double inputs[] : inputsArray) {
            System.out.print(Arrays.toString(inputs) + " -> ");
            System.out.println(Arrays.toString(network.predict(inputs)));
        }

        // Trains network
        System.out.println();
        System.out.println("TRAINING:");
        network.train(inputsArray, targetsArray, 0.1, 300);
        System.out.println();

        // Display network state after training
        System.out.println("NETWORK:");
        System.out.println(network);
        System.out.println();

        System.out.println("PREDICTIONS:");
        for (double inputs[] : inputsArray) {
            System.out.print(Arrays.toString(inputs) + " -> ");
            System.out.println(Arrays.toString(network.predict(inputs)));
        }

/*        final double[] f0 = network.predict(new double[]{0});
        final double[] f1 = network.predict(new double[]{1});
        System.out.println("SLOPE: " + (f1[0]-f0[0]));
        System.out.println("y-intercept: " + f0[0]);
        System.out.println();*/

    }

}
