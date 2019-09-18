package kc.ml.dnn.tests;

import kc.ml.dnn.network.Layer;
import kc.ml.dnn.network.Network;
import kc.ml.dnn.network.Neuron;
import kc.ml.dnn.utility.Trainer;

import java.util.Arrays;

/**
 * Builds and trains a DNN model of a linear relationship.
 * See more information about the data here: https://www.desmos.com/calculator/ljc52eizwm
 */
public class LinearRegressionTest {

    // Define training data - model will fit inputs -> targets
    private static final double[][] inputsArray = {
            {-10},
            {-1},
            {2},
            {10},
            {20},
            {22},
    };
    private static final double[][] targetsArray = {
            {11},
            {-1},
            {21},
            {34},
            {41},
            {60},
    };

    public static void main(String[] args) {

        // Builds network with three hidden layers
        Network network = new Network();
        network.setSeed(1000);
        Layer i = new Layer(1);
        Layer h = new Layer(10);
        Layer h2 = new Layer(10);
        Layer h3 = new Layer(10);
        Layer o = new Layer(1);
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
        network.train(inputsArray, targetsArray, 0.0001, 120);
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

        System.out.println("SLOPE: " + Arrays.toString(network.predict(new double[]{1})));
        final double[] f0 = network.predict(new double[]{0});
        final double[] f1 = network.predict(new double[]{1});
        System.out.println("SLOPE: " + (f1[0]-f0[0]));
        System.out.println("y-intercept: " + f0[0]);
        System.out.println();

    }

}
