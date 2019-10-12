package kc.ml.dnn.tests;

import kc.ml.dnn.network.Layer;
import kc.ml.dnn.network.Network;

import java.util.Arrays;

import static kc.ml.dnn.math.ActivationFunction.IDENTITY;
import static kc.ml.dnn.math.ActivationFunction.RELU;
import static kc.ml.dnn.math.ActivationFunction.SIGMOID;

public class SimpleSigmoidTest {

    // Define training data - model will fit inputs -> targets
    private static final double[][] inputsArray = {
/*            {0,0},
            {0,1},
            {1,0},*/
            {1,1},
    };
    private static final double[][] targetsArray = {
/*            {0},
            {1},
            {1},*/
            {0},
    };

    public static void main(String[] args) {

        double C_act = 1 / (1 + Math.exp(-2));
        double D_act = 1 / (1 + Math.exp(-2));
        double E = C_act + D_act;
        double E_act = 1 / (1 + Math.exp(-E));

        double J_deriv_E_act = 2*E_act;

        double S_deriv_E = E_act * (1-E_act);
        //System.out.println(S_deriv_E);

        double S_deriv_C = C_act * (1-C_act);
        double S_deriv_D = D_act * (1-D_act);
        //System.out.println(S_deriv_C);

        double J_deriv_ac = J_deriv_E_act * S_deriv_E * S_deriv_C;
        double J_deriv_bc = J_deriv_E_act * S_deriv_E * S_deriv_C;
        double J_deriv_ad = J_deriv_E_act * S_deriv_E * S_deriv_D;
        double J_deriv_bd = J_deriv_E_act * S_deriv_E * S_deriv_D;

        double J_deriv_ce = J_deriv_E_act * S_deriv_E * C_act;
        double J_deriv_de = J_deriv_E_act * S_deriv_E * D_act;

        System.out.println("ac " + J_deriv_ac);
        System.out.println("bc " + J_deriv_bc);

        System.out.println("ad " + J_deriv_ad);
        System.out.println("bd " + J_deriv_bd);

        System.out.println("ce " + J_deriv_ce);
        System.out.println("de " + J_deriv_de);

        //System.exit(0);

        // Builds network with one hidden layer
        Network network = new Network();
        //network.setSeed(1000);
        Layer i = new Layer(2);
        Layer h = new Layer(2, SIGMOID);
        Layer o = new Layer(1, SIGMOID);
        network.addLayers(i, h, o);

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

    }

}
