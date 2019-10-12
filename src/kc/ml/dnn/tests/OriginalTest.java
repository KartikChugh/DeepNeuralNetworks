package kc.ml.dnn.tests;

import kc.ml.dnn.math.ActivationFunction;
import kc.ml.dnn.network.Layer;
import kc.ml.dnn.network.Network;

import static kc.ml.dnn.math.ActivationFunction.*;

public class OriginalTest {

    public static void main(String[] args) {
        Network network = new Network();
        network.setSeed(111111);
        Layer i = new Layer(2);
        Layer h = new Layer(3);
        Layer h2 = new Layer(3);
        Layer o = new Layer(2);
        network.addLayers(i,h,h2,o);

        System.out.println(network);

        network.train(null, null, 0, 0);
    }

}
