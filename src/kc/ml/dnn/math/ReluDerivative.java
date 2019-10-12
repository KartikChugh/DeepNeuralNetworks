package kc.ml.dnn.math;

import kc.ml.dnn.network.Neuron;

public class ReluDerivative extends FunctionDerivative {

    ReluDerivative(Neuron f) {
        super(f);
    }

    @Override
    double function(double x) {
        return x < 0 ? 0 : 1;
    }
}
