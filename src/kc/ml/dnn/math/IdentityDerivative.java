package kc.ml.dnn.math;

import kc.ml.dnn.network.Neuron;

public class IdentityDerivative extends FunctionDerivative {

    IdentityDerivative(Neuron f) {
        super(f);
    }

    @Override
    double function(double x) {
        return 1;
    }
}
