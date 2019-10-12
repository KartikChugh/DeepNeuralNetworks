package kc.ml.dnn.math;

import kc.ml.dnn.network.Neuron;

public class SigmoidDerivative extends FunctionDerivative {

    SigmoidDerivative(Neuron f) {
        super(f);
    }

    @Override
    double function(double x) {
        double S = ActivationFunction.SIGMOID.function(x);
        return S * (1-S);
    }
}
