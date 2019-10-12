package kc.ml.dnn.math;

import kc.ml.dnn.network.Neuron;

public abstract class FunctionDerivative implements Symbolic {

    private Neuron f;

    FunctionDerivative(Neuron f) {
        this.f = f;
    }

    abstract double function(double x);

    @Override
    public final double evaluate() {
        return function(f.getSummation());
    }

    @Override
    public final String toString() {
        return getClass().getSimpleName() + "@" + f;
    }
}
