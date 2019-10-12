package kc.ml.dnn.math;

import kc.ml.dnn.math.FunctionDerivative;
import kc.ml.dnn.math.IdentityDerivative;
import kc.ml.dnn.network.Neuron;

public enum ActivationFunction { // TODO make it easy for user to add custom functions

    IDENTITY {
        @Override
        double function(double x) {
            return x;
        }

        @Override
        public FunctionDerivative getDerivativeAt(Neuron f) {
            return new IdentityDerivative(f);
        }
    },

    RELU {
        @Override
        double function(double x) {
            return x < 0 ? 0 : x;
        }

        @Override
        public FunctionDerivative getDerivativeAt(Neuron f) {
            return new ReluDerivative(f);
        }
    },

    SIGMOID {
        @Override
        double function(double x) {
            return 1 / (1 + Math.exp(-x));
        }

        @Override
        public FunctionDerivative getDerivativeAt(Neuron f) {
            return new SigmoidDerivative(f);
        }
    };

    public void apply(Neuron neuron) {
        neuron.setActivation(function(neuron.getSummation()));
    }

    abstract double function(double x);

    public abstract FunctionDerivative getDerivativeAt(Neuron f);

}
