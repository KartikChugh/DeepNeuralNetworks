package kc.ml.dnn.network;

public enum ActivationFunction {

    IDENTITY {
        @Override
        public double function(double x) {
            return x;
        }
    },
    RELU {
        @Override
        public double function(double x) {
            return x < 0 ? 0 : x;
        }
    },
    SIGMOID { // Derivative: f(x) (1-f(x))
        @Override
        public double function(double x) {
            return 1 / (1 + Math.exp(-x));
        }
    };

    public void apply(Neuron neuron) {
        neuron.setActivation(function(neuron.getActivation()));
    }

    abstract double function(double x);

}
