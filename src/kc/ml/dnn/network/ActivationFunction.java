package kc.ml.dnn.network;

public enum ActivationFunction {

    IDENTITY {
        @Override
        public void apply(Neuron neuron) {
            return;
        }
    },
    RELU {
        @Override
        public void apply(Neuron neuron) {
            double value = neuron.getActivation();
            neuron.setActivation(value < 0 ? -value : value);
        }
    },
    SIGMOID { // Derivative: f(x) (1-f(x))
        @Override
        public void apply(Neuron neuron) {
            double value = neuron.getActivation();
            neuron.setActivation(1 / (1 + Math.exp(-value)));
        }
    };

    public abstract void apply(Neuron neuron);
}
