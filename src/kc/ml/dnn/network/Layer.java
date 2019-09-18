package kc.ml.dnn.network;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

// TODO better way of adding layers
public class Layer {

    private List<Neuron> neurons = new ArrayList<>();
    private List<Neuron> neuronsAndBias = new ArrayList<>();

    public Layer(int neuronCount) {
        this(neuronCount, false);
    }

    public Layer(int neuronCount, boolean addBias) {
        this(neuronCount, ActivationFunction.IDENTITY, addBias);
    }

    public Layer(int neuronCount, ActivationFunction activationFunction) {
        this(neuronCount, activationFunction, false);
    }

    public Layer(int neuronCount, ActivationFunction activationFunction, boolean addBias) {
        for (int i = 0; i < neuronCount; i++) {
            Neuron n = new Neuron(activationFunction);
            neurons.add(n);
            neuronsAndBias.add(n);
        }
        if (addBias) {
            neuronsAndBias.add(new Neuron(true));
        }
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public List<Neuron> getNeuronsAndBias() {
        return neuronsAndBias;
    }

    public void fireNeuronsAndBias() {
        for (Neuron neuron : neuronsAndBias) {
            neuron.fireConnections();
        }
    }

    /**
     *
     */
    public void activateNeurons() {
        for (Neuron neuron : neurons) {
            neuron.activate();
        }
    }

    // FIXME outdated
    @Override
    public String toString() {
        final DecimalFormat df = new DecimalFormat("#.###");

        StringBuilder sb = new StringBuilder();

        for (Neuron neuron : neurons) {
            for (Connection connection : neuron.getConnections()) {
                sb.append(df.format(connection.getWeight())).append(" ");
            }
            sb.append(" || ");
        }
        return sb.toString();
    }
}
