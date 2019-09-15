package kc.ml.dnn.network;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class Layer {

    private List<Neuron> neurons = new ArrayList<>();

    public Layer(int neuronCount) {
        this(neuronCount, ActivationFunction.IDENTITY);
    }

    public Layer(int neuronCount, ActivationFunction activationFunction) {
        for (int i = 0; i < neuronCount; i++) {
            neurons.add(new Neuron(activationFunction));
        }
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public void fireNeurons() {
        for (Neuron neuron : neurons) {
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

    public void clearNeurons() {
        for (Neuron neuron : neurons) {
            neuron.clear();
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
