package kc.ml.dnn.network;

import java.text.DecimalFormat;

public class Connection implements Component {

    private Neuron neuronForward;
    private double weight;

    Connection(Neuron neuronForward, double weight) {
        this.neuronForward = neuronForward;
        this.weight = weight;
    }

    @Override
    public double getValue() {
        return getWeight();
    }

    public Neuron getNeuronForward() {
        return neuronForward;
    }

    /**
     *
     * @param activation
     */
    public void fire(double activation) {
        neuronForward.changeActivation(activation * weight);  // add weighted left Neuron value to right Neuron
    }

    public double getWeight() {
        return weight;
    }

    public void changeWeight(double delta) {
        weight += delta;
    }

    // TODO improve
    @Override
    public String toString() {
        final DecimalFormat df = new DecimalFormat("#.########");
        return df.format(weight);
    }
}