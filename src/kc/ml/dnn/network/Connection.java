package kc.ml.dnn.network;

import kc.ml.dnn.math.Symbolic;

import java.text.DecimalFormat;

public class Connection implements Symbolic {

    private Neuron neuronForward;
    private double weight;

    Connection(Neuron neuronForward, double weight) {
        this.neuronForward = neuronForward;
        this.weight = weight;
    }

    @Override
    public double evaluate() {
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
        neuronForward.changeSummation(activation * weight);  // add weighted left Neuron value to right Neuron
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
        final DecimalFormat df = new DecimalFormat("#.###");//#####
        return Integer.toHexString(hashCode()).substring(0,3) + ", " + df.format(weight);
    }
}