package kc.ml.dnn.network;

import kc.ml.dnn.utility.Trainer;

import java.util.*;

public class Network {

    /**
     * The layers of the network
     */
    private List<Layer> layers = new ArrayList<>();

    /**
     * Generates random weights
     */
    private Random weightInitializer = new Random();

    /**
     * Sets the seed used for random weight initialization
     * @param seed the seed
     */
    public void setSeed(long seed) {
        weightInitializer.setSeed(seed);
    }

    // GETTERS/SETTERS

    /**
     * Gets all layers
     * @return list of layers
     */
    public List<Layer> getLayers() {
        return layers;
    }

    /**
     * Gets the input layer
     * @return first layer
     */
    private Layer getInputLayer() {
        return layers.get(0);
    }

    /**
     * Gets the input layer's neurons
     * @return input neurons
     */
    public List<Neuron> getInputNeurons() {
        return getInputLayer().getNeurons();
    }

    /**
     * Gets the output layer
     * @return last layer
     */
    private Layer getOutputLayer() {
        return layers.get(layers.size()-1);
    }

    /**
     * Gets the output layer's neurons
     * @return output neurons
     */
    public List<Neuron> getOutputNeurons() {
        return getOutputLayer().getNeurons();
    }

    /**
     * Plugs values into input neurons
     * @param inputs values
     */
    private void setInputs(double[] inputs) {
        List<Neuron> inputNeurons = getInputNeurons();
        for (int i = 0; i < inputs.length; i++) {
            inputNeurons.get(i).setActivation(inputs[i]);
        }
    }

    /**
     * Gets the output layer's activations
     * @return activations
     */
    private double[] getOutputs() {
        return getOutputNeurons().stream().mapToDouble(Neuron::getActivation).toArray();
    }

    // LAYERS

    /**
     * Adds layers to the network
     * @param newLayers layers to add
     */
    public void addLayers(Layer... newLayers) {
        for (Layer newLayer : newLayers) {
            addLayer(newLayer);
        }
    }

    /**
     * Adds a layer and connects the preceding layer to it
     * @param newLayer layer to add
     */
    private void addLayer(Layer newLayer) {
        layers.add(newLayer);
        if (layers.size() >= 2) {
            Layer precedingLayer = layers.get(layers.size()-2);
            fullyConnectLayers(precedingLayer, newLayer);
        }
    }

    /**
     * Fully connects one layer to another
     * @param left preceding layer, which will propagate values forward
     * @param right succeeding layer, which will receive values
     */
    private void fullyConnectLayers(Layer left, Layer right) {
        for (Neuron leftNeuron : left.getNeurons()) {
            for (Neuron rightNeuron : right.getNeurons()) {
                leftNeuron.addConnection(new Connection(rightNeuron, weightInitializer.nextGaussian() * 0.1));
            }
        }
    }

    /**
     * Clears all neuron activations
     */
    public void clear() {
        for (Layer layer : layers) {
            layer.clearNeurons();
            for (Neuron neuron : layer.getNeurons()) {
                neuron.clear();
            }
        }
    }

    // PROPAGATION

    /**
     * Propagates input values forward and gets output values
     * @param inputs values
     */
    public double[] feedforward(double[] inputs) {
        setInputs(inputs);
        for (Layer layer : layers) {
            layer.activateNeurons();
            layer.fireNeurons();
        }
        return getOutputs();
    }

    /**
     * Makes predictions for an input vector
     *
     * Feeds forward and clears network
     * @param inputs values fed into the network
     * @return values output by the network
     */
    public double[] predict(double[] inputs) {
        double[] outputs = feedforward(inputs);
        clear();
        return outputs;
    }

    /**
     * Trains the network on the provided examples
     *
     * @param inputsArray array of input vectors
     * @param targetsArray array of target vectors
     * @param learningRate magnitude of weight adjustments
     */
    public void train(double[][] inputsArray, double[][] targetsArray, double learningRate, double epochs) {
        Trainer.train(this, inputsArray, targetsArray, learningRate, epochs);
    }

    // TODO improve
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
/*        for (Layer layer : layers) {
            sb.append(layer).append("\n");
        }*/
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                sb.append(neuron);
                sb.append(" >>");
                sb.append(neuron.getConnections());
                sb.append("  ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}