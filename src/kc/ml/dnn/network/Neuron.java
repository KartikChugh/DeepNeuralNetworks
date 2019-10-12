package kc.ml.dnn.network;

import kc.ml.dnn.math.ActivationFunction;
import kc.ml.dnn.math.Symbolic;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron implements Symbolic {

    private static Random randomLabeler = new Random(13); // TODO more elegant labeling

    private Integer id = randomLabeler.nextInt(1000);
    private double activation = 0;
    private double summation = 0;
    private List<Connection> connections = new ArrayList<>();;
    private ActivationFunction activationFunction;
    private boolean isBias = false; // TODO better bias implementation

    /**
     * Constructs neuron with specified activation function
     * @param activationFunction
     */
    Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    Neuron(boolean isBias) {
        this(ActivationFunction.IDENTITY);
        if (isBias) {
            this.isBias = true;
            this.activation = 1;
        }
    }

    @Override
    public double evaluate() {
        return getActivation();
    }

    /**
     * Apply activation function to the neuron
     */
    public void activate() {
        activationFunction.apply(this);
    }

    public void setActivation(double activation) {
        this.activation = activation;
    }

    public double getActivation() {
        return activation;
    }

    public void changeSummation(double delta) {
        summation += delta;
    }

    public void setSummation(double summation) {
        this.summation = summation;
    }

    public double getSummation() {
        return summation;
    }

    public void clear() {
        summation = 0;
        activation = 0;
    }

    public boolean isBias() {
        return isBias;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void addConnection(Connection connection) {
        connections.add(connection);
    }

    public List<Connection> getConnections() {
        return connections;
    }

    public void fireConnections() {
        for (Connection connection : connections) {
            connection.fire(activation);
        }
    }

    @Override
    public String toString() {
        return id + "";
        //return activation + "";
    }
}