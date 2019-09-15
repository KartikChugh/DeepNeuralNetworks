package kc.ml.dnn.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron implements Component {

    private static Random randomLabeler = new Random(13); // TODO more elegant labeling

    private Integer id = randomLabeler.nextInt(1000);
    private double activation = 0;
    private List<Connection> connections = new ArrayList<>();;
    private ActivationFunction activationFunction;

    /**
     * Constructs neuron with specified activation function
     * @param activationFunction
     */
    Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public double getValue() {
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

    public void changeActivation(double delta) {
        activation += delta;
    }

    public double getActivation() {
        return activation;
    }

    public void clear() {
        activation = 0;
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