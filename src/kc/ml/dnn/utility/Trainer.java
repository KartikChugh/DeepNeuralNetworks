package kc.ml.dnn.utility;

import kc.ml.dnn.math.Product;
import kc.ml.dnn.math.ProductSum;
import kc.ml.dnn.network.Connection;
import kc.ml.dnn.network.Layer;
import kc.ml.dnn.network.Network;
import kc.ml.dnn.network.Neuron;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Trainer {

    /**
     * Stores symbolic derivatives of all neurons w.r.t. all respective, preceding connections
     *
     * N -> (C -> P)..., where dN/dC = P
     */
    private static Map<Neuron, Map<Connection, ProductSum>> symbolicDerivatives = new HashMap<>();

    /**
     * Stores the gradient of the loss function w.r.t. connections
     *
     * Each double is a partial derivative that helps determine how much to adjust its corresponding weight
     *
     * C -> x, where dJ/dC -> x
     */
    private static Map<Connection, Double> numericalGradient = new HashMap<>();

    /**
     * Network to train
     */
    private static Network network;

    /**
     * Prevents instantiation
     */
    private Trainer() {}

    /**
     * Trains the network on the provided examples
     *
     * Builds symbolic derivatives of output neurons w.r.t. each connection in network,
     * and then runs training epochs.
     *
     * @param inputsArray array of input vectors
     * @param targetsArray array of target vectors
     * @param learningRate magnitude of weight adjustments
     */
    public static void train(Network aNetwork, double[][] inputsArray, double[][] targetsArray, double learningRate, double epochs) {
        symbolicDerivatives.clear();
        numericalGradient.clear();
        network = aNetwork;

        validateExamples(inputsArray, targetsArray);
        buildSymbolicDerivatives();
        System.out.println("EPOCH, LOSS");
        for (int epoch = 0; epoch < epochs; epoch++) {
            epoch(inputsArray, targetsArray, learningRate);
            System.out.println(epoch + ", " + totalLoss(inputsArray, targetsArray));
        }
    }

    /**
     * Validates training examples for sizes consistent with eachother and
     * the network's input/output layers
     *
     * @param inputsArray array of input vectors
     * @param targetsArray array of target vectors
     */
    private static void validateExamples(double[][] inputsArray, double[][] targetsArray) {
        if (inputsArray.length != targetsArray.length) {
            throw new IllegalArgumentException("Training set size inconsistent: "
                    + inputsArray.length + ", "
                    + targetsArray.length);
        }

        for (double[] inputs : inputsArray) {
            if (inputs.length != network.getInputNeurons().size()) {
                throw new IllegalArgumentException("Input vector size doesn't match input layer: "
                        + Arrays.toString(inputs));
            }
        }

        for (double[] targets : targetsArray) {
            if (targets.length != network.getOutputNeurons().size()) {
                throw new IllegalArgumentException("Target vector size doesn't match output layer: "
                        + Arrays.toString(targets));
            }
        }
    }

    /**
     * Build symbolic derivatives of neurons w.r.t. preceding connections,
     * including through application of the chain rule when possible
     */
    private static void buildSymbolicDerivatives() {
        for (Layer left : network.getLayers()) {
            for (Neuron B : left.getNeuronsAndBias()) {
                for (Connection bc : B.getConnections()) {
                    Neuron C = bc.getNeuronForward();
                    buildSymbolicDerivative(C, bc, B);
                    if (symbolicDerivatives.containsKey(B)) {
                        chainDerivatives(C, bc, B);
                    }
                }
            }
        }
    }

    /**
     * Creates/adds to registry of derivatives of forward neuron
     *
     * C -> (bc -> B, ...), where d.C/d.bc = B
     */
    private static void buildSymbolicDerivative(Neuron C, Connection bc, Neuron B) {
        Map<Connection, ProductSum> derivsOfC = symbolicDerivatives.getOrDefault(C, new HashMap<>());
        derivsOfC.put(bc, new ProductSum(new Product(B))); // bc -> B
        symbolicDerivatives.putIfAbsent(C, derivsOfC); // register C if not done already
    }

    /**
     * Given - d.C/d.bc = B
     * For - d.B/d.ab = A
     * Then - d.C/d.ab = ... + A*bc
     * Proof: ... + B*bc = C, ... + A*ab = B --> ... + A*ab*bc = C
     * @param C forward neuron
     * @param B preceding neuron that has its own derivatives
     * @param bc preceding connection
     */
    private static void chainDerivatives(Neuron C, Connection bc, Neuron B) {
        Map<Connection, ProductSum> derivsOfB = symbolicDerivatives.get(B);
        for (Connection ab : derivsOfB.keySet()) {
            Product A = derivsOfB.get(ab).getProducts().get(0);
            Product A_bc = Product.joinComponents(A, bc);

            Map<Connection, ProductSum> derivsOfC = symbolicDerivatives.get(C);
            ProductSum fullDerivOfC_ab = derivsOfC.getOrDefault(ab, new ProductSum());
            fullDerivOfC_ab.appendProduct(A_bc);
            derivsOfC.putIfAbsent(ab, fullDerivOfC_ab);
        }
    }

    /**
     * A single epoch of the learning process
     *
     * @param inputsArray
     * @param targetsArray
     * @param learningRate
     */
    private static void epoch(double[][] inputsArray, double[][] targetsArray, double learningRate) {
        for (int i = 0; i < inputsArray.length; i++) {

            // Use current input/target vectors
            final double[] inputs = inputsArray[i];
            final double[] targets = targetsArray[i];

            // Get predictions without clearing neurons afterwards
            double[] predictions = network.feedforward(inputs);

            // Perform gradient descent
            double[] lossOutputDerivatives = computeLossOutputDerivatives(targets, predictions);
            computeGradient(lossOutputDerivatives);
            adjustWeights(learningRate);

            // Clear network
            network.clear();
        }

        // Adjust weights and clean up
        adjustWeights(learningRate);
        numericalGradient.clear();
    }

    /**
     * Computes derivatives of loss function w.r.t. output neurons
     * @param targets
     * @param predictions
     * @return
     */
    private static double[] computeLossOutputDerivatives(double[] targets, double[] predictions) {
        double[] lossOutputDerivatives = new double[targets.length];
        for (int i = 0; i < targets.length; i++) {
            lossOutputDerivatives[i] = 2*(predictions[i] - targets[i]);
        }
        return lossOutputDerivatives;
    }

    /**
     * Computes gradient of loss function w.r.t. each weight
     *
     * Multiplies derivative of output neuron w.r.t. weight by derivative of
     * loss function w.r.t. output
     * @param lossDerivatives
     */
    private static void computeGradient(double[] lossDerivatives) {
        final List<Neuron> outputNeurons = network.getOutputNeurons();
        for (Neuron outputNeuron : outputNeurons) {
            Map<Connection, ProductSum> derivs = symbolicDerivatives.get(outputNeuron);
            for (Connection connection : derivs.keySet()) {
                ProductSum symbolicDerivative = derivs.get(connection);
                final double outputDerivative = symbolicDerivative.sum();
                final double lossDerivative = lossDerivatives[outputNeurons.indexOf(outputNeuron)];

                final double derivative = (outputDerivative * lossDerivative);
                double cumulativeDerivative = numericalGradient.getOrDefault(connection, 0.0);
                cumulativeDerivative += derivative;
                numericalGradient.put(connection, cumulativeDerivative);
            }
        }
    }

    /**
     *  Adjusts each weight according to the gradient of the cost function
     *  and the provided learning rate
     * @param learningRate
     */
    private static void adjustWeights(double learningRate) {
        for (Connection c : numericalGradient.keySet()) {
            double delta = learningRate * -numericalGradient.get(c);
            c.changeWeight(delta);
        }
    }

    /**
     * Computes square deviation of a prediction from a target
     * This is the error pertaining to a single output neuron
     * @param target the ground truth value the network should fit to
     * @param prediction the value predicted by the network
     * @return the prediction error
     */
    private static double error(double target, double prediction) {
        double difference = target - prediction;
        return difference * difference;
    }

    /**
     * Computes loss for one set of inputs and targets (training example)
     * @param inputs the values fed into network
     * @param targets the values the network should output
     * @return loss for example
     */
    private static double exampleLoss(double[] inputs, double[] targets) {
        double exampleLoss = 0;
        double[] predictions = network.predict(inputs);
        for (int j = 0; j < targets.length; j++) {
            double target = targets[j];
            double prediction = predictions[j];
            exampleLoss += error(target, prediction);
        }
        return exampleLoss;
    }

    /**
     * Computes loss for all sets of inputs and targets (all training examples)
     *
     * Implementation is Mean Squared Error - http://mccormickml.com/2014/03/04/gradient-descent-derivation/
     * @param inputsArray each element is a set of input values
     * @param targetsArray each element is a set of output values
     * @return total loss for all examples
     */
    private static double totalLoss(double[][] inputsArray, double[][] targetsArray) {
        double allLoss = 0;
        for (int i = 0; i < inputsArray.length; i++) {
            double[] inputs = inputsArray[i];
            double[] targets = targetsArray[i];
            allLoss += exampleLoss(inputs, targets);
        }
        return allLoss/inputsArray.length;
    }

}
