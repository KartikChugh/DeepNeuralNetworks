package kc.ml.dnn.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Product {

    private List<Symbolic> symbolics = new ArrayList<>();

    public Product(Symbolic... symbolics) {
        for (Symbolic symbolic : symbolics) {
            appendComponent(symbolic);
        }
    }

    public static Product joinComponents(Product product, Symbolic... newSymbolics) {
        List<Symbolic> symbolics = new ArrayList<>(product.getSymbolics());
        symbolics.addAll(Arrays.asList(newSymbolics));
        return new Product(symbolics.toArray(new Symbolic[0]));
    }

    public List<Symbolic> getSymbolics() {
        return symbolics;
    }

    public void appendComponent(Symbolic symbolic) {
        symbolics.add(symbolic);
    }

    public double multiply() {

        if (symbolics.isEmpty()) return 0;

        double product = 1;
        for (Symbolic symbolic : symbolics) {
            product *= symbolic.evaluate();
        }
        return product;
    }

    @Override
    public String toString() {
        return symbolics.toString();
    }

    public Product copy() {
        return new Product(symbolics.toArray(new Symbolic[0]));
    }
}
