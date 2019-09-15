package kc.ml.dnn.math;

import kc.ml.dnn.network.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Product {

    private List<Component> components = new ArrayList<>();

    public Product(Component... components) {
        for (Component component : components) {
            appendComponent(component);
        }
    }

    public static Product joinComponents(Product product, Component... newComponents) {
        List<Component> components = product.getComponents();
        components.addAll(Arrays.asList(newComponents));
        return new Product(components.toArray(new Component[0]));
    }

    public List<Component> getComponents() {
        return components;
    }

    public void appendComponent(Component component) {
        components.add(component);
    }

    public double multiply() {

        if (components.isEmpty()) return 0;

        double product = 1;
        for (Component component : components) {
            product *= component.getValue();
        }
        return product;
    }

    @Override
    public String toString() {
        return components.toString();
    }

    public Product copy() {
        return new Product(components.toArray(new Component[0]));
    }
}
