package kc.ml.dnn.math;

import java.util.ArrayList;
import java.util.List;

public class ProductSum {

    private List<Product> products = new ArrayList<>();

    public ProductSum(Product... products) {
        for (Product product : products) {
            appendProduct(product);
        }
    }

    public List<Product> getProducts() {
        return products;
    }

    public void appendProduct(Product product) {
        products.add(product);
    }

    /**
     * Computes each product and adds them all together
     * Ex: [AB,CDE,F] --> AB + CDE + F = 1*2 + 2*12*0 + 11 = 13
     * @return sum of products
     */
    public double sum() {
        double sum = 0;
        for (Product product : products) {
            sum += product.multiply();
        }
        return sum;
    }

    @Override
    public String toString() {
        return products.toString();
    }
}
