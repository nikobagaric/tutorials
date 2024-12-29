#include <iostream>
#include <vector>
#include <cmath>

std::vector<double> predict(const std::vector<double> &x, double m, double b)
{
    /*
     * A function that takes in a set of points on the x axis and
     * applies m*x+b to each point; finally returning a set of
     * predictions in a vector
     */
    std::vector<double> predictions;
    for (double xi : x)
    {
        predictions.push_back(m * xi + b);
    }
    return predictions;
}

double meanSquaredError(const std::vector<double> &y_pred,
                        const std::vector<double> &y_true)
{
    /*
     * A function that takes in a set of true values
     * and a set of predicted values and calculated the MSE
     */
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++)
    {
        sum += std::pow(y_pred[i] - y_true[i], 2);
    }
    return sum / y_true.size();
}

void gradientDescent(const std::vector<double> &x, const std::vector<double> &y,
                     double &m, double &b, double learning_rate)
{
    /*
    * A function for the gradient descent optimization algorithm
    * Takes in the set of 
    */
    double m_gradient = 0.0;
    double b_gradient = 0.0;
    size_t n = x.size();

    for(size_t i = 0; i < n; ++i) {
        double prediction = m * x[i] + b;
        m_gradient += -2.0 * x[i] * (y[i] - prediction);
        b_gradient += -2.0 * (y[i] - prediction);
    }

    m -= (m_gradient / n) * learning_rate;
    b -= (b_gradient / n) * learning_rate;
}

int main() {
    // Example dataset: y = 2x + 1
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {3, 5, 7, 9, 11};

    // Initialize parameters
    double m = 0.0; // slope
    double b = 0.0; // intercept
    double learning_rate = 0.01;
    int epochs = 100;

    // Train the model
    for (int i = 0; i < epochs; ++i) {
        std::vector<double> y_pred = predict(x, m, b);
        double loss = meanSquaredError(y, y_pred);
        gradientDescent(x, y, m, b, learning_rate);

        if (i % 10 == 0) { // Print progress every 10 epochs
            std::cout << "Epoch " << i << ": Loss = " << loss << ", m = " << m << ", b = " << b << "\n";
        }
    }

    // Test the model
    std::cout << "\nFinal Model: y = " << m << "x + " << b << "\n";
    std::vector<double> test_x = {6, 7, 8};
    std::vector<double> predictions = predict(test_x, m, b);
    for (size_t i = 0; i < test_x.size(); ++i) {
        std::cout << "For x = " << test_x[i] << ", predicted y = " << predictions[i] << "\n";
    }

    return 0;
}