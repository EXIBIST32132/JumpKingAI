#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Activation function and its derivative
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

// Initialize random weights between -1 and 1
float random_weight() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

// Neural Network class
class NeuralNetwork {
public:
    NeuralNetwork();
    std::vector<float> forward(const std::vector<float>& inputs);
    void train(const std::vector<std::vector<float>>& training_data,
               const std::vector<std::vector<float>>& target_data,
               int epochs, float learning_rate);

private:
    int input_size = 8;
    int hidden1_size = 16;
    int hidden2_size = 16;
    int hidden3_size = 16;
    int output_size = 8;

    std::vector<float> hidden1;
    std::vector<float> hidden2;
    std::vector<float> hidden3;
    std::vector<float> outputs;

    std::vector<std::vector<float>> weights_input_hidden1;
    std::vector<std::vector<float>> weights_hidden1_hidden2;
    std::vector<std::vector<float>> weights_hidden2_hidden3;
    std::vector<std::vector<float>> weights_hidden3_output;

    std::vector<float> bias_hidden1;
    std::vector<float> bias_hidden2;
    std::vector<float> bias_hidden3;
    std::vector<float> bias_output;

    void initialize_weights();
};

NeuralNetwork::NeuralNetwork() {
    srand(static_cast<unsigned>(time(0)));

    hidden1.resize(hidden1_size);
    hidden2.resize(hidden2_size);
    hidden3.resize(hidden3_size);
    outputs.resize(output_size);

    initialize_weights();
}

void NeuralNetwork::initialize_weights() {
    weights_input_hidden1.resize(input_size, std::vector<float>(hidden1_size));
    weights_hidden1_hidden2.resize(hidden1_size, std::vector<float>(hidden2_size));
    weights_hidden2_hidden3.resize(hidden2_size, std::vector<float>(hidden3_size));
    weights_hidden3_output.resize(hidden3_size, std::vector<float>(output_size));

    bias_hidden1.resize(hidden1_size);
    bias_hidden2.resize(hidden2_size);
    bias_hidden3.resize(hidden3_size);
    bias_output.resize(output_size);

    for (auto& layer : {&weights_input_hidden1, &weights_hidden1_hidden2, &weights_hidden2_hidden3, &weights_hidden3_output}) {
        for (auto& row : *layer) {
            for (auto& weight : row) {
                weight = random_weight();
            }
        }
    }

    for (auto& biases : {&bias_hidden1, &bias_hidden2, &bias_hidden3, &bias_output}) {
        for (auto& bias : *biases) {
            bias = random_weight();
        }
    }
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& inputs) {
    for (int i = 0; i < hidden1_size; i++) {
        hidden1[i] = 0;
        for (int j = 0; j < input_size; j++) {
            hidden1[i] += inputs[j] * weights_input_hidden1[j][i];
        }
        hidden1[i] = sigmoid(hidden1[i] + bias_hidden1[i]);
    }

    for (int i = 0; i < hidden2_size; i++) {
        hidden2[i] = 0;
        for (int j = 0; j < hidden1_size; j++) {
            hidden2[i] += hidden1[j] * weights_hidden1_hidden2[j][i];
        }
        hidden2[i] = sigmoid(hidden2[i] + bias_hidden2[i]);
    }

    for (int i = 0; i < hidden3_size; i++) {
        hidden3[i] = 0;
        for (int j = 0; j < hidden2_size; j++) {
            hidden3[i] += hidden2[j] * weights_hidden2_hidden3[j][i];
        }
        hidden3[i] = sigmoid(hidden3[i] + bias_hidden3[i]);
    }

    for (int i = 0; i < output_size; i++) {
        outputs[i] = 0;
        for (int j = 0; j < hidden3_size; j++) {
            outputs[i] += hidden3[j] * weights_hidden3_output[j][i];
        }
        outputs[i] = sigmoid(outputs[i] + bias_output[i]);
    }

    return outputs;
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& training_data,
                          const std::vector<std::vector<float>>& target_data,
                          int epochs, float learning_rate) {
    // Training logic will go here
    // (e.g., backpropagation and weight updates)
}

int main() {
    NeuralNetwork nn;
    std::vector<float> inputs = {1.0f, 0.5f, 0.2f, 0.8f, 0.6f, 0.9f, 0.4f, 0.7f};

    std::vector<float> outputs = nn.forward(inputs);

    std::cout << "Outputs: ";
    for (const auto& output : outputs) {
        std::cout << output << " ";
    }
    std::cout << std::endl;

    return 0;
}
