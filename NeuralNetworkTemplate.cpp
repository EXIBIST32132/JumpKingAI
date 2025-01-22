#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

std::atomic<bool> is_training(true);
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

    void save(const std::string &filename);

    void load(const std::string &filename);

private:
    int input_size = 8;
    int hidden1_size = 128;
    int hidden2_size = 128;
    int hidden3_size = 128;
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
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < training_data.size() && is_training; i++) {
            // Forward pass
            std::vector<float> outputs = forward(training_data[i]);

            // Calculate loss
            std::vector<float> output_errors(output_size);
            for (int j = 0; j < output_size; j++) {
                output_errors[j] = target_data[i][j] - outputs[j];
                total_loss += output_errors[j] * output_errors[j];
            }

            // Backpropagation
            std::vector<float> delta_output(output_size);
            for (int j = 0; j < output_size; j++) {
                delta_output[j] = output_errors[j] * sigmoid_derivative(outputs[j]);
            }

            // Hidden3 -> Output
            std::vector<float> delta_hidden3(hidden3_size, 0.0f);
            for (int j = 0; j < hidden3_size; j++) {
                for (int k = 0; k < output_size; k++) {
                    delta_hidden3[j] += delta_output[k] * weights_hidden3_output[j][k];
                    weights_hidden3_output[j][k] += learning_rate * hidden3[j] * delta_output[k];
                }
                delta_hidden3[j] *= sigmoid_derivative(hidden3[j]);
            }

            // Repeat for other layers...

            // Update biases
            for (int j = 0; j < output_size; j++) {
                bias_output[j] += learning_rate * delta_output[j];
            }
            for (int j = 0; j < hidden3_size; j++) {
                bias_hidden3[j] += learning_rate * delta_hidden3[j];
            }
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / training_data.size() << std::endl;

        if (!is_training) {
            std::cout << "Training stopped by user." << std::endl;
            break;
        }
    }

}

std::ofstream operator<<(const std::ofstream & lhs, char * str);

void NeuralNetwork::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for saving." << std::endl;
        return;
    }

    // Save weights and biases for all layers
    for (const auto& layer : {weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_hidden3, weights_hidden3_output}) {
        for (const auto& row : layer) {
            for (float weight : row) {
                file << weight << " ";
            }
            file << "\n";
        }
    }

    for (const auto& biases : {bias_hidden1, bias_hidden2, bias_hidden3, bias_output}) {
        for (float bias : biases) {
            file << bias << " ";
        }
        file << "\n";
    }

    file.close();
}


void NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for loading." << std::endl;
        return;
    }

    // Load weights and biases for all layers
    for (auto& layer : {&weights_input_hidden1, &weights_hidden1_hidden2, &weights_hidden2_hidden3, &weights_hidden3_output}) {
        for (auto& row : *layer) {
            for (float& weight : row) {
                file >> weight;
            }
        }
    }

    for (auto& biases : {&bias_hidden1, &bias_hidden2, &bias_hidden3, &bias_output}) {
        for (float& bias : *biases) {
            file >> bias;
        }
    }

    file.close();
}



int main() {
    int epochs;
    std::cout << "Enter the number of epochs: ";
    std::cin >> epochs;
    // Thread for listening to stop training
    std::thread input_thread([]() {
    std::string input;
    while (is_training) {
        std::cout << "Type 'q' to stop training: ";
        std::cin >> input;
        if (input == "q" || input == "Q") {
            is_training = false;
            break;
        }
    }
});

    NeuralNetwork nn;

    // Load pre-trained weights
    nn.load("weights_and_biases.txt");

    // Example training data
    std::vector<std::vector<float>> training_data = {
        {1.0f, 0.5f, 0.2f, 0.8f, 0.6f, 0.9f, 0.4f, 0.7f},
        {0.2f, 0.3f, 0.9f, 0.1f, 0.5f, 0.4f, 0.6f, 0.8f}
    };

    std::vector<std::vector<float>> target_data = {
        {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}
    };

    // Train and save
    nn.train(training_data, target_data, epochs, 0.01f);
    nn.save("weights_and_biases.txt");

    // Load and test
    nn.load("weights_and_biases.txt");
    std::vector<float> test_input = {1.0f, 0.5f, 0.2f, 0.8f, 0.6f, 0.9f, 0.4f, 0.7f};
    std::vector<float> outputs = nn.forward(test_input);

    std::cout << "Test output after loading: ";
    for (float output : outputs) {
        std::cout << output << " ";
    }
    std::cout << std::endl;
    input_thread.join();

    return 0;

}
