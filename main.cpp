#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>

std::atomic<bool> is_training(true);

class NeuralNetwork {
public:
    void train(const std::vector<std::vector<float>> &inputs,
               const std::vector<std::vector<float>> &outputs, int epochs, float lr) {
        std::cout << "Training started...\n";
        // Simulate training
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "Training completed.\n";
    }

    std::vector<float> forward(const std::vector<float> &input) {
        return {input[0] + 0.1f, input[1] - 0.1f}; // Dummy prediction
    }
};

// Activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

float random_weight() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

class GeoNN {
public:
    GeoNN();
    std::vector<float> forward(const std::vector<float>& inputs);
    void train(const std::vector<std::vector<float>>& training_data,
               const std::vector<std::vector<float>>& target_data,
               int epochs, float learning_rate);

    void save(const std::string& filename);
    void load(const std::string& filename);

private:
    int input_size = 256; // Adjust for feature size (e.g., from CNN)
    int hidden_size = 128;
    int output_size = 2; // Latitude and Longitude

    std::vector<float> hidden_layer;
    std::vector<float> output_layer;

    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;

    std::vector<float> bias_hidden;
    std::vector<float> bias_output;

    void initialize_weights();
};

GeoNN::GeoNN() {
    srand(static_cast<unsigned>(time(0)));

    hidden_layer.resize(hidden_size);
    output_layer.resize(output_size);

    initialize_weights();
}

void GeoNN::initialize_weights() {
    weights_input_hidden.resize(input_size, std::vector<float>(hidden_size));
    weights_hidden_output.resize(hidden_size, std::vector<float>(output_size));

    bias_hidden.resize(hidden_size);
    bias_output.resize(output_size);

    for (auto& row : weights_input_hidden) {
        for (auto& weight : row) {
            weight = random_weight();
        }
    }

    for (auto& row : weights_hidden_output) {
        for (auto& weight : row) {
            weight = random_weight();
        }
    }

    for (auto& bias : bias_hidden) {
        bias = random_weight();
    }

    for (auto& bias : bias_output) {
        bias = random_weight();
    }
}

std::vector<float> GeoNN::forward(const std::vector<float>& inputs) {
    for (int i = 0; i < hidden_size; i++) {
        hidden_layer[i] = 0;
        for (int j = 0; j < input_size; j++) {
            hidden_layer[i] += inputs[j] * weights_input_hidden[j][i];
        }
        hidden_layer[i] = sigmoid(hidden_layer[i] + bias_hidden[i]);
    }

    for (int i = 0; i < output_size; i++) {
        output_layer[i] = 0;
        for (int j = 0; j < hidden_size; j++) {
            output_layer[i] += hidden_layer[j] * weights_hidden_output[j][i];
        }
        output_layer[i] = output_layer[i] + bias_output[i]; // Linear output
    }

    return output_layer;
}

void GeoNN::train(const std::vector<std::vector<float>>& training_data,
                  const std::vector<std::vector<float>>& target_data,
                  int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < training_data.size() && is_training; i++) {
            std::vector<float> outputs = forward(training_data[i]);

            // Calculate error
            std::vector<float> output_errors(output_size);
            for (int j = 0; j < output_size; j++) {
                output_errors[j] = target_data[i][j] - outputs[j];
                total_loss += output_errors[j] * output_errors[j];
            }

            // Backpropagation
            std::vector<float> delta_output(output_size);
            for (int j = 0; j < output_size; j++) {
                delta_output[j] = output_errors[j]; // No activation on output layer
            }

            // Update weights
            for (int j = 0; j < hidden_size; j++) {
                for (int k = 0; k < output_size; k++) {
                    weights_hidden_output[j][k] += learning_rate * hidden_layer[j] * delta_output[k];
                }
            }

            std::vector<float> delta_hidden(hidden_size, 0.0f);
            for (int j = 0; j < hidden_size; j++) {
                for (int k = 0; k < output_size; k++) {
                    delta_hidden[j] += delta_output[k] * weights_hidden_output[j][k];
                }
                delta_hidden[j] *= sigmoid_derivative(hidden_layer[j]);
            }

            for (int j = 0; j < input_size; j++) {
                for (int k = 0; k < hidden_size; k++) {
                    weights_input_hidden[j][k] += learning_rate * training_data[i][j] * delta_hidden[k];
                }
            }
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / training_data.size() << std::endl;
    }
}

std::vector<std::vector<float>> load_csv(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float value;

        while (iss >> value) {
            row.push_back(value);
            if (iss.peek() == ',') iss.ignore();
        }

        data.push_back(row);
    }

    return data;
}

// Run the Python script
void run_python_script(const std::string &script) {
    std::string command = "python3 " + script;
    if (system(command.c_str()) != 0) {
        std::cerr << "Error running Python script.\n";
    }
}

int main() {
    NeuralNetwork nn;

    while (true) {
        std::cout << "Running data generation script...\n";
        run_python_script("/Users/jonathanst-georges/CLionProjects/NeuralNetwork/data_generator.py");

        std::cout << "Loading training data...\n";
        std::vector<std::vector<float>> features = load_csv("features.csv");
        std::vector<std::vector<float>> labels = load_csv("labels.csv");

        std::cout << "Training the model...\n";
        nn.train(features, labels, 1000, 0.01f);

        std::cout << "Testing the model...\n";
        std::vector<float> test_input = features[0]; // Use the first feature as a test
        std::vector<float> prediction = nn.forward(test_input);

        std::cout << "Prediction: " << prediction[0] << ", " << prediction[1] << "\n";

        std::cout << "Continue training? (y/n): ";
        char choice;
        std::cin >> choice;

        if (choice == 'n' || choice == 'N') {
            std::cout << "Exiting...\n";
            break;
        }
    }

    return 0;
}
