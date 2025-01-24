#include "neural_network.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
    initialize_weights();
}

void NeuralNetwork::initialize_weights() {
    weights_input_hidden.resize(input_size, std::vector<float>(hidden_size));
    weights_hidden_output.resize(hidden_size, std::vector<float>(output_size));

    bias_hidden.resize(hidden_size);
    bias_output.resize(output_size);

    srand(static_cast<unsigned>(time(0)));
    for (auto& row : weights_input_hidden)
        for (auto& weight : row) weight = static_cast<float>(rand()) / RAND_MAX;

    for (auto& row : weights_hidden_output)
        for (auto& weight : row) weight = static_cast<float>(rand()) / RAND_MAX;

    for (auto& bias : bias_hidden) bias = static_cast<float>(rand()) / RAND_MAX;
    for (auto& bias : bias_output) bias = static_cast<float>(rand()) / RAND_MAX;
}

float NeuralNetwork::sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float NeuralNetwork::sigmoid_derivative(float x) {
    return x * (1 - x);
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& training_data,
                          const std::vector<std::vector<float>>& target_data,
                          int epochs, float learning_rate) {
    // Implement backpropagation here (similar to the initial implementation)
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& inputs) {
    // Forward pass implementation
    return {}; // Return predictions
}

void NeuralNetwork::save_weights(const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& row : weights_input_hidden)
        for (float weight : row) file << weight << "\n";
    for (const auto& row : weights_hidden_output)
        for (float weight : row) file << weight << "\n";
}

void NeuralNetwork::save_biases(const std::string& filename) {
    std::ofstream file(filename);
    for (float bias : bias_hidden) file << bias << "\n";
    for (float bias : bias_output) file << bias << "\n";
}

void NeuralNetwork::load_weights(const std::string& filename) {
    std::ifstream file(filename);
    for (auto& row : weights_input_hidden)
        for (float& weight : row) file >> weight;
    for (auto& row : weights_hidden_output)
        for (float& weight : row) file >> weight;
}

void NeuralNetwork::load_biases(const std::string& filename) {
    std::ifstream file(filename);
    for (float& bias : bias_hidden) file >> bias;
    for (float& bias : bias_output) file >> bias;
}

std::vector<std::vector<float>> load_csv(const std::string& filename) {
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
