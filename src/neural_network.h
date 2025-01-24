#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    std::vector<float> forward(const std::vector<float>& inputs);
    void train(const std::vector<std::vector<float>>& training_data,
               const std::vector<std::vector<float>>& target_data,
               int epochs, float learning_rate);
    void save_weights(const std::string& filename);
    void save_biases(const std::string& filename);
    void load_weights(const std::string& filename);
    void load_biases(const std::string& filename);

private:
    int input_size, hidden_size, output_size;

    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;

    std::vector<float> bias_hidden;
    std::vector<float> bias_output;

    void initialize_weights();
    float sigmoid(float x);
    float sigmoid_derivative(float x);
};

std::vector<std::vector<float>> load_csv(const std::string& filename);

#endif // NEURAL_NETWORK_H
