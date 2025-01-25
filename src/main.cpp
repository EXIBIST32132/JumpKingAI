#include <iostream>
#include "neural_network.h"
#include "game_simulation.h"

int main() {
    NeuralNetwork nn(8, 512, 2); // Input size: 256, Hidden size: 128, Output size: 2

    std::cout << "Welcome to Jump King AI Trainer!\n";
    std::cout << "Options:\n";
    std::cout << "1. Train the Neural Network\n";
    std::cout << "2. Test the AI in the game\n";
    std::cout << "Choose an option (1 or 2): ";

    int choice;
    std::cin >> choice;

    if (choice == 1) {
        // Training mode
        std::vector<std::vector<float>> training_data = load_csv("training_data.csv");
        std::vector<std::vector<float>> target_data = load_csv("target_data.csv");

        std::cout << "Starting training...\n";
        nn.train(training_data, target_data, 1000000, 0.01f); // Train for 1000 epochs with 0.01 learning rate

        std::cout << "Saving weights and biases...\n";
        nn.save_weights("weights.txt");
        nn.save_biases("biases.txt");
        std::cout << "Training complete and weights saved!\n";

    } else if (choice == 2) {
        // Testing mode
        std::cout << "Loading weights and biases...\n";
        nn.load_weights("weights.txt");
        nn.load_biases("biases.txt");

        std::cout << "Starting AI test in the game...\n";
        run_game_simulation(nn); // Starts screen capture and AI-based control
    } else {
        std::cout << "Invalid option. Exiting.\n";
    }

    return 0;
}
