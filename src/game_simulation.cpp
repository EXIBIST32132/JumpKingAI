#include "game_simulation.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <Windows.h>
#include <iostream>

// Helper functions for screen capture and key pressing
cv::Mat capture_screen(int x, int y, int width, int height) {
    // Capture a specific area of the screen
    HDC hScreen = GetDC(NULL);
    HDC hDC = CreateCompatibleDC(hScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreen, width, height);
    SelectObject(hDC, hBitmap);

    BitBlt(hDC, 0, 0, width, height, hScreen, x, y, SRCCOPY);

    BITMAPINFOHEADER bi = { sizeof(BITMAPINFOHEADER), width, -height, 1, 32, BI_RGB };
    cv::Mat mat(height, width, CV_8UC4); // BGRA
    GetDIBits(hDC, hBitmap, 0, height, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    DeleteObject(hBitmap);
    DeleteDC(hDC);
    ReleaseDC(NULL, hScreen);

    return mat;
}

void press_key(WORD virtual_key, int delay_ms = 50) {
    // Simulate a key press
    keybd_event(virtual_key, 0, 0, 0);             // Key down
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    keybd_event(virtual_key, 0, KEYEVENTF_KEYUP, 0); // Key up
}

std::vector<float> process_screen_data(const cv::Mat& screen) {
    // Resize the screen to match the input size of the neural network
    cv::Mat resized_screen;
    cv::resize(screen, resized_screen, cv::Size(16, 16)); // Example: 16x16

    // Flatten the resized screen into a vector
    std::vector<float> inputs;
    resized_screen.convertTo(resized_screen, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]
    inputs.assign((float*)resized_screen.datastart, (float*)resized_screen.dataend);

    return inputs;
}

void run_game_simulation(NeuralNetwork& nn) {
    // Define the screen area to capture (adjust these values based on your setup)
    int screen_x = 100;  // Starting x-coordinate of the capture area
    int screen_y = 100;  // Starting y-coordinate of the capture area
    int screen_width = 800; // Width of the capture area
    int screen_height = 600; // Height of the capture area

    while (true) {
        // Capture the screen
        cv::Mat screen = capture_screen(screen_x, screen_y, screen_width, screen_height);
        if (screen.empty()) {
            std::cerr << "Failed to capture the screen.\n";
            continue;
        }

        // Preprocess the screen data
        std::vector<float> inputs = process_screen_data(screen);

        // Get AI's decision
        std::vector<float> outputs = nn.forward(inputs);

        // Interpret outputs (e.g., jump or move)
        float jump_probability = outputs[0];
        float move_probability = outputs[1];

        std::cout << "Jump Probability: " << jump_probability << ", Move Probability: " << move_probability << "\n";

        if (jump_probability > 0.5) {
            std::cout << "Jumping...\n";
            press_key(VK_SPACE); // Simulate the spacebar for jumping
        }

        if (move_probability > 0.5) {
            std::cout << "Moving...\n";
            press_key(VK_RIGHT); // Simulate the right arrow key for moving
        }

        // Delay before the next loop iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
