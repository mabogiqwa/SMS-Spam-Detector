#ifndef RNN_H
#define RNN_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

class RNN
{
public:
    RNN(int inputSize, int hiddenSize, int outputSize);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<int>>& batchSeq);
    int predict(const std::vector<std::vector<int>>& inputSeq);

    // Add training method
    void train(const std::vector<std::vector<int>>& sequences,
               const std::vector<int>& labels,
               double learningRate = 0.01,
               int epochs = 100);

private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<double>> inputToHiddenWeights; //Wxh
    std::vector<std::vector<double>> hiddenToHiddenWeights; //Whh
    std::vector<std::vector<double>> hiddenToOutputWeights; //Why

    std::vector<double> hiddenBias; //bh
    std::vector<double> outputBias; //by

    std::vector<double> tanh(const std::vector<double>& x);
    std::vector<double> softmax(const std::vector<double>& x);
};

#endif // RNN_H
