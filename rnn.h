#ifndef RNN_H
#define RNN_H

#include <vector>
#include <random>

// Forward declaration for internal state
struct ForwardState;

class RNN
{
private:
    int inputSize, hiddenSize, outputSize;

    // Network weights
    std::vector<std::vector<double>> inputToHiddenWeights;
    std::vector<std::vector<double>> hiddenToHiddenWeights;
    std::vector<std::vector<double>> hiddenToOutputWeights;

    // Biases
    std::vector<double> hiddenBias;
    std::vector<double> outputBias;

    std::vector<std::vector<double>> m_inputToHidden, v_inputToHidden;
    std::vector<std::vector<double>> m_hiddenToHidden, v_hiddenToHidden;
    std::vector<std::vector<double>> m_hiddenToOutput, v_hiddenToOutput;
    std::vector<double> m_hiddenBias, v_hiddenBias;
    std::vector<double> m_outputBias, v_outputBias;

    std::vector<double> tanh(const std::vector<double>& x);
    std::vector<double> softmax(const std::vector<double>& x);
    ForwardState forwardWithState(const std::vector<int>& sequence);

public:
    RNN(int inputS, int hiddenS, int outputS);

    std::vector<std::vector<double>> forward(const std::vector<std::vector<int>>& batchSeq);
    int predict(const std::vector<std::vector<int>>& inputSeq);
    void train(const std::vector<std::vector<int>>& sequences,
               const std::vector<int>& labels,
               double learningRate,
               int epochs);
    void save_model_text(std::string filepath);
};

#endif
