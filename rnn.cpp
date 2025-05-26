#include "rnn.h"
#include <iostream>
#include <algorithm>
#include <numeric>

RNN::RNN(int inputS, int hiddenS, int outputS) : inputSize(inputS), hiddenSize(hiddenS), outputSize(outputS)
{
    // Use proper random seed and Xavier initialization
    std::random_device rd;
    std::default_random_engine gen(rd());

    // Xavier initialization for better gradient flow
    double xavier_input = std::sqrt(2.0 / (inputSize + hiddenSize));
    double xavier_hidden = std::sqrt(2.0 / (hiddenSize + hiddenSize));
    double xavier_output = std::sqrt(2.0 / (hiddenSize + outputSize));

    std::uniform_real_distribution<double> dist_input(-xavier_input, xavier_input);
    std::uniform_real_distribution<double> dist_hidden(-xavier_hidden, xavier_hidden);
    std::uniform_real_distribution<double> dist_output(-xavier_output, xavier_output);

    inputToHiddenWeights.resize(inputSize, std::vector<double>(hiddenSize));
    hiddenToHiddenWeights.resize(hiddenSize, std::vector<double>(hiddenSize));
    hiddenToOutputWeights.resize(hiddenSize, std::vector<double>(outputSize));

    hiddenBias.resize(hiddenSize, 0.0);
    outputBias.resize(outputSize, 0.0);

    // Initialize with Xavier initialization
    for (auto& row : inputToHiddenWeights) {
        for (auto& val : row) { val = dist_input(gen); }
    }
    for (auto& row : hiddenToHiddenWeights) {
        for (auto& val : row) { val = dist_hidden(gen); }
    }
    for (auto& row : hiddenToOutputWeights) {
        for (auto& val : row) { val = dist_output(gen); }
    }
}

std::vector<double> RNN::tanh(const std::vector<double>& x)
{
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        result[i] = std::tanh(x[i]);

    return result;
}

std::vector<double> RNN::softmax(const std::vector<double>& x)
{
    std::vector<double> exps(x.size());

    // Find max for numerical stability
    double max_val = *std::max_element(x.begin(), x.end());

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        exps[i] = std::exp(x[i] - max_val);
        sum += exps[i];
    }

    if (sum > 0) {
        for (double& val : exps) { val /= sum; }
    }

    return exps;
}

std::vector<std::vector<double>> RNN::forward(const std::vector<std::vector<int>>& batchSeq)
{
    std::vector<std::vector<double>> batchOutputs;

    for (const auto& sequence : batchSeq) {
        if (sequence.empty()) {
            std::vector<double> defaultOutput(outputSize, 1.0 / outputSize);
            batchOutputs.push_back(defaultOutput);
            continue;
        }

        std::vector<double> h(hiddenSize, 0.0);

        for (size_t t = 0; t < sequence.size(); ++t) {
            int wordId = sequence[t];

            if (wordId == 0) continue; // Skip padding

            std::vector<double> x(inputSize, 0.0);
            if (wordId > 0 && wordId < inputSize) {
                x[wordId] = 1.0;
            }

            std::vector<double> xh(hiddenSize, 0.0);
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < inputSize; ++j) {
                    xh[i] += x[j] * inputToHiddenWeights[j][i];
                }
                xh[i] += hiddenBias[i];
            }

            std::vector<double> hh(hiddenSize, 0.0);
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; ++j) {
                    hh[i] += h[j] * hiddenToHiddenWeights[j][i];
                }
            }

            for (int i = 0; i < hiddenSize; ++i) {
                h[i] = std::tanh(xh[i] + hh[i]);
            }
        }

        std::vector<double> y(outputSize, 0.0);
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < hiddenSize; j++) {
                y[i] += h[j] * hiddenToOutputWeights[j][i];
            }
            y[i] += outputBias[i];
        }

        std::vector<double> probs = softmax(y);
        batchOutputs.push_back(probs);
    }

    return batchOutputs;
}

int RNN::predict(const std::vector<std::vector<int>>& inputSeq)
{
    std::vector<std::vector<double>> probs = forward(inputSeq);

    if (probs.empty()) return -1;

    std::vector<double> prob = probs[0];
    auto max_it = std::max_element(prob.begin(), prob.end());
    return std::distance(prob.begin(), max_it);
}

// Simplified but more effective training using mini-batches
void RNN::train(const std::vector<std::vector<int>>& sequences,
                const std::vector<int>& labels,
                double learningRate,
                int epochs)
{
    std::cout << "Training RNN for " << epochs << " epochs..." << std::endl;

    // Create shuffled indices for mini-batch training
    std::vector<size_t> indices(sequences.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::default_random_engine gen(rd());

    int batchSize = 32; // Mini-batch size

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data each epoch
        std::shuffle(indices.begin(), indices.end(), gen);

        double totalLoss = 0.0;
        int correct = 0;
        int processed = 0;

        // Process in mini-batches
        for (size_t batchStart = 0; batchStart < indices.size(); batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, indices.size());

            // Accumulate gradients for this batch
            std::vector<std::vector<double>> outputWeightGrads(hiddenSize, std::vector<double>(outputSize, 0.0));
            std::vector<double> outputBiasGrads(outputSize, 0.0);

            double batchLoss = 0.0;
            int batchCorrect = 0;

            for (size_t b = batchStart; b < batchEnd; ++b) {
                size_t i = indices[b];

                // Forward pass
                std::vector<std::vector<int>> singleSeq = {sequences[i]};
                std::vector<std::vector<double>> output = forward(singleSeq);

                if (output.empty()) continue;

                std::vector<double> probs = output[0];
                int trueLabel = labels[i];

                if (trueLabel < 0 || trueLabel >= outputSize) continue;

                // Calculate loss
                double loss = -std::log(std::max(probs[trueLabel], 1e-15));
                batchLoss += loss;

                // Check accuracy
                int predicted = std::distance(probs.begin(),
                                            std::max_element(probs.begin(), probs.end()));
                if (predicted == trueLabel) batchCorrect++;

                // Calculate output gradients (simplified cross-entropy derivative)
                for (int j = 0; j < outputSize; ++j) {
                    double grad = probs[j] - (j == trueLabel ? 1.0 : 0.0);
                    outputBiasGrads[j] += grad;

                    // Get final hidden state (re-run forward pass for this sequence)
                    std::vector<double> h(hiddenSize, 0.0);
                    for (size_t t = 0; t < sequences[i].size(); ++t) {
                        int wordId = sequences[i][t];
                        if (wordId == 0) continue;

                        std::vector<double> x(inputSize, 0.0);
                        if (wordId > 0 && wordId < inputSize) {
                            x[wordId] = 1.0;
                        }

                        std::vector<double> xh(hiddenSize, 0.0);
                        for (int hi = 0; hi < hiddenSize; hi++) {
                            for (int xi = 0; xi < inputSize; ++xi) {
                                xh[hi] += x[xi] * inputToHiddenWeights[xi][hi];
                            }
                            xh[hi] += hiddenBias[hi];
                        }

                        std::vector<double> hh(hiddenSize, 0.0);
                        for (int hi = 0; hi < hiddenSize; hi++) {
                            for (int hj = 0; hj < hiddenSize; ++hj) {
                                hh[hi] += h[hj] * hiddenToHiddenWeights[hj][hi];
                            }
                        }

                        for (int hi = 0; hi < hiddenSize; ++hi) {
                            h[hi] = std::tanh(xh[hi] + hh[hi]);
                        }
                    }

                    // Accumulate weight gradients
                    for (int k = 0; k < hiddenSize; ++k) {
                        outputWeightGrads[k][j] += grad * h[k];
                    }
                }

                processed++;
            }

            // Update weights using accumulated gradients
            if (processed > 0) {
                double batchLearningRate = learningRate / (batchEnd - batchStart);

                // Update output weights and biases
                for (int i = 0; i < hiddenSize; ++i) {
                    for (int j = 0; j < outputSize; ++j) {
                        hiddenToOutputWeights[i][j] -= batchLearningRate * outputWeightGrads[i][j];
                    }
                }

                for (int j = 0; j < outputSize; ++j) {
                    outputBias[j] -= batchLearningRate * outputBiasGrads[j];
                }
            }

            totalLoss += batchLoss;
            correct += batchCorrect;
        }

        if (processed > 0 && (epoch % 5 == 0 || epoch == epochs - 1)) {
            double avgLoss = totalLoss / processed;
            double accuracy = (double)correct / processed;
            std::cout << "Epoch " << epoch << ": Loss = " << avgLoss
                     << ", Accuracy = " << (accuracy * 100) << "%" << std::endl;
        }
    }
}
