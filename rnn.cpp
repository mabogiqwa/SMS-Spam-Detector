#include "rnn.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <iomanip>

RNN::RNN(int inputS, int hiddenS, int outputS) : inputSize(inputS), hiddenSize(hiddenS), outputSize(outputS)
{
    std::random_device rd;
    std::default_random_engine gen(rd());

    // Improved Xavier/He initialization
    double xavier_input = std::sqrt(6.0 / (inputSize + hiddenSize));
    double xavier_hidden = std::sqrt(6.0 / (hiddenSize + hiddenSize));
    double xavier_output = std::sqrt(6.0 / (hiddenSize + outputSize));

    std::uniform_real_distribution<double> dist_input(-xavier_input, xavier_input);
    std::uniform_real_distribution<double> dist_hidden(-xavier_hidden, xavier_hidden);
    std::uniform_real_distribution<double> dist_output(-xavier_output, xavier_output);

    inputToHiddenWeights.resize(inputSize, std::vector<double>(hiddenSize));
    hiddenToHiddenWeights.resize(hiddenSize, std::vector<double>(hiddenSize));
    hiddenToOutputWeights.resize(hiddenSize, std::vector<double>(outputSize));

    hiddenBias.resize(hiddenSize, 0.1); // Small positive bias
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

    // Initialize momentum terms for Adam optimizer
    m_inputToHidden.resize(inputSize, std::vector<double>(hiddenSize, 0.0));
    v_inputToHidden.resize(inputSize, std::vector<double>(hiddenSize, 0.0));
    m_hiddenToHidden.resize(hiddenSize, std::vector<double>(hiddenSize, 0.0));
    v_hiddenToHidden.resize(hiddenSize, std::vector<double>(hiddenSize, 0.0));
    m_hiddenToOutput.resize(hiddenSize, std::vector<double>(outputSize, 0.0));
    v_hiddenToOutput.resize(hiddenSize, std::vector<double>(outputSize, 0.0));
    m_hiddenBias.resize(hiddenSize, 0.0);
    v_hiddenBias.resize(hiddenSize, 0.0);
    m_outputBias.resize(outputSize, 0.0);
    v_outputBias.resize(outputSize, 0.0);
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
    double max_val = *std::max_element(x.begin(), x.end());

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        exps[i] = std::exp(std::min(x[i] - max_val, 700.0)); // Prevent overflow
        sum += exps[i];
    }

    if (sum > 1e-15) {
        for (double& val : exps) { val /= sum; }
    } else {
        // Uniform distribution if sum is too small
        std::fill(exps.begin(), exps.end(), 1.0 / x.size());
    }

    return exps;
}

// Store intermediate states for backpropagation
struct ForwardState {
    std::vector<std::vector<double>> hiddenStates;
    std::vector<std::vector<double>> inputVectors;
    std::vector<double> finalOutput;
};

ForwardState RNN::forwardWithState(const std::vector<int>& sequence)
{
    ForwardState state;
    std::vector<double> h(hiddenSize, 0.0);

    for (size_t t = 0; t < sequence.size(); ++t) {
        int wordId = sequence[t];
        if (wordId == 0) continue; // Skip padding

        // One-hot encoding with bounds checking
        std::vector<double> x(inputSize, 0.0);
        if (wordId > 0 && wordId < inputSize) {
            x[wordId] = 1.0;
        }

        // Store input vector
        state.inputVectors.push_back(x);

        // Compute new hidden state
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

        // Apply activation with gradient clipping
        for (int i = 0; i < hiddenSize; ++i) {
            double val = xh[i] + hh[i];
            h[i] = std::tanh(std::max(-10.0, std::min(10.0, val))); // Clip for stability
        }

        state.hiddenStates.push_back(h);
    }

    // Compute output
    std::vector<double> y(outputSize, 0.0);
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSize; j++) {
            y[i] += h[j] * hiddenToOutputWeights[j][i];
        }
        y[i] += outputBias[i];
    }

    state.finalOutput = softmax(y);
    return state;
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

        ForwardState state = forwardWithState(sequence);
        batchOutputs.push_back(state.finalOutput);
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

void RNN::save_model_text(std::string filepath) {
    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
        return;
    }

    out << std::fixed << std::setprecision(10);

    // Write header
    out << "# RNN Model Text Format v1.0\n";
    out << "# Architecture: " << inputSize << " " << hiddenSize << " " << outputSize << "\n";

    // Helper function to write matrix in text format
    auto write_matrix_text = [&](const auto& mat, const std::string& name) {
        out << "\n[" << name << "]\n";
        out << mat.size() << " " << mat[0].size() << "\n";
        for (const auto& row : mat) {
            for (size_t i = 0; i < row.size(); ++i) {
                out << row[i];
                if (i < row.size() - 1) out << " ";
            }
            out << "\n";
        }
    };

    auto write_vector_text = [&](const auto& vec, const std::string& name) {
        out << "\n[" << name << "]\n";
        out << vec.size() << "\n";
        for (size_t i = 0; i < vec.size(); ++i) {
            out << vec[i];
            if (i < vec.size() - 1) out << " ";
        }
        out << "\n";
    };

    write_matrix_text(inputToHiddenWeights, "inputToHiddenWeights");
    write_matrix_text(hiddenToHiddenWeights, "hiddenToHiddenWeights");
    write_matrix_text(hiddenToOutputWeights, "hiddenToOutputWeights");
    write_vector_text(hiddenBias, "hiddenBias");
    write_vector_text(outputBias, "outputBias");

    out.close();
    std::cout << "Model saved in text format to " << filepath << std::endl;
}

void RNN::train(const std::vector<std::vector<int>>& sequences,
                const std::vector<int>& labels,
                double learningRate,
                int epochs)
{
    std::cout << "Training RNN for " << epochs << " epochs..." << std::endl;

    // Create shuffled indices
    std::vector<size_t> indices(sequences.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::default_random_engine gen(rd());

    int batchSize = 16; // Smaller batch size for better gradients

    // Adam optimizer parameters
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0; // time step for Adam

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);

        double totalLoss = 0.0;
        int correct = 0;
        int processed = 0;

        for (size_t batchStart = 0; batchStart < indices.size(); batchStart += batchSize) {
            size_t batchEnd = std::min(batchStart + batchSize, indices.size());
            t++; // increment time step

            // Initialize gradient accumulators
            std::vector<std::vector<double>> grad_inputToHidden(inputSize, std::vector<double>(hiddenSize, 0.0));
            std::vector<std::vector<double>> grad_hiddenToHidden(hiddenSize, std::vector<double>(hiddenSize, 0.0));
            std::vector<std::vector<double>> grad_hiddenToOutput(hiddenSize, std::vector<double>(outputSize, 0.0));
            std::vector<double> grad_hiddenBias(hiddenSize, 0.0);
            std::vector<double> grad_outputBias(outputSize, 0.0);

            double batchLoss = 0.0;
            int batchCorrect = 0;

            for (size_t b = batchStart; b < batchEnd; ++b) {
                size_t i = indices[b];

                if (labels[i] < 0 || labels[i] >= outputSize) continue;

                ForwardState state = forwardWithState(sequences[i]);

                if (state.finalOutput.empty() || state.hiddenStates.empty()) continue;

                int trueLabel = labels[i];

                // Calculate loss with label smoothing
                double smoothing = 0.1;
                double targetProb = 1.0 - smoothing + smoothing / outputSize;
                double nonTargetProb = smoothing / outputSize;

                double loss = 0.0;
                for (int j = 0; j < outputSize; ++j) {
                    double target = (j == trueLabel) ? targetProb : nonTargetProb;
                    loss -= target * std::log(std::max(state.finalOutput[j], 1e-15));
                }
                batchLoss += loss;

                // Check accuracy
                int predicted = std::distance(state.finalOutput.begin(),
                                            std::max_element(state.finalOutput.begin(), state.finalOutput.end()));
                if (predicted == trueLabel) batchCorrect++;

                // Backward pass - output layer gradients
                std::vector<double> outputGrad(outputSize);
                for (int j = 0; j < outputSize; ++j) {
                    double target = (j == trueLabel) ? targetProb : nonTargetProb;
                    outputGrad[j] = state.finalOutput[j] - target;
                    grad_outputBias[j] += outputGrad[j];
                }

                // Final hidden state
                if (!state.hiddenStates.empty()) {
                    std::vector<double> finalHidden = state.hiddenStates.back();

                    // Gradients for output weights
                    for (int j = 0; j < outputSize; ++j) {
                        for (int k = 0; k < hiddenSize; ++k) {
                            grad_hiddenToOutput[k][j] += outputGrad[j] * finalHidden[k];
                        }
                    }

                    // Hidden layer gradients (simplified - only final timestep)
                    std::vector<double> hiddenGrad(hiddenSize, 0.0);
                    for (int k = 0; k < hiddenSize; ++k) {
                        for (int j = 0; j < outputSize; ++j) {
                            hiddenGrad[k] += outputGrad[j] * hiddenToOutputWeights[k][j];
                        }
                        // Derivative of tanh
                        hiddenGrad[k] *= (1.0 - finalHidden[k] * finalHidden[k]);
                        grad_hiddenBias[k] += hiddenGrad[k];
                    }

                    // Input-to-hidden gradients (last timestep)
                    if (!state.inputVectors.empty()) {
                        std::vector<double> lastInput = state.inputVectors.back();
                        for (int j = 0; j < inputSize; ++j) {
                            for (int k = 0; k < hiddenSize; ++k) {
                                grad_inputToHidden[j][k] += hiddenGrad[k] * lastInput[j];
                            }
                        }
                    }

                    // Hidden-to-hidden gradients (simplified)
                    if (state.hiddenStates.size() > 1) {
                        std::vector<double> prevHidden = state.hiddenStates[state.hiddenStates.size() - 2];
                        for (int j = 0; j < hiddenSize; ++j) {
                            for (int k = 0; k < hiddenSize; ++k) {
                                grad_hiddenToHidden[j][k] += hiddenGrad[k] * prevHidden[j];
                            }
                        }
                    }
                }

                processed++;
            }

            // Adam optimizer update
            if (processed > 0) {
                double batchLearningRate = learningRate / (batchEnd - batchStart);
                double bias_correction1 = 1.0 - std::pow(beta1, t);
                double bias_correction2 = 1.0 - std::pow(beta2, t);

                // Update input-to-hidden weights
                for (int i = 0; i < inputSize; ++i) {
                    for (int j = 0; j < hiddenSize; ++j) {
                        double grad = grad_inputToHidden[i][j];
                        m_inputToHidden[i][j] = beta1 * m_inputToHidden[i][j] + (1 - beta1) * grad;
                        v_inputToHidden[i][j] = beta2 * v_inputToHidden[i][j] + (1 - beta2) * grad * grad;

                        double m_hat = m_inputToHidden[i][j] / bias_correction1;
                        double v_hat = v_inputToHidden[i][j] / bias_correction2;

                        inputToHiddenWeights[i][j] -= batchLearningRate * m_hat / (std::sqrt(v_hat) + epsilon);
                    }
                }

                // Update hidden-to-hidden weights
                for (int i = 0; i < hiddenSize; ++i) {
                    for (int j = 0; j < hiddenSize; ++j) {
                        double grad = grad_hiddenToHidden[i][j];
                        m_hiddenToHidden[i][j] = beta1 * m_hiddenToHidden[i][j] + (1 - beta1) * grad;
                        v_hiddenToHidden[i][j] = beta2 * v_hiddenToHidden[i][j] + (1 - beta2) * grad * grad;

                        double m_hat = m_hiddenToHidden[i][j] / bias_correction1;
                        double v_hat = v_hiddenToHidden[i][j] / bias_correction2;

                        hiddenToHiddenWeights[i][j] -= batchLearningRate * m_hat / (std::sqrt(v_hat) + epsilon);
                    }
                }

                // Update hidden-to-output weights
                for (int i = 0; i < hiddenSize; ++i) {
                    for (int j = 0; j < outputSize; ++j) {
                        double grad = grad_hiddenToOutput[i][j];
                        m_hiddenToOutput[i][j] = beta1 * m_hiddenToOutput[i][j] + (1 - beta1) * grad;
                        v_hiddenToOutput[i][j] = beta2 * v_hiddenToOutput[i][j] + (1 - beta2) * grad * grad;

                        double m_hat = m_hiddenToOutput[i][j] / bias_correction1;
                        double v_hat = v_hiddenToOutput[i][j] / bias_correction2;

                        hiddenToOutputWeights[i][j] -= batchLearningRate * m_hat / (std::sqrt(v_hat) + epsilon);
                    }
                }

                // Update biases
                for (int i = 0; i < hiddenSize; ++i) {
                    double grad = grad_hiddenBias[i];
                    m_hiddenBias[i] = beta1 * m_hiddenBias[i] + (1 - beta1) * grad;
                    v_hiddenBias[i] = beta2 * v_hiddenBias[i] + (1 - beta2) * grad * grad;

                    double m_hat = m_hiddenBias[i] / bias_correction1;
                    double v_hat = v_hiddenBias[i] / bias_correction2;

                    hiddenBias[i] -= batchLearningRate * m_hat / (std::sqrt(v_hat) + epsilon);
                }

                for (int i = 0; i < outputSize; ++i) {
                    double grad = grad_outputBias[i];
                    m_outputBias[i] = beta1 * m_outputBias[i] + (1 - beta1) * grad;
                    v_outputBias[i] = beta2 * v_outputBias[i] + (1 - beta2) * grad * grad;

                    double m_hat = m_outputBias[i] / bias_correction1;
                    double v_hat = v_outputBias[i] / bias_correction2;

                    outputBias[i] -= batchLearningRate * m_hat / (std::sqrt(v_hat) + epsilon);
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

        // Decay learning rate
        if (epoch > 0 && epoch % 20 == 0) {
            learningRate *= 0.8;
        }
    }
}
