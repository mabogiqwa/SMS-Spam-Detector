#include <iostream>
#include <algorithm>
#include <random>
#include "data_utils.h"
#include "rnn.h"

int main()
{
    DataUtils util1;
    std::vector<std::pair<std::string, std::string>> data = util1.get_data();

    // Improved hyperparameters
    int vocabSize, hiddenSize, numClasses;
    int maxSeqLength = 30; // Reduced for better processing

    std::vector<std::vector<std::string>> tokenizedData;
    std::unordered_map<std::string, int> vocab;
    std::vector<std::vector<int>> sequences;

    // Label encoding
    std::unordered_map<std::string, int> labelToInt = {
        {"ham", 0},
        {"spam", 1},
        {"social", 2},
        {"promotional", 3}
    };

    std::unordered_map<int, std::string> intToLabel = {
        {0, "ham"},
        {1, "spam"},
        {2, "social"},
        {3, "promotional"}
    };

    std::vector<int> labels;

    // Process data
    for (const auto& sms : data) {
        std::string normalizedMessage = util1.normalize(sms.second);
        std::vector<std::string> tokens = util1.tokenize(normalizedMessage);
        tokenizedData.push_back(tokens);

        if (labelToInt.find(sms.first) != labelToInt.end()) {
            labels.push_back(labelToInt[sms.first]);
        } else {
            std::cerr << "Unknown label: " << sms.first << std::endl;
            labels.push_back(-1);
        }
    }

    // Build vocabulary with better frequency filtering
    vocab = util1.build_vocab(tokenizedData, 2); // Min frequency of 2
    vocab["<PAD>"] = 0;
    vocab["<UNK>"] = vocab.size();

    // Convert to sequences
    for (const auto& tokens : tokenizedData) {
        sequences.push_back(util1.text_to_sequence(tokens, vocab, maxSeqLength));
    }

    vocabSize = vocab.size() + 10; // Extra buffer for unknown tokens
    hiddenSize = 64; // Reduced to prevent overfitting
    numClasses = 4;

    std::cout << "Dataset Statistics:" << std::endl;
    std::cout << "Total messages: " << data.size() << std::endl;
    std::cout << "Vocabulary size: " << vocabSize << std::endl;
    std::cout << "Hidden size: " << hiddenSize << std::endl;
    std::cout << "Max sequence length: " << maxSeqLength << std::endl;

    // Check class distribution
    std::vector<int> classCount(numClasses, 0);
    for (int label : labels) {
        if (label >= 0 && label < numClasses) {
            classCount[label]++;
        }
    }

    std::cout << "\nClass distribution:" << std::endl;
    for (int i = 0; i < numClasses; ++i) {
        std::cout << intToLabel[i] << ": " << classCount[i] << " samples" << std::endl;
    }

    // Shuffle data before splitting
    std::vector<size_t> indices(sequences.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::default_random_engine gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Create balanced train/test split
    size_t trainSize = static_cast<size_t>(0.8 * sequences.size());

    std::vector<std::vector<int>> trainSeq, testSeq;
    std::vector<int> trainLabels, testLabels;

    for (size_t i = 0; i < trainSize; ++i) {
        trainSeq.push_back(sequences[indices[i]]);
        trainLabels.push_back(labels[indices[i]]);
    }

    for (size_t i = trainSize; i < indices.size(); ++i) {
        testSeq.push_back(sequences[indices[i]]);
        testLabels.push_back(labels[indices[i]]);
    }

    std::cout << "\nTraining set size: " << trainSize << std::endl;
    std::cout << "Test set size: " << (sequences.size() - trainSize) << std::endl;

    // Initialize model
    RNN model(vocabSize, hiddenSize, numClasses);

    // Train with better hyperparameters
    std::cout << "\nStarting training..." << std::endl;
    model.train(trainSeq, trainLabels, 0.01, 20); // Higher learning rate, more epochs

    // Evaluate on test set
    int correctPredictions = 0;
    int totalPredictions = 0;
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));

    std::cout << "\nEvaluating model on test set..." << std::endl;

    for (size_t i = 0; i < testSeq.size() && i < testLabels.size(); ++i) {
        std::vector<std::vector<int>> singleMessage = {testSeq[i]};
        int predictedLabel = model.predict(singleMessage);
        int actualLabel = testLabels[i];

        if (actualLabel == -1 || predictedLabel < 0 || predictedLabel >= numClasses) {
            continue;
        }

        totalPredictions++;

        if (predictedLabel == actualLabel) {
            correctPredictions++;
        }

        confusionMatrix[actualLabel][predictedLabel]++;
    }

    // Calculate and display metrics
    double accuracy = (double)correctPredictions / totalPredictions;
    double errorRate = 1.0 - accuracy;

    std::cout << "\n=== Model Performance Metrics (Test Set) ===" << std::endl;
    std::cout << "Total predictions: " << totalPredictions << std::endl;
    std::cout << "Correct predictions: " << correctPredictions << std::endl;
    std::cout << "Accuracy: " << (accuracy * 100) << "%" << std::endl;
    std::cout << "Error Rate: " << (errorRate * 100) << "%" << std::endl;

    // Display confusion matrix
    std::cout << "\n=== Confusion Matrix ===" << std::endl;
    std::cout << "Rows = Actual, Columns = Predicted" << std::endl;
    std::cout << "\t\t";
    for (int i = 0; i < numClasses; ++i) {
        std::cout << intToLabel[i] << "\t";
    }
    std::cout << std::endl;

    for (int i = 0; i < numClasses; ++i) {
        std::cout << intToLabel[i] << "\t\t";
        for (int j = 0; j < numClasses; ++j) {
            std::cout << confusionMatrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    // Calculate per-class metrics
    std::cout << "\n=== Per-Class Metrics ===" << std::endl;
    double totalF1 = 0.0;
    int validClasses = 0;

    for (int i = 0; i < numClasses; ++i) {
        int truePositives = confusionMatrix[i][i];
        int falseNegatives = 0;
        int falsePositives = 0;
        int actualClassTotal = 0;

        for (int j = 0; j < numClasses; ++j) {
            actualClassTotal += confusionMatrix[i][j];
            if (j != i) {
                falseNegatives += confusionMatrix[i][j];
            }
        }

        for (int j = 0; j < numClasses; ++j) {
            if (j != i) {
                falsePositives += confusionMatrix[j][i];
            }
        }

        double precision = (truePositives + falsePositives > 0) ?
                          (double)truePositives / (truePositives + falsePositives) : 0.0;
        double recall = (actualClassTotal > 0) ?
                       (double)truePositives / actualClassTotal : 0.0;
        double f1Score = (precision + recall > 0) ?
                        2 * (precision * recall) / (precision + recall) : 0.0;

        std::cout << intToLabel[i] << ":" << std::endl;
        std::cout << "  Precision: " << (precision * 100) << "%" << std::endl;
        std::cout << "  Recall: " << (recall * 100) << "%" << std::endl;
        std::cout << "  F1-Score: " << (f1Score * 100) << "%" << std::endl;
        std::cout << "  Support: " << actualClassTotal << " samples" << std::endl;

        if (actualClassTotal > 0) {
            totalF1 += f1Score;
            validClasses++;
        }
    }

    if (validClasses > 0) {
        double macroF1 = totalF1 / validClasses;
        std::cout << "\nMacro-averaged F1-Score: " << (macroF1 * 100) << "%" << std::endl;
    }

    model.save_model_text("model-weights.txt");

    return 0;
}
