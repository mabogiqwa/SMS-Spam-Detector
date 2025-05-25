#include <iostream>
#include "data_utils.h"

int main()
{
    DataUtils util1;
    std::vector<std::pair<std::string, std::string>> data = util1.get_data();

    std::vector<std::vector<std::string>> tokenizedData;
    std::unordered_map<std::string, int> vocab;
    std::vector<std::vector<int>> sequences;

    for (const auto& sms : data) {
        std::string normalizedMessage = util1.normalize(sms.second);
        std::vector<std::string> tokens = util1.tokenize(normalizedMessage);
        tokenizedData.push_back(tokens);
    }

    vocab = util1.build_vocab(tokenizedData);

    for (const auto& tokens : tokenizedData) { //Assigns unique id to the tokenized messages
        sequences.push_back(util1.text_to_sequence(tokens, vocab));
    }

    std::cout << "Average message length: " << util1.compute_average_length(tokenizedData);

    return 0;
}
