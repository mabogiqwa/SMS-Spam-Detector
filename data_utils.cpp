#include "data_utils.h"

std::string DataUtils::normalize(const std::string& input)
{
    std::string result;

    for (char c: input) {
        if (std::isalnum(c) || std::isspace(c))
        {
            result += std::tolower(c);
        }
    }
    return result;
}

double DataUtils::compute_average_length(const std::vector<std::vector<std::string>>& tokenizedData)
{
    size_t length = 0;

    for (const auto& sms: tokenizedData) {
        length += sms.size();
    }

    return double(length) / tokenizedData.size();
}

std::vector<std::string> DataUtils::tokenize(const std::string &text)
{
    std::stringstream ss(text);
    std::string word;
    std::vector<std::string> tokens;

    while (ss >> word) {
        tokens.push_back(word);
    }

    return tokens;
}

std::vector<std::pair<std::string, std::string>> DataUtils::get_data()
{
    std::string fileDirectory = "data/synthetic_sms_dataset.csv";
    //Going later use QFileDialog to retrieve file directories
    std::string line, label, message;
    std::vector<std::pair<std::string, std::string>> data;

    std::ifstream ins(fileDirectory);

    if (ins.fail()) {
        std::cerr << "Error opening file!" << std::endl;
        exit(1);
    }

    std::getline(ins, line);

    while (std::getline(ins, line))
    {
        std::stringstream ss(line);
        std::getline(ss, label,',');
        std::getline(ss, message);

        data.emplace_back(label, message);
    }

    ins.close();

    return data;
}

std::unordered_map<std::string, int> DataUtils::build_vocab(const std::vector<std::vector<std::string>>& tokenized_data, int start_index)
{
    std::unordered_map<std::string, int> vocab;
    int index = start_index;

    for (const auto& tokens : tokenized_data) {
        for (const auto& token : tokens) {
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = index++;
            }
        }
    }

    return vocab;
}

std::vector<int> DataUtils::text_to_sequence(const std::vector<std::string> &tokens, const std::unordered_map<std::string, int> &vocab, int maxLen)
{
    std::vector<int> seq;
    for (const auto& token : tokens) {
        seq.push_back(vocab.count(token) ? vocab.at(token) : 0);
    }

    if (maxLen > 0) {
        if (seq.size() > maxLen) {
            seq.resize(maxLen);
        } else {
            seq.insert(seq.end(), maxLen - seq.size(), 0);
        }
    }

    return seq;
}

void DataUtils::print_data(std::vector<std::pair<std::string, std::string>> data)
{
    int iteration = 0;

    while (iteration < data.size())
    {
        std::cout << "Label: " << data[iteration].first << ", Message: " << data[iteration].second << std::endl;
        iteration++;
    }
}
