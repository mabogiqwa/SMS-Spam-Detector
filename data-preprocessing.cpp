#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <set>

std::vector<std::pair<std::string, std::string>> get_data();
//Postcondition: Retrieves/Parses message data from .csv file

std::string normalize(const std::string&);
//Postcondition: Converts uppercase to lowercase and removes special characters e.g [.,?,',{}]

std::vector<std::string> tokenize(const std::string&);
//Postcondition: Adds each word/token in a sms as a vector<std::string> index

std::unordered_map<std::string, int> build_vocab(const std::vector<std::vector<std::string>>& tokenized_data, int start_index = 1);
//Postcondition: Assigns each token a unique int id

std::vector<int> text_to_sequence(const std::vector<std::string> &tokens, const std::unordered_map<std::string, int> &vocab, int maxLen = 0);
//Postcondition: Converts  tokens to sequences of int IDs and truncates to fixed length

void print_data(std::vector<std::pair<std::string, std::string>>);
//Postcondition: Prints message data

double compute_average_length(const std::vector<std::vector<std::string>>& tokenizedData)
{
    size_t length = 0;

    for (const auto& sms: tokenizedData) {
        length += sms.size();
    }

    return double(length) / tokenizedData.size();
}

void write_to_csv(const std::vector<std::pair<std::string, std::string>>& data, std::vector<std::vector<int>>& sequences)
{
    std::ofstream outFile("data/preprocessed-data.txt");

    if (outFile.fail()) {
        std::cerr << "Error occurred opening file" << std::endl;
        exit(1);
    }

    outFile << "Label,Sequence\n";

    int count = 0;
    for (const auto& sms : data) {
        outFile << sms.first << ",";

        for (size_t j = 0; j < sequences[count].size(); ++j) {
            outFile << sequences[count][j];
            if (j < sequences[count].size() - 1) { outFile << " ";}
        }
        outFile << "\n";
        count++;
    }

    outFile.close();
}

int main()
{
    std::vector<std::pair<std::string, std::string>> data = get_data();

    std::vector<std::vector<std::string>> tokenizedData;
    std::unordered_map<std::string, int> vocab;
    std::vector<std::vector<int>> sequences;

    for (const auto& sms : data) {
        std::string normalizedMessage = normalize(sms.second);
        std::vector<std::string> tokens = tokenize(normalizedMessage);
        tokenizedData.push_back(tokens);
    }

    vocab = build_vocab(tokenizedData);

    for (const auto& tokens : tokenizedData) { //Assigns unique id to the tokenized messages
        sequences.push_back(text_to_sequence(tokens, vocab));
    }

    /*
    int count = 0;
    for (const auto& sms : data) {
        std::string label = sms.first;
        std::cout << label << std::endl;

        if (++count >= 50) { break; }
    }
    */
    //write_to_csv(data, sequences);

    std::cout << "Average message length: " << compute_average_length(tokenizedData);

    return 0;
}

void write_to_csv(const std::vector<std::pair<std::string, std::string>> data, std::vector<int> sequences)
{

}

std::string normalize(const std::string& input)
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

std::vector<std::string> tokenize(const std::string &text)
{
    std::stringstream ss(text);
    std::string word;
    std::vector<std::string> tokens;

    while (ss >> word) {
        tokens.push_back(word);
    }

    return tokens;
}

std::vector<std::pair<std::string, std::string>> get_data()
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

std::unordered_map<std::string, int> build_vocab(const std::vector<std::vector<std::string>>& tokenized_data, int start_index)
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

std::vector<int> text_to_sequence(const std::vector<std::string> &tokens, const std::unordered_map<std::string, int> &vocab, int maxLen)
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

void print_data(std::vector<std::pair<std::string, std::string>> data)
{
    int iteration = 0;

    while (iteration < data.size())
    {
        std::cout << "Label: " << data[iteration].first << ", Message: " << data[iteration].second << std::endl;
        iteration++;
    }
}
