#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <set>
using namespace std;

class DataUtils
{
public:
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

    double compute_average_length(const std::vector<std::vector<std::string>>& tokenizedData);
};

#endif // DATA_UTILS_H
