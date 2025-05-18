#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cctype>
#include <sstream>

std::vector<std::pair<std::string, std::string>> get_data();
//Postcondition: Retrieves/Parses message data from .csv file

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

void print_data(std::vector<std::pair<std::string, std::string>>);
//Postcondition: Prints message data

int main()
{
    std::vector<std::pair<std::string, std::string>> data = get_data();

    //print_data(data);

    for (const auto& sms : data) {
        std::string message = sms.second;
        std::cout << normalize(message) << std::endl;
    }

    return 0;
}

std::vector<std::pair<std::string, std::string>> get_data()
{
    std::string fileDirectory = "data/synthetic_sms_dataset_with_social.csv";
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

void print_data(std::vector<std::pair<std::string, std::string>> data)
{
    int iteration = 0;

    while (iteration < data.size())
    {
        std::cout << "Label: " << data[iteration].first << ", Message: " << data[iteration].second << std::endl;
        iteration++;
    }
}
