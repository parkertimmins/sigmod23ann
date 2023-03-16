//
// Created by parker on 07.03.23.
//

#ifndef SIGMOD23ANN_GRADE_CPP
#define SIGMOD23ANN_GRADE_CPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <unordered_set>
#include <cassert>

using namespace std;

void read(uint32_t n, const std::string &file_path, std::vector<std::vector<uint32_t>>& data) {
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);

    assert(ifs.is_open());
    data.resize(n);

    const int rowLen = 100;
    std::vector<uint32_t> buff(rowLen);
    int counter = 0;
    while (ifs.read((char *)buff.data(), rowLen * sizeof(uint32_t))) {
        std::vector<uint32_t> row(rowLen);
        for (int d = 0; d < rowLen; d++) {
            row[d] = static_cast<uint32_t>(buff[d]);
        }
        data[counter++] = std::move(row);
    }

    ifs.close();
}


int main(int argc, char **argv) {
    // Also accept other path for source data
    if (argc != 4) {
        throw "requires size and two files to compare";
    }

    auto baseline = std::string(argv[1]);
    auto test = std::string(argv[2]);
    auto n = stoi(argv[3]);

    std::cout << "loaded baseline file: " << baseline << std::endl;
    std::cout << "loaded test file: " << test << std::endl;
    std::cout << "num points: " << n << std::endl;

    vector<vector<uint32_t>> baselineData;
    vector<vector<uint32_t>> testData;
    read(n, baseline, baselineData);
    read(n, test, testData);

    vector<uint32_t> corrects;
    for (int i = 0; i < n; ++i) {
        auto b = baselineData[i];
        auto t = testData[i];




        std::unordered_set<uint32_t> tSet(t.begin(), t.end());

        int numCorrect = 0;
        for (auto& bItem : b) {
//            std::cout << bItem << "," ;
            if (tSet.find(bItem) != tSet.end()) {
                numCorrect++;
            }
        }
//        std::cout << std::endl;
        corrects.push_back(numCorrect);
    }

    uint64_t totalCorrect = 0;
    for (uint32_t c : corrects) {
        totalCorrect  += c;
    }

    double avg = static_cast<double>(totalCorrect) / corrects.size();
    std::cout << "avg percent: " << avg << std::endl;

    return 0;
}


#endif //SIGMOD23ANN_GRADE_CPP
