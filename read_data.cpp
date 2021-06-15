#include "read_data.hpp"
#include <iostream> //cout
#include <fstream> //ifstream
#include <sstream> //istringstream
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

/* 

Input: dataset file
Output: returns dataset, data in two-dimensional vector.

*/

// function to read the medical insurance dataset
// vector<vector<float> > read_data(string inputfile){
  
//   ifstream fin;
//   fin.open(inputfile);

//   vector<vector<float> > dataset;
//   vector<float> temp;

//   float age, sex, bmi, children, smoker, charges;
//   string temp_sex, temp_smoker;

//   string line;

//   int l = 0;

//   if(fin.is_open()){
//     cout << "File open successfully" << endl;
    
//     while(getline(fin,line)){
//       l++;
      
//       replace(line.begin(), line.end(), ',', ' ');
//       istringstream temp_row(line);

//       temp_row >> age >> temp_sex >> bmi >> children >> temp_smoker >> charges;

//       temp.push_back(age);
//       if (temp_sex == "male"){
//         sex = 0; // male
//       }
//       else {
//         sex = 1; // female
//       }
//       temp.push_back(sex);
//       temp.push_back(bmi);
//       temp.push_back(children);
//       if (temp_smoker == "yes"){
//         smoker = 1; //yes 
//       }
//       else {
//         smoker = 0;   //no
//       }
//       temp.push_back(smoker);
//       temp.push_back(charges);

//       dataset.push_back(temp);
//       temp.clear();
//     }
//     cout << l << endl;
//   }
//   else{
//     cout << "Unable to open the specified file " << endl;
//   }

//   return dataset;
// }


//function to read mnist dataset
vector<vector<float> > read_data(string inputfile) {
  vector<vector<float> > dataset;

  ifstream fin;
  fin.open(inputfile);

  int l = 0;
  string line;

  if (fin.is_open()) {
    cout << "File open successfully" << endl;

    while (getline(fin, line)){
      l++;
      istringstream linestream(line);
      vector<float> row;

      while (linestream) {
        string row_value;
        int val = 0;

        if (!getline(linestream, row_value, ','))
          break;
        try {
          row.push_back(stof(row_value));
          val++;
        }
        catch (const std::invalid_argument err) {
          cout << "Invalid value found in the file: " << inputfile << " line: " << l << " value: " << val << endl;
          err.what();
        }
      }

      dataset.push_back(row);
      row.clear();
    }
    cout << l << endl;
  }
  else{
    cout << "Unable to open the specified file " << endl;
  }

  return dataset;
  }