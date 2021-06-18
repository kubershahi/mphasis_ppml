#include <vector>   // for vector operations
#include <string>   // for string operations
#include <iostream> // input output operation: cout
#include <fstream>  // file stream operation: ifstream
#include <sstream>  // string stream operation: istringstream
#include <algorithm>    // replace functionality
#include "read_data.hpp"
#include <Eigen/Dense>

/* 

Input: dataset file
Output: returns dataset, data in two-dimensional vector.

*/

using namespace std;

// // function to read the medical insurance dataset
// void read_data(string inputfile, vector<vector<float> > &X, vector<float> &Y){
  
//   ifstream fin;                    // declaring the input filestream
//   fin.open(inputfile);             // opening the file

//   vector<float> temp;              // declaring a temp vector to hold content of a row

//   float age, sex, bmi, children, smoker, charges;  // declaring the six features of the dataset
//   string temp_sex, temp_smoker;    // two features are in strings, needs to be converted

//   string line;                     // declaring a string to hold the content of a line in the dataset file

//   int l = 0;                       // declaring a integer to count the number of line

//   if(fin.is_open()){               // if the dataset file is open  
//     cout << "File open successfully" << endl;
    
//     while(getline(fin,line)){      // getting the content of a line in variable line
//       l++;                         // increasing the line count
      
//       replace(line.begin(), line.end(), ',', ' ');   // replacing the commas with whitespace in a line
//       istringstream temp_row(line);    // creating a string stream from the string line 

//       temp_row >> age >> temp_sex >> bmi >> children >> temp_smoker >> charges; 
//                     // strong the values in the string stream line in the appropriate variable

//       temp.push_back(age);           // pushing the age value in the temp_row
//       if (temp_sex == "male"){       // if the sex is male
//         sex = 0.0; // male             // setting sex to 0 
//       }
//       else {
//         sex = 1.0; // female           // else 1
//       }
//       temp.push_back(sex);           // pushing sex
//       temp.push_back(bmi);           // pushing bmi 
//       temp.push_back(children);      // pushing number of children
//       if (temp_smoker == "yes"){     // if a smoker
//         smoker = 1.0; //yes            // setting smoke to 1
//       }
//       else {
//         smoker = 0.0;   //no           // else to a 1
//       }
//       temp.push_back(smoker);        // pushing smoker
      
//       Y.push_back(charges);       // pushing the final insurance charges as y_values
//       X.push_back(temp);          // pushing the row into the dataset as x_values
//       temp.clear();               // clearing the temp to store the next line
//     }
//     cout << "Lines read: " << l << endl;               // display the number of read lines
//   }
//   else{
//     cout << "Unable to open the specified file " << endl;  // output if file can't be opened
//   }
// }


//function to read mnist dataset
void read_data(string inputfile, vector< vector<float> > &X, vector< float> &Y) {

  ifstream fin;                     // declaring the input file stream
  fin.open(inputfile);              // opening the inputfile

  int l = 0;                        // declaring a integer to track the number of line
  string line;                      // declaring a string to hold the read line of the input file

  if (fin.is_open()) {              // if the input file is open
    cout << "File opened successfully " << endl; 

    while (getline(fin, line)){     // storing the line of input file on the variable line
      l++;                          // increasing the line read counter
      istringstream linestream(line); // converting the read line into an string stream
      vector<float> row;            // declaring a vector to store the current row

      int val = 0;                 // declaring a variable to track the number of values in a row
      while (linestream) {         // while the string stream is not null
        string row_value;          // declaring a string to hold the row values

        if (!getline(linestream, row_value, ',')) // storing the values from stream into row_value one by one
          break;                                  // at the end of row break the while loop
        try { 
          if (val < 784) {                                
            row.push_back(stof(row_value));         // pushing the current value into the row for X values
            val++;
          }
          else if (val == 784)                      // pushing the current value into the Y for y values
          {
            Y.push_back(stof(row_value));
          }
        }
        catch (const invalid_argument err) {      // if there is a error catch the error and display it
          cout << "Invalid value found in the file: " << inputfile << " line: " << l << " value: " << val << endl;
          err.what();
        }
      }

      X.push_back(row);                     // pushing the row into the dataset
      row.clear();                                // clearing the row vector to store the next row
    }
    cout << "Lines read successfully: " << l << endl;                            // displaying the number or lines reads from the input file
  }
  else{
    cout << "Unable to open the specified file " << endl; // output if file can't be opened
  }
}

//function to read mnist dataset
void read_data2(string inputfile, Eigen::MatrixXi &X, Eigen::MatrixXi &Y) {}