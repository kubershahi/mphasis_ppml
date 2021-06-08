
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>


using namespace std;

// vector<vector<float>> read_data(void){
//   ifstream

// }
int main() {

  string line;

  ifstream fin;
  fin.open("datasets/medical_cost/insurance_int.csv");

  vector<vector<float> > dataset;
  vector<float> temp;

  float age, sex, bmi, children, smoker, charges;
  string temp_sex, temp_smoker;

  int count = 0;

  if(fin.is_open()){
    cout << "File open successfully" << endl;
    
    while(getline(fin,line)){
      
      replace(line.begin(), line.end(), ',', ' ');
      istringstream temp_row(line);

      temp_row >> age >> temp_sex >> bmi >> children >> temp_smoker >> charges;

      temp.push_back(age);
      if (temp_sex == "male"){
        sex = 0; // male
      }
      else {
        sex = 1; // female
      }
      temp.push_back(sex);
      temp.push_back(bmi);
      temp.push_back(children);
      if (temp_smoker == "yes"){
        smoker = 1; //yes 
      }
      else {
        smoker = 0;   //no
      }
      temp.push_back(smoker);
      temp.push_back(charges);

      dataset.push_back(temp);
      temp.clear();

      count++;
    }

    cout << count << endl;

  }
  else{
    cout << "Unable to open the specified file " << endl;
  }

  for (int j = 0; j < 10; j++ ){
    cout << dataset[j][0]<< " " << dataset[j][1] << " " << dataset[j][2] << " " <<dataset[j][3] << " " << dataset[j][4] << " " << dataset[j][5] << endl;
  }

  cout << dataset.size() << endl;

  int size = 10;
  vector<vector<float> > train_X(dataset.begin(), dataset.begin() + size);
  cout << train_X.size() << endl;
  cout << train_X[1][0]<< " " << train_X[1][1] << " " << train_X[1][2] << " " <<train_X[1][3] << " " << train_X[1][4] << " " << train_X[1][5] << endl;

}