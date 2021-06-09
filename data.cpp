#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "read_data.hpp"

using namespace std;

int main() {
  vector<vector<float> > dataset = read_data();

  for (int j = 0; j < 10; j++ ){
    cout << dataset[j][0]<< " " << dataset[j][1] << " " << dataset[j][2] << " " <<dataset[j][3] << " " << dataset[j][4] << " " << dataset[j][5] << endl;
  }

  cout << dataset.size() << endl;

  int size = 10;
  vector<vector<float> > train_X(dataset.begin(), dataset.begin() + size);
  cout << train_X.size() << endl;
  cout << train_X[1][0]<< " " << train_X[1][1] << " " << train_X[1][2] << " " <<train_X[1][3] << " " << train_X[1][4] << " " << train_X[1][5] << endl;

}