#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>

using namespace std;

int main(){
  vector<float> data;

  float f = 10.00;
  data.push_back(f);
  float g = 20.00;
  data.push_back(g);

  cout << data[0] << endl;
}