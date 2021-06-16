#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void insert(vector<int>  &l, int item){
  l.push_back(item);
}

int main(){

  vector<int > l;
  insert(l,1);
  
  for (int i = 0; i < 4; i++){
    cout << l[i] << endl;
  }
}