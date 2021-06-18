#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// void insert(vector<int>  &l, int item){
//   l.push_back(item);
// }

int main(){

  MatrixXf X = MatrixXf::Random(10,10);
  
  // for (int i = 0; i < 5; i++){
  //   cout << X.row(i) << endl;
  // }
  cout << X << endl;
  cout << "\n" << endl;
  Map<MatrixXf> X1(X.data()+5,5,X.cols());

  cout << X1 << endl;
}