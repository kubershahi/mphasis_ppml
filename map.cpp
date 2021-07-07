#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>
#include <Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;

#define SCALING_FACTOR 8192 // Precision of 13 bits

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi64;

// void insert(vector<int>  &l, int item){
//   l.push_back(item);
// }


// function that takes float to unit64
MatrixXi64 floattouint64(MatrixXd A)
{
  MatrixXi64 res(A.rows(),A.cols());

  for (int i = 0; i < A.rows(); i++)
  {
    for (int j = 0; j < A.cols(); j++)
    {
      double a = A(i,j);
      if ( a >= 0)
      {
        res(i,j) = (uint64_t) (a * SCALING_FACTOR);
        //cout<< res(i,j) << " is positive"<<endl;
      }
      else
      {
        a = abs(a * SCALING_FACTOR);
        res(i,j) = (uint64_t) pow(2,64) - a;
        //cout<< res(i,j) << " is negative"<<endl;
      }
    }
  } 
  return res;
}

//function that takes unit64 to float
MatrixXd uint64tofloat(MatrixXi64 A)
{
  MatrixXd res(A.rows(),A.cols());
  for (int i = 0; i < A.rows(); i++)
  {
    for (int j = 0; j < A.cols(); j++)
    {
      uint64_t a = A(i,j);
      if (a & (1UL << 63))
      {
        res(i,j) = -((double) pow(2,64) - a)/SCALING_FACTOR;
        //cout<< res(i,j) << " is negative"<<endl;
      }
      else
      {
        res(i,j) = ((double) a)/SCALING_FACTOR;
        //cout<< res(i,j) << " is positive"<<endl;
      }
        
    }
  } 
  return res;
}

MatrixXi64 truncate(MatrixXi64 A, int factor)
{
  MatrixXi64 res(A.rows(),A.cols());
  for (int i = 0; i < A.rows(); i++)
  {
    for (int j = 0; j < A.cols(); j++)
    {
      uint64_t a = A(i,j);
      if (a & (1UL << 63))
      {
        res(i,j) = pow(2,64) - (pow(2,64) - a)/factor;
        //cout<< res(i,j) << " is negative"<<endl;
      }
      else
      {
        res(i,j) = a/factor;
        //cout<< res(i,j) << " is positive"<<endl;
      }
        
    }
  } 
  return res;
}

// ==================================

int N = 6; //6
int N_test = 6;
int d = 2; //5
int B = 3; //3
int NUM_EPOCHS = 3; // change; shuffle order

MatrixXi64 idealLinearRegression(MatrixXi64 X, MatrixXi64 Y, MatrixXi64 w) // ideal functionality
{
  // w -= a/|B| X^T .(X.w - Y)
  //int t = (N * NUM_EPOCHS)/B; // E = 1
  //float eta = 0.01;
  for(int e = 0; e < NUM_EPOCHS; e ++)
  { cout<< "Epoch Number: "<< e+1;
    cout<<" Progress: ";
    float epoch_loss = 0.0;

    for(int i = 0; i < int(N/B); i ++)
    { 
      cout<<"===";
      MatrixXi64 YY = X.block(B * i,0,B,X.cols()) * w; // YY = X_B_i.w

      //truncation:
      //YY /= SCALING_FACTOR;
      YY = truncate(YY, SCALING_FACTOR);

      //test
      //MatrixXd YYtest = uint64tofloat(YY); // descaling
      //cout<< "yhat: "<< endl << YYtest << endl;

      MatrixXi64 D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i

      //test
      //MatrixXd Dtest = uint64tofloat(D);// descaling
      //cout<< "diff: "<< endl << Dtest << endl;
      
      // Loss Computation
      MatrixXd loss = uint64tofloat(D).transpose() * uint64tofloat(D);
      //MatrixXd loss = temp_loss.cast<double>();

      MatrixXi64 delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
      //cout<< "grad_raw: " << endl << delta << endl;

      //truncation:
      //delta /= SCALING_FACTOR;
      delta = truncate(delta, SCALING_FACTOR);
      //test
      //MatrixXd gradtest = uint64tofloat(delta);// descaling
      //cout<< "grad: " << endl << gradtest << endl;


      delta = truncate(delta, 128); // eta/B * delta, eta/B = 128 = 2^7 
      w = w - delta; // w -= a/B * delta
      //cout<<"weights: "<< endl << uint64tofloat(w) <<endl;
      epoch_loss += loss(0,0);
    }
    cout<<endl;
    cout<< "Loss: "<< epoch_loss/N << endl;
  }
  
  return w;
}



MatrixXf predict(MatrixXf X, MatrixXf Y, MatrixXf w)
{ MatrixXf pred = X * w;
  MatrixXf diff = pred - Y;
  MatrixXf loss = diff.transpose() * diff;
  cout<< "Test Loss: "<< loss(0,0)/X.rows() << endl;
  return pred;
}

// ==================================

void bin(long n)
{
  long i;
  cout << "0";
  for (i = 1 << 30; i > 0; i = i / 2)
  {
    if((n & i) != 0) cout << "1";
    else cout << "0";
  }
}


int main(){

  // MatrixXf X = MatrixXf::Random(1);
  // MatrixXf Y = MatrixXf::Random(1);

  //MatrixXd X(6,2);
  //X << 4.3245,1.2341, 2.3451,8.34156, 1.437245,0.83464, 3.43425,2.76845, 1.3452,4.9234, 6.97324,7.3245;
  //MatrixXd Y(6,1);
  //Y << 3,1,4,3,1,2;
  //MatrixXd w(2,1);
  //w << 1.65921924,1.62628418;

  // MatrixXf X(6,2);
  // X << 4.243,1.244,-2.983,8.382,-1.534,0.913,3.142,2.434,-1.039,4.012,6.144,7.782;
  // MatrixXf Y(6,1);
  // Y << 2,14,-1,1,7,8;
  // MatrixXf w(2,1);
  // w << 1.65921924,1.62628418;

  // cout << "Here is the matrix X:\n" << X <<endl;
  // cout << "Here is the matrix w:\n" << w <<endl;
  // //cout << "Here is the expected X.w:\n" << X * w <<endl;

  // X = X * SCALING_FACTOR; // double to uint_64
  // Y = Y * SCALING_FACTOR; // double to uint_64
  // w = w * SCALING_FACTOR; // double to uint_64

  // //cout << "Here is the SCALED matrix X:\n" << X <<endl;
  // //cout << "Here is the SCALED matrix w:\n" << w <<endl;

  // MatrixXi64 X_ = X.cast<uint64_t>();
  // MatrixXi64 Y_ = Y.cast<uint64_t>();
  // MatrixXi64 w_ = w.cast<uint64_t>();

  // cout << "Here is the SCALED and MAPPED matrix X:\n" << X_ <<endl;
  // cout << "Here is the SCALED and MAPPED matrix w:\n" << w_ <<endl;

  // //MatrixXi64 U = X_ * w_;
  // //U /= SCALING_FACTOR; // truncation

  // //MatrixXd U_ = U.cast<double>();
  // //U_ /= SCALING_FACTOR; // descaling

  // //cout << "Here is the calculated X.w:\n" << U_ <<endl;

  // // Figure NEGATIVE NUMBERS OUT

  // //MatrixXd test1(3,1);
  // //test1 << 1,2,-3;
  // //test1 = test1 * SCALING_FACTOR;
  // //MatrixXi64 T1 = test1.cast<uint64_t>();
  // //cout << "Here is the SCALED and MAPPED matrix T1:\n" << T1 <<endl;

  // //MatrixXd T1_ = uint64tofloat(T1);
  // //cout << "Here is the DESCALED matrix T1_:\n" << T1_ <<endl;

  // MatrixXi64 new_w = idealLinearRegression(X_,Y_,w_);
  // MatrixXd new_w_f = uint64tofloat(new_w); // descaling

  // cout<<"Final weights are: "<< new_w_f <<endl;


  // =================================
  // mapping example
  
  double X = 10.15723545348;
  double Y = 5.23423452345;
  double Z = X * Y; // the value we want to approximate
  

  cout << endl << "X, 1010.001010: "<< X << endl;
  cout << "Y, 101.001111: "<< Y << endl;
  cout << "Z, 110101.001010111: " << Z << endl << endl;

  X = X * SCALING_FACTOR; 
  Y = Y * SCALING_FACTOR;

  cout << fixed;
  cout << "After scaling by 13 bits i.e 2^13: " << endl;
  cout << "X: "<< X << endl;
  cout << "Y: "<< Y << endl << endl;
  
  cout << "After changing to integer " << endl;
  uint64_t x = (uint64_t) X ;
  uint64_t y = (uint64_t) Y ;

  cout << "x, 1010001010: "<< x << endl;
  cout << "y, 101001111 : "<< y << endl << endl;
  
  uint64_t z = x * y;
  cout << "z before truncation, 110101001010010110: "<< z << endl;
  z /= SCALING_FACTOR; // truncation product
  cout << "z after  truncation, 110101001010      : "<< z << endl << endl;

  double zz = (double) z;
  zz /= SCALING_FACTOR;
  cout << "z after cast,        110101.001010     : "<< zz << endl << endl;

  // // selecting certain columns of a Matrix
  // Map<MatrixXf> X1(X.data()+5,5,X.cols());

  // casting a matrix;
  // MatrixXi x = X.cast<int>();
}