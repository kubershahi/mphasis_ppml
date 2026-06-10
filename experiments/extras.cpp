void verify()
{
  /*

  MatrixXi shares[2];

  share(X, shares);
  MatrixXi X_0 = shares[0];
  MatrixXi X_1 = shares[1];
  share(Y, shares);
  MatrixXi Y_0 = shares[0];
  MatrixXi Y_1 = shares[1];


  //cout << "Here is the matrix X_0 + X_1:\n" << X_0 + X_1 <<endl;
  //cout << "Here is the matrix Y_0 + Y_1:\n" << Y_0 + Y_1 <<endl;

  // Multiplication
  MatrixXi w_0 = MatrixXi::Random(5,1) / 10000000;
  MatrixXi w_1 = MatrixXi::Random(5,1) / 10000000;
  MatrixXi w = rec(w_0, w_1);
  //cout << "Here is the matrix w_0 + w_1:\n" << w_0 + w_1 <<endl;
  
  //Masking

  MatrixXi U = MatrixXi::Random(X.rows(),X.cols()) / 10000;
  share(U, shares);
  MatrixXi U_0 = shares[0]; 
  MatrixXi U_1 = shares[1];

  MatrixXi E_0 = X_0 - U_0;
  MatrixXi E_1 = X_1 - U_1;
  MatrixXi E = rec(E_0, E_1);

  cout << "Here is the matrix U + E:\n" << U + E <<endl;
  //cout<< "Dimensions of X is: "<<X.rows()<<" "<<X.cols()<<endl;

  MatrixXi V = MatrixXi::Random(w.rows(),w.cols()) / 10000;
  share(V, shares);
  MatrixXi V_0 = shares[0]; 
  MatrixXi V_1 = shares[1];

  MatrixXi F_0 = w_0 - V_0;
  MatrixXi F_1 = w_1 - V_1;
  MatrixXi F = rec(F_0, F_1);

  MatrixXi Z = U * V;
  share(Z, shares);
  MatrixXi Z_0 = shares[0];
  MatrixXi Z_1 = shares[1]; 

  MatrixXi prod_0 = mult(0, X_0, w_0, E, F, Z_0);
  MatrixXi prod_1 = mult(1, X_1, w_1, E, F, Z_1);

  MatrixXi prod_test = X * w;
  cout << "Here is the required product:\n" << prod_test <<endl;
  MatrixXi prod = rec(prod_0, prod_1);
  cout << "Here is the calculated product:\n" << prod <<endl;

  */
}

//==========================================
  // Random data:
  //==========================================

  //MatrixXi X = MatrixXi::Random(N,d) / 10000000; // n = 6, d = 5, training samples
  //cout << "Here is the matrix X:\n" << X <<endl;
  //MatrixXi Y = MatrixXi::Random(N,1) / 10000000; // labels
  //cout << "Here is the matrix Y:\n" << Y <<endl;
  //MatrixXi w = MatrixXi::Random(d,1) / 100000000;

  //MatrixXi new_w = linearRegression(X,Y,w);
  //cout << "Final weights (under Privacy Preserving) are:\n" << new_w <<endl;
  //==========================================

  //==========================================
  // MODEL PREDICTION:
  //==========================================
/*
  cout << endl << "==================================="<<endl;
  cout << "PREDICTION (using trained weights):"<<endl;
  cout << "==================================="<<endl<<endl;
  MatrixXd pred = predict(X1_test, Y1_test, new_w);
  cout << pred <<endl;

  
  cout << endl << "Single example predictions: " << endl;
  for (int k = 500; k < 701; k += 100){
    cout << "True Label: " << Y1_test.row(k) << endl;
    MatrixXd pred_i = predict(X1_test.row(k), ideal_w);
    cout << "Ideal Prediction: " << pred_i <<endl;
    MatrixXd pred_p = predict(X1_test.row(k), new_w);
    cout << "PP Prediction: " << pred_p <<endl;
  }
  */