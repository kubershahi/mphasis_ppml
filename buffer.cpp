//==========================================
  // LOADING DATA:
  //==========================================
  // cout<<"Reading Data:"<<endl;
  // vector<vector<double> > X_train_input;   
  // vector<double> Y_train_input;            
  
  // // training data
  // read_insurance_data("datasets/medical/insurance_train.csv", X_train_input, Y_train_input);

  // MatrixXd X_train(N, d); 
  // MatrixXd Y_train(N, 1); 

  // for (int i = 0; i < N; i++)
  // {
  //   X_train.row(i) = VectorXd::Map(&X_train_input[i][0], d)/10.0;
  //   Y_train.row(i) = VectorXd::Map(&Y_train_input[i],1)/10000.0;
  // }

  // vector<vector<double> > X_test_input;    
  // vector<double> Y_test_input;             

  // read_insurance_data("datasets/medical/insurance_test.csv", X_test_input, Y_test_input); 

  // MatrixXd X_test(N_test, d); 
  // MatrixXd Y_test(N_test, 1); 

  // for (int i = 0; i < N_test; i++)
  // {
  //   X_test.row(i) = VectorXd::Map(&X_test_input[i][0], d)/10.0;
  //   Y_test.row(i) = VectorXd::Map(&Y_test_input[i],1)/10000.0;
  // }
  // MatrixXd w1 = MatrixXd::Random(d,1);
  
  //==========================================
  // MODEL PREDICTION:
  //==========================================

  // cout << endl << "==================================="<<endl;
  // cout << "PREDICTION (using trained weights):"<<endl;
  // cout << "==================================="<<endl<<endl;
  // MatrixXd pred = predict(X_test, Y_test, ideal_w);
  // //cout << pred <<endl;