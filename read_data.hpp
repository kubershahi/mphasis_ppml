#ifndef READ_DATA_HPP
#define READ_DATA_HPP

#include<vector>
#include<Eigen/Dense>

void read_data(std::string inputfile, std::vector<std::vector<float> > &X, std::vector<float> &Y);
void read_data2(std::string inputfile, Eigen::MatrixXi &X, Eigen::MatrixXi &Y);

#endif