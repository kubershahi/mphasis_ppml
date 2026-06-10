#ifndef SECML_READ_DATA_HPP
#define SECML_READ_DATA_HPP

#include<vector>
#include<Eigen/Dense>

void read_insurance_data(std::string inputfile, std::vector<std::vector<double> > &X, std::vector<double> &Y);
void read_data(std::string inputfile, std::vector<std::vector<double> > &X, std::vector<double> &Y);

#endif