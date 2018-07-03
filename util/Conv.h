#include "Eigen/Dense"
using namespace Eigen;
using namespace std;
#ifndef CONV_H
#define CONV_H
class conv{ 
public 
MatrixXf validConv(MatrixXf basePic, MatrixXf kernel) {
	int rBase = basePic.rows();
	int cBase = basePic.cols();
	int rKer = kernel.rows();
	int cKer = kernel.cols();
	int rAns = rBase - rKer + 1;
	int cAns = cBase - cKer + 1;
	MatrixXf ans(rAns, cAns);
	for (int i = 0; i < rAns; i++) {
		for (int j = 0; j < cAns; j++) {
			ans(i, j) = basePic.block(i, j, rKer, cKer).cwiseProduct(kernel).sum();
		}
	}
	return ans;
}

double[][]
} 
#endif