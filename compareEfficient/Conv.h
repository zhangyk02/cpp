#include "Eigen/Dense"
#include <string>
#include <iostream>
#include <vector>
using namespace Eigen;
using namespace std;
#ifndef CONV_H
#define CONV_H
class Conv {
public:
	static MatrixXf validConv(MatrixXf basePic, MatrixXf kernel) {
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

	static double** validConv(double** basePic, double** kernel, int rBase, int cBase, int rKer, int cKer) {
		// int rBase = end(basePic) - begin(basePic[0]);
		// int cBase = end(basePic[0]) - begin(basePic[0]);
		// int rKer = end(kernel) - begin(kernel);
		// int cKer = end(kernel[0]) - begin(kernel[0]);
		int rAns = rBase - rKer + 1;
		int cAns = cBase - cKer + 1;
		double **ans = new double*[rAns];
		for (int i = 0; i < rAns; i++) {
			ans[i] = new double[cAns];
			for (int j = 0; j < cAns; j++) {
				ans[i][j] = 0;
				for (int m = 0; m < rKer; m++)
					for (int n = 0; n < cKer; n++)
						ans[i][j] += basePic[i + m][j + n] * kernel[m][n];
			}
		}
		return ans;
	}

	static vector<vector<double> > validConv(vector<vector<double> > basePic, vector<vector<double> > kernel) {
		int rBase = basePic.size();
		int cBase = basePic[0].size();
		int rKer = kernel.size();
		int cKer = kernel[0].size();
		int rAns = rBase - rKer + 1;
		int cAns = cBase - cKer + 1;
		vector<vector<double> > ans = vector<vector<double> >(rAns, vector<double>(cAns, 0.0));
		for (int i = 0; i < rAns; i++) {
			for (int j = 0; j < cAns; j++) {
				for (int m = 0; m < rKer; m++)
					for (int n = 0; n < cKer; n++)
						ans[i][j] += basePic[i + m][j + n] * kernel[m][n];
			}
		}
		return ans;
	}

	static void printDoubleMatrix(double** mat, int r, int c) {
		// int r = end(mat) - begin(mat);
		// int c = end(mat[0]) - begin(mat[0]);
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				cout << mat[i][j];
				if (j < c - 1) cout << ",";
			}
			cout << endl;
		}
	}

	static void printVectorMatrix(vector<vector<double> > mat) {
		int r = mat.size();
		int c = mat[0].size();
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				cout << mat[i][j];
				if (j < c - 1) cout << ",";
			}
			cout << endl;
		}
	}

};
#endif