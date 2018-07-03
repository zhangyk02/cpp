#include "Eigen/Dense"
#ifndef UTIL_H
#define UTIL_H
using namespace Eigen;
using namespace std;
class Util {
public:
	static MatrixXf validConv(MatrixXf, MatrixXf);
	static MatrixXf fullConv(MatrixXf, MatrixXf);
	static MatrixXf extendMatrix(MatrixXf, int, int, int, int);

	//每一行内部做softmax，不同的行是不同的sample出的分类概率
	static MatrixXf rowSoftMax(MatrixXf);
	//每一列每部做softmax，不同的列是不同的sample出的分类概率
	static MatrixXf colSoftMax(MatrixXf);

	static int maxIndex(VectorXf);
};

MatrixXf Util::validConv(MatrixXf basePic, MatrixXf kernel) {
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

MatrixXf Util::extendMatrix(MatrixXf baseMat, int upAdded, int bottomAdded, int leftAdded, int rightAdded) {
	MatrixXf ans = MatrixXf::Zero(baseMat.rows() + upAdded + bottomAdded, baseMat.cols() + leftAdded + rightAdded);
	ans.block(upAdded, leftAdded, baseMat.rows(), baseMat.cols()) = baseMat;
	return ans;
}

MatrixXf Util::fullConv(MatrixXf basePic, MatrixXf kernel) {
	// 需要将basePic上下各扩展rKer-1行，左右各扩展cKer-1列

	int rBase = basePic.rows();
	int cBase = basePic.cols();
	int rKer = kernel.rows();
	int cKer = kernel.cols();

	return validConv(extendMatrix(basePic, rKer - 1, rKer - 1, cKer - 1, cKer - 1), kernel);

}



MatrixXf Util::rowSoftMax(MatrixXf mat) {
	MatrixXf ans(mat.rows(), mat.cols());
	for (int j = 0; j < mat.rows(); ++j)
	{
		float max = 0.0;
		float sum = 0.0;
		for (int k = 0; k < mat.cols(); ++k)
			if (max < mat(j, k))
				max = mat(j, k);
		for (int k = 0; k < mat.cols(); ++k)
		{
			mat(j, k) = exp(mat(j, k) - max);  // prevent data overflow
			sum += mat(j, k);
		}
		for (int k = 0; k < mat.cols(); ++k)
			ans(j, k) = mat(j, k) / sum;
	}
	return ans;
}

MatrixXf Util::colSoftMax(MatrixXf mat) {
	MatrixXf ans(mat.rows(), mat.cols());
	for (int j = 0; j < mat.cols(); ++j)
	{
		float max = 0.0;
		float sum = 0.0;
		for (int k = 0; k < mat.rows(); ++k)
			if (max < mat(k, j))
				max = mat(k, j);
		for (int k = 0; k < mat.rows(); ++k)
		{
			mat(k, j) = exp(mat(k, j) - max);  // prevent data overflow
			sum += mat(k, j);
		}
		for (int k = 0; k < mat.rows(); ++k)
			ans(k, j) = mat(k, j) / sum;
	}
	return ans;
}

int Util::maxIndex(VectorXf v) {
	if (v.rows() == 1) return 0;
	double max = v(0);
	int ind = 0;
	for (int i = 1; i < v.rows(); i++) {
		if (v(i) > max) {
			max = v(i);
			ind = i;
		}

	}
	return ind;
}

#endif