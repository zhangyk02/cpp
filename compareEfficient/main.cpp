#include <iostream>
#include "Conv.h"
#include "MyStopWatch.h"
using namespace std;

template < class T >
void ClearVector( vector< T >& vt )
{
	vector< T > vtTemp;
	vtTemp.swap( vt );
};

void Clear2dArray( double** &d2array , int r )
{
	for (int i = 0; i < r; i++)
		delete [] d2array[i];
	delete [] d2array;
};



int main() {
	// int*a = new int[4];
	// double b[5];
	// char c[6];

	// //cout << begin(a) << " " << end(a) << endl;
	// cout << begin(b) << " " << end(b) << endl;

	// cout << begin(c) << " " << end(c) << endl;

	// //cout << "end(array)-begin(array): " << end(a) - begin(a) << endl;
	// cout << "length a: "<< sizeof(a)/sizeof(a[0])<<endl;
	// cout << "end(array)-begin(array): " << end(b) - begin(b) << endl;
	// cout << "end(array)-begin(array): " << end(c) - begin(c) << endl;

	// int d[10][5];
	// double** e = new double*[7];
	// for (int i=0;i<7;i++) e[i]=new double[3];

	// cout << "end(array[][])-begin(array[]][]): " << end(d) - begin(d) << endl;
	// //cout << "end(array[][])-begin(array[][]): " << end(e) - begin(e) << endl;

	// cout << "end(array[])-begin(array[]): " << end(d[0]) - begin(d[0]) << endl;
	//cout << "end(array[])-begin(array[]): " << end(e[0]) - begin(e[0]) << endl;
	//

	vector<vector<double> > ones5x5 = vector<vector<double> >(5, vector<double>(5, 1.0));
	Conv::printVectorMatrix(ones5x5);

	double** dOnes5x7 = new double*[5];
	for (int i = 0; i < 5; i++) {
		dOnes5x7[i] = new double[7];
		for (int j = 0; j < 7; j++)
			dOnes5x7[i][j] = 1.5;
	}

	Conv::printDoubleMatrix(dOnes5x7, 5, 7);

	MatrixXf baseEigen9x7 = MatrixXf::Ones(9, 7);
	vector<vector<double> > ones9x7 = vector<vector<double> >(9, vector<double>(7, 1.0));
	double** dOnes9x7 = new double*[9];
	for (int i = 0; i < 9; i++) {
		dOnes9x7[i] = new double[7];
		for (int j = 0; j < 7; j++)
			dOnes9x7[i][j] = 1.0;
	}

	MatrixXf baseEigen4x3 = MatrixXf::Ones(4, 3);
	vector<vector<double> > ones4x3 = vector<vector<double> >(4, vector<double>(3, 1.0));
	double** dOnes4x3 = new double*[4];
	for (int i = 0; i < 4; i++) {
		dOnes4x3[i] = new double[3];
		for (int j = 0; j < 7; j++)
			dOnes4x3[i][j] = 1.0;
	}

	MatrixXf convResultEigen = Conv::validConv(baseEigen9x7, baseEigen4x3);
	vector<vector<double> > convResultVector = Conv::validConv(ones9x7, ones4x3);
	double** convResultDouble = Conv::validConv(dOnes9x7, dOnes4x3, 9, 7, 4, 3);

	cout << "convresult:" << endl << endl;
	cout << "Eigen:" << endl << convResultEigen << endl;
	cout << "vector:" << endl;
	Conv::printVectorMatrix(convResultVector);
	cout << "double:" << endl;
	Conv::printDoubleMatrix(convResultDouble, 6, 5);

	Clear2dArray(convResultDouble, 6);

	MatrixXf baseEigen28x28 = MatrixXf::Ones(28, 28);
	vector<vector<double> > vectorOnes28x28 = vector<vector<double> >(28, vector<double>(28, 1.0));
	double** doubleOnes28x28 = new double*[28];
	for (int i = 0; i < 28; i++) {
		doubleOnes28x28[i] = new double[28];
		for (int j = 0; j < 28; j++)
			doubleOnes28x28[i][j] = 1.0;
	}

	MatrixXf baseEigen5x5 = MatrixXf::Ones(5, 5);
	vector<vector<double> > vectorOnes5x5 = vector<vector<double> >(5, vector<double>(5, 1.0));
	double** doubleOnes5x5 = new double*[5];
	for (int i = 0; i < 5; i++) {
		doubleOnes5x5[i] = new double[5];
		for (int j = 0; j < 5; j++)
			doubleOnes5x5[i][j] = 1.0;
	}

	int batchSize = 50;
	int inputChannel = 6;
	int outputChannel = 4;
	int batchNum = 1200;

	MyStopWatch sw;
	for (int bn = 0; bn < batchNum; bn++)
		for (int b = 0; b < batchSize; b++)
			for (int ic = 0; ic < inputChannel; ic++)
				for (int oc = 0; oc < outputChannel; oc++)
					Conv::validConv(baseEigen28x28, baseEigen5x5);
	cout << "matrix conv: " << sw.timeIntervalFromLastClick() << "ms" << endl;

	for (int bn = 0; bn < batchNum; bn++)
		for (int b = 0; b < batchSize; b++)
			for (int ic = 0; ic < inputChannel; ic++)
				for (int oc = 0; oc < outputChannel; oc++)
				{
					vector<vector<double> > tmpVector = Conv::validConv(vectorOnes28x28, vectorOnes5x5);
					ClearVector(tmpVector);
				}
	cout << "vector conv: " << sw.timeIntervalFromLastClick() << "ms" << endl;

	for (int bn = 0; bn < batchNum; bn++)
		for (int b = 0; b < batchSize; b++)
			for (int ic = 0; ic < inputChannel; ic++)
				for (int oc = 0; oc < outputChannel; oc++)
				{
					double ** tmpDouble = Conv::validConv(doubleOnes28x28, doubleOnes5x5, 28, 28, 5, 5);
					Clear2dArray(tmpDouble, 24);
				}
	cout << "double conv: " << sw.timeIntervalFromLastClick() << "ms" << endl;









	return 0;
}