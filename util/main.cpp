#include "Util.h"
#include "Eigen/Dense"
#include <iostream>
#include <string>
#include <map>
#include <vector>
using namespace std;
using namespace Eigen;

int main() {
	time_t start, stop;
	start = time(NULL);


	MatrixXf m = MatrixXf::Random(9, 7);
	MatrixXf n = MatrixXf::Random(3, 3);
	RowVectorXf b = RowVectorXf::Random(10);
	RowVectorXf a = RowVectorXf::Random(10);
	RowVectorXd c = RowVectorXd::Zero(10);
	cout << Util::extendMatrix(n, 2, 1, 4, 2) << endl << endl;
	cout << "m=" << endl << m << endl;
	cout << "colwise:" << endl << m.colwise().reverse() << endl;
	cout << "rowwise:" << endl << m.rowwise().reverse() << endl;
	cout << "rot180:" << endl << m.colwise().reverse().rowwise().reverse() << endl;
	cout << "rot180:" << endl << m.rowwise().reverse().colwise().reverse() << endl;
	cout << "test simple way to rot180:" << endl << m.reverse() << endl;

	cout << Util::validConv(m, n) << endl << endl;
	cout << Util::fullConv(m, n) << endl << endl;

	cout << endl << endl << m << endl;
	cout << endl << m.row(2) << endl;
	cout << endl << m.col(4) << endl;

	for (long i = 0; i < 1000000000; i++);

	stop = time(NULL);

	cout << "time " << stop - start << endl;

	vector<map<string, string> > layerConfig;
	//vector<int> x;
	map<string, double> paraConfig;
	paraConfig["batch_size"] = 50;
	paraConfig["variation_threshold"] = 0.001;
	paraConfig["class_num"] = 10;
	paraConfig["log_on"] = 1;
	paraConfig["detail_log_on"] = 1;

	



	return 0;
	//cout<<b*a<<endl;
}