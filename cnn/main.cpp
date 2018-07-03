#include <iostream>
#include "Eigen/Dense"
using namespace Eigen;
using namespace std;
int main()
{  
    MatrixXd m = MatrixXd::Random(3,3);                 //使用Random随机初始化3*3的矩阵
    m = (m + MatrixXd::Constant(3,3,1.2)) * 50;  
    cout << "m =" << endl << m << endl;  
    VectorXd v(3);                                      //这表示任意大小的（列）向量。
    v << 1, 2, 3;  
    cout << "m * v =" << endl << m * v << endl;
}