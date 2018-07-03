#include <iostream>
#include "Eigen/Dense"
using namespace Eigen;
using namespace std;


// int main(){
// 	Matrix2d mat;
//   mat << 1, 2,
//          3, 4;
//     	Matrix2d mat2;
//   mat2 << 3, 2,
//          5, 1;
//   cout<<mat.rows()<<mat.cols()<<mat2.rows()<<mat2.cols()<<endl;
//   cout<<mat.array()<<endl;
//   cout<<mat2.array()<<endl;
//   cout<<mat.cwiseProduct(mat2).sum();
//   //cout<<mat.dot(mat2)<<endl;
//   // cout << "Here is mat.sum():       " << mat.sum()       << endl;
//   // cout << "Here is mat.prod():      " << mat.prod()      << endl;
//   // cout << "Here is mat.mean():      " << mat.mean()      << endl;
//   // cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << endl;
//   // cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << endl;
//   // cout << "Here is mat.trace():     " << mat.trace()     << endl;
//   return 0;
// }

//转置
// int main()
// {
//   Matrix2d mat;
//   mat << 1, 2,
//          3, 4;
//   Vector2d u(-1,1), v(2,0);
//   std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
//   std::cout << "Here is mat*u:\n" << mat*u << std::endl;
//   std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
//   std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
//   std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
//   std::cout << "Let's multiply mat by itself" << std::endl;
//   mat = mat*mat;
//   std::cout << "Now mat is mat:\n" << mat << std::endl;
// }

// int main()
// {
//     MatrixXd m = MatrixXd::Random(3,3);                 //使用Random随机初始化3*3的矩阵
//     m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
//     cout << "m =" << endl << m << endl;
//     VectorXd v(3);                                      //这表示任意大小的（列）向量。
//     v << 1, 2, 3;
//     cout << "m * v =" << endl << m * v << endl;
// }

//向量内积、求共轭向量
// int main()
// {
//   Vector3d v(1,2,3);
//   Vector3d w(0,1,2);
//   cout << "Dot product: " << v.dot(w) << endl;
//   double dp = v.adjoint()*w; // automatic conversion of the inner product to a scalar
//   cout << "Dot product via a matrix product: " << dp << endl;
//   cout << "v.adjoint()"<<v.adjoint()<<endl;
//   //cout << "v*w"<<v*w<<endl;
//   cout << "Cross product:\n" << v.cross(w) << endl;
// }


//矩阵子块（子矩阵）
int main()
{
	Eigen::MatrixXf m(5, 4);
	m <<  1, 2, 3, 4,
	5, 6, 7, 8,
	9, 10, 11, 12,
	13, 14, 15, 16,
	17, 18, 19, 20;
	cout << "Block in the middle" << endl;
	cout << m.block<2, 2>(1, 1) << endl << endl;
	// MatrixXf x = MatrixXf::Zero(2, 2);
	// m.block(0, 0, 2, 2) = x;
	cout << m << endl;
	// for (int i = 1; i <= 3; ++i)
	// {
	// 	cout << "Block of size " << i << "x" << i << endl;
	// 	cout << m.block(0, 0, i, i) << endl << endl;
	// }

	cout << endl << endl;

	Map<VectorXf> v(m.data(), m.size());
	Map<MatrixXf> n(v.data(), m.rows(), m.cols());

	cout << "v:" << endl << v << endl;

	cout << "n:" << endl << n << endl;

	MatrixXf p = MatrixXf(2, 4);

	cout << p << endl;

	cout<<m.block(0,0,2,2)<<endl;

	cout<<"m: "<<m.size()<<endl;

	m.zero();

	cout<<m<<endl;


}



