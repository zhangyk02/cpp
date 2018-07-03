#include "StaticTest.h"
#include <iostream>
using namespace std;

int StaticTest::Sum = 0;//静态成员在此初始化

StaticTest::StaticTest(int a, int b, int c)
{
     A = a;
     B = b;
     C = c;
     Sum += A + B + C;
 }

 void StaticTest::GetNumber()
 {
     cout << "Number = " << endl;
 }
 
 void StaticTest::GetSum()
 {
     cout << "Sum = " << Sum <<endl;
 }
 
 void StaticTest::f1(StaticTest &s)
 {
     
     cout << s.A << endl;//静态方法不能直接调用一般成员，可以通过对象引用实现调用
     cout << Sum <<endl;
 }