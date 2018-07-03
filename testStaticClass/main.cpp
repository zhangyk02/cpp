#include "StaticTest.h"
#include <stdlib.h>
 

int main(void)
 {
     StaticTest M(3, 7, 10), N(14, 9, 11);
     M.GetNumber();
     N.GetSum();
     M.GetNumber();
     N.GetSum();
     StaticTest::f1(M);
     system("pause");
     return 0;
 }