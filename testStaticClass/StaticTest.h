 class StaticTest
 {
 public:
     StaticTest(int a, int b, int c);
     void GetNumber();
     void GetSum();
     static void f1(StaticTest &s);
 private:
     int A, B, C;
   static int Sum;
 };