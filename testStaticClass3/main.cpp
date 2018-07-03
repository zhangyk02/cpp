#include<iostream>
using namespace std;
public static class A{
	public:
	static void printInt(int a){
	cout<<a<<endl;
	}
}
int main() 
{ 
int a=1;
A::printInt(a);
return 0;
}