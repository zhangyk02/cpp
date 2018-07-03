
#include <iostream>
#ifndef Polymorphic_h
#define Polymorphic_h
using namespace std;

class base{
public:
	virtual void pr(){
		cout<<"base"<<endl;
	}
};
class son:public base{
public:
	void pr(){
		cout<<"son"<<endl;
	}
};
class grandson:public son{
public:
	void pr(){
		cout<<"grandson"<<endl;
	}
};
#endif