#include <math.h>
#ifndef ACTFUNC_H
#define ACTFUNC_H
class ActFunc{
protected:
	double input;
	double output;
public:
	ActFunc();
	~ActFunc();
	virtual void forward();
	virtual double getDerive();
	void setInput(double);
	double getOutput();
};

void ActFunc::setInput(double x){
	input = x;
}

double ActFunc::getOutput(){
	return output;
}

ActFunc::ActFunc(){

}

ActFunc::~ActFunc(){

}

void ActFunc::forward(){
	output=input;
}

double ActFunc::getDerive(){
	return output;
}

class Sigmoid:public ActFunc{
public:
	void forward(){
		output = 1/(1+exp(-input));
	}
	double getDerive(){
		return (1-output)*output;
	}
};

class Relu:public ActFunc{
public:
	void forward(){
		if (input>=0)
			output = input;
		else
			output = 0;
	}
	double getDerive(){
		if (output>=0)
			return 1.0;
		else 
			return 0.0;
	}
};

class Tanh:public ActFunc{
public:
	void forward(){
		output = 2/(1+exp(-2*input))-1;
	}
	double getDerive(){
		return 1-output*output;
	}
};

class Prelu:public ActFunc{
private:
	double alpha;
public:
	Prelu(double a){
		alpha = a<0?-a:a;
	} 
	Prelu(){
		alpha = 0.1;
	}
	void forward(){
		if (input>=0)
			output = input;
		else
			output = alpha*input;
	}
	double getDerive(){
		if (output>=0)
			return 1.0;
		else 
			return alpha;
	}
};

#endif