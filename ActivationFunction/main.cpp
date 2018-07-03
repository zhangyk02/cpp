#include <iostream>
#include "Actfunc.h"
using namespace std;

int main()
{
    ActFunc *foo=new Prelu();
    foo->setInput(-0.3);
    foo->forward();
    cout<<"test:"<<foo->getOutput()<<"  "<<foo->getDerive()<<endl;
    return 1;
}