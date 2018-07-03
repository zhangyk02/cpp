#include <iostream>
#include "Polymorphic.h"
using namespace std;
int main() {
	base gs = grandson();
	gs.pr();
	base s = son();
	s.pr();
	base b;
	b.pr();

	base* x[3];
	x[0] = new grandson();;
	x[1] = new son();
	x[2] = new base();
	x[0]->pr();
	x[1]->pr();

	x[2]->pr();

	int i = 3;
	// base* y = new base*[i];
	// y[0] = new grandson();;
	// y[1] = new son();
	// y[2] = new base();
	// y[1]->pr();

	base** value = new base* [i];
	value[0] = new grandson();
	value[1] = new son();
	value[2] = new base();
	value[0]->pr();
	value[1]->pr();
	value[2]->pr();

	base* t = new base[i];
	t[0] = base();
	t[1] = son();
	t[2] = grandson();
	t[0].pr();
	t[1].pr();
	t[2].pr();








	// son *ps;
	// ps->pr();
}