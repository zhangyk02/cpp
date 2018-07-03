#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <sstream>
using namespace std;
int main() {
	vector<int> a;
	a.push_back(1);
	a.push_back(2);
	cout << a[0] << " " << a[1] << endl;
	a.clear();
	a[100] = 2;
	cout << a[0] << " " << a[50] << " " << a[100] << endl;

	map<string, string> configMap;
	configMap["batch_size"] = "100";
	configMap["max_round"] = "50";
	string love = "77";
	string love2 = " 77";
	string love3 = "77 ";
	int ilove = atoi(love.c_str());
	int ilove2 = atoi(love2.c_str());
	int ilove3 = atoi(love3.c_str());
	cout << ilove << " " << ilove2 << " " << ilove3 << endl;

	ostringstream buf;
	double d = 10.24;
	buf << "d=" << d << endl;

	string bufToString = buf.str();

	cout << bufToString << bufToString;

	a = vector< int >(7);
	for (int i = 0; i < a.size(); i++) {
		cout << "a[" << i << "]=" << a[i] << " ";

	}

}