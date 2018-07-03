#include<string>
#include<iostream>
#include<map>
#ifndef TESTENUM_H
#define TESTENUM_H
#define _MAX_PATH 32768
using namespace std;
class SwitchString {
public:
	SwitchString() {
		Initialize();
	}

// 	void enumValue(string s){
// cout<<s_mapStringValues[s]<<endl
// 	}

	int run() {
		// Loop until the user stops the program
		while (1)
		{
			// Get the user's input
			cout << "Please enter a string (end to terminate): ";
			cout.flush();
			cin.getline(szInput, _MAX_PATH);
			// Switch on the value
			cout << szInput << " " << s_mapStringValues[szInput] << endl;
			switch (s_mapStringValues[szInput])
			{
			case evStringValue1:
				cout << "Detected the first valid string." << endl;
				break;
			case evStringValue2:
				cout << "Detected the second valid string." << endl;
				break;
			case evStringValue3:
				cout << "Detected the third valid string." << endl;
				break;
			case evEnd:
				cout << "Detected program end command. "
				     << "Programm will be stopped." << endl;
				return (0);
			default:
				cout << "'" << szInput
				     << "' is an invalid string. s_mapStringValues now contains "
				     << s_mapStringValues.size()
				     << " entries." << endl;
				break;
			}
		}
		return 1;
	}

private:
	// Value-Defintions of the different String values
	enum StringValue { evNotDefined,
	                   evStringValue1,
	                   evStringValue2,
	                   evStringValue3,
	                   evEnd
	                 };

// Map to associate the strings with the enum values
	std::map<std::string, StringValue> s_mapStringValues;

// User input
	char szInput[_MAX_PATH];

// Intialization
	void Initialize()
	{
		s_mapStringValues["First Value"] = evStringValue1;
		s_mapStringValues["Second Value"] = evStringValue2;
		s_mapStringValues["Third Value"] = evStringValue3;
		s_mapStringValues["end"] = evEnd;

		cout << "s_mapStringValues contains "
		     << s_mapStringValues.size()
		     << " entries." << endl;
	}

};
#endif