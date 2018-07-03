#include <iostream>
#include <stdio.h>    
#include <sys/time.h>   
#include "MyStopWatch.h"   
using namespace std;
        

// class MyStopWatch{
// private:
// 	int64_t initTime;
// 	int64_t lastClickTime;
// 	int64_t recentClickTime;
// int64_t getCurrentTime()      //直接调用这个函数就行了，返回值最好是int64_t，long long应该也可以
//     {    
//        struct timeval tv;    
//        gettimeofday(&tv,NULL);    //该函数在sys/time.h头文件中
//        return tv.tv_sec * 1000 + tv.tv_usec / 1000;    
//     }   
// public:
// 	MyStopWatch(){
// 		initTime = getCurrentTime();
// 		lastClickTime = initTime;
// 	}

// 	void click(){
// 		recentClickTime = getCurrentTime();
// 	}

// 	int64_t timeIntervalFromLastClick(){
// 		click();
// 		int64_t ans = recentClickTime - lastClickTime;
// 		lastClickTime = recentClickTime;
// 		return ans;
// 	}	

// 	int64_t timeIntervalFromInit(){
// 		click();
// 		int64_t ans = recentClickTime - initTime;
// 		lastClickTime = recentClickTime;
// 		return ans;
// 	}	
// };

int main()
{
	MyStopWatch sw;
	for (int i=0;i<100000000;i++);
	cout<<sw.timeIntervalFromLastClick()<<" "<<sw.timeIntervalFromInit()<<endl;

	for (int i=0;i<100000000;i++);
	cout<<sw.timeIntervalFromLastClick()<<" "<<sw.timeIntervalFromInit()<<endl;

	return 0;

}