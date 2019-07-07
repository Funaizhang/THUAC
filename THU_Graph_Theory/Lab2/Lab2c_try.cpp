//  
//  Lab2c.cpp
//  
//  Created by Zhang Naifu 2018280351 on 26/05/2019.
//
//
/*
C. 必修课(2)

问题描述
某校的计算机系有 n 门必修课程。学生需要修完所有必修课程才能毕业。 有些课程有前置课程，需要先修完它们才能修这些课程；而其他课程没有。
我们定义一个“课程序列”，为课程的非空序列，满足序列中每门课程都是后一门课程的前置课程。形式化地说，令课程序列为 {l1,l2,⋯,lk}，则对于 1≤i<k，满足 li 是 li+1 的前置课程，且该课程序列的长度为 k。

现在校方想进行课程改革，对培养计划进行减负。他们需要知道：

在所有课程序列中，长度的最大值；
最多能同时选出多少个不相交的课程序列，使得它们的长度均等于第一问的答案。
我们称两个课程序列不相交，当且仅当不存在一门课程，同时属于这两个课程序列。

输入格式
从标准输入读入数据。
输入的第一行包含一个整数 n，表示课程的数量。
接下来 n 行，第 i(1≤i≤n) 行每行 ci+1 个整数。 第一个整数为 ci(0≤ci<n)，表示该课程的前置课程数量。 接下来 ci 个互不相同的整数，表示该课程的前置课程的编号。 相邻两数之间用一个空格隔开。
该校保证，每名入学的学生，一定能够在有限的时间内毕业。即，前置课程关系不会存在环。

输出格式
输出到标准输出。
输出一行两个整数，分别表示这两问的答案。

样例输入1
5
3 2 3 5
0
1 2
0
1 4
样例输出1
3 1
样例解释1
一种可能的解释：
1号课程为“数据结构”，前置课程为“程序设计基础” “面向对象程序设计基础”和“离散数学(2)”；
2号课程为“程序设计基础”，无前置课程；
3号课程为“面向对象程序设计基础”，前置课程为“程序设计基础”；
4号课程为“离散数学(1)”，无前置课程；
5号课程为“离散数学(2)”，前置课程为“离散数学(1)”。
第一问的答案为3。由于“数据结构”同时属于仅有的两个最长课程序列，因此第二问的答案为1。

样例输入2
13
0
1 1
2 2 8
1 3
1 4
1 2
0
1 7
0
1 9
1 10
3 1 7 11
1 12
样例输出2
5 2

数据规模与约定
对于30%的数据，1≤n≤10；
对于70%的数据，1≤n≤100；
对于100%的数据，1≤n≤1000，∑ci≤10n。

时间限制：1s
空间限制：256MB

提示
[第一问可用关键路径求解，第二问可用最大流求解。]
*/

// #include <stdio.h> 
// #include <stdlib.h> 
#include <math.h> 
// #include <vector>
#include <iostream>
#include <list>
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
using namespace std;


// const int MAX_ci= 1000000;
const int MAX_n = 1000;

struct Course {
    int p;
    int max;
    // vector<int> pre; //prerequisites
    int *pre; //prerequisites
    // int *post; //courses that need it as pre
};

int find_cmax(int id, struct Course *courses) {
    int cmax = 0;

    if (courses[id].p == 0) {
        courses[id].max = 1;
        return 1;
    }

    //loops all the prereqs
    for (int j = 0; j < courses[id].p; j++) {
        if (courses[id].max == 0) {
            // cout << "Course" << id << ": if \t cmax=" << cmax << " " << 1+find_cmax(courses[id].pre[j], courses) << endl;
            cmax = MAX(cmax, 1 + find_cmax(courses[id].pre[j], courses));
        } else {
            // cout << "Course" << id << ": else \t cmax=" << cmax << " " << courses[id].max << endl;
            cmax = MAX(cmax, courses[id].max);
        }
    }

    // cout << "Finally: " << cmax << endl;
    courses[id].max = cmax;
    return cmax;
}



int main() { 
    int n, m, p=0, q;
    int x, y;

    // read n
    cin >> n;

    // read courses & their prerequisites
    Course courses[n];
    for (int i=0; i<n; i++) {
        cin >> courses[i].p;
        courses[i].max = 0;
        courses[i].pre = (int*) malloc(courses[i].p * sizeof(int));

        // read all the ci
        for (int j = 0; j < courses[i].p; j++) {
            int p_temp;
            cin >> p_temp;
            courses[i].pre[j] = p_temp - 1;
        }
    }

    // classTime array holds the time needed for each class
    int classTime[n];
    // maxTime finds the longest course path
    int maxTime = 0;
    // inCount is number of courses with no pre; outCount is number of courses with maxTime
    int inCount = 0, outCount = 0;

    for (int i=0; i<n; i++) {
        classTime[i] = find_cmax(i, courses);

        // compute maxTime and outCount
        if (classTime[i] > maxTime) {
            maxTime = classTime[i];
            outCount = 1;
        } else if (classTime[i] == maxTime) {
            outCount++;
        }

        // compute inCount
        if (courses[i].p == 0)
            inCount++;
    }

    int result;
    result = MIN(inCount, outCount);

    cout << maxTime << " " << result << endl;

    for (int i=0; i<n; i++) {
        free(courses[i].pre);
    }
    return 0; 
} 