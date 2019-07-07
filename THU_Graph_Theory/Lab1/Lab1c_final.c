//  
//  Lab1c.c
//  
//  Created by Zhang Naifu 2018280351 on 01/04/2019.
//
//
/*
问题描述
某校的计算机系有 n 门必修课程。学生需要修完所有必修课程才能毕业。
每门课程都需要一定的学时去完成。有些课程有前置课程，需要先修完它们才能修这些课程；而其他课程没有。 不同于大多数学校，学生可以在任何时候进行选课，且同时选课的数量没有限制。
现在校方想进行课程改革，需要知道：
从入学开始，每门课程最早可能完成的时间（单位：学时）；
对每一门课程，若减少其学时，是否能降低入学到毕业的最短时间。

输入格式
从标准输入读入数据。
输入的第一行包含一个整数 n，表示课程的数量。
接下来 n 行，第 i(1≤i≤n) 行每行 ci+2 个整数。 第一个整数为 ti(1≤ti≤10000)，表示修完该课程所需的学时。 第二个整数为 ci(0≤ci<n)，表示该课程的前置课程数量。 接下来 ci 个互不相同的整数，表示该课程的前置课程的编号。 相邻两数之间用一个空格隔开。
该校保证，每名入学的学生，一定能够在有限的时间内毕业。

输出格式
输出到标准输出。
输出共 n 行。
第 i(1≤i≤n) 行包含两个整数：第一个整数表示编号为 i 的课程最早可能完成的时间；第二个整数表示，若将该课程的学时减少1，入学到毕业的最短时间能减少多少。 相邻两数之间用一个空格隔开。

样例输入
5
64 3 2 3 5
48 0
32 1 2
48 0
48 1 4

样例输出
160 1
48 0
80 0
48 1
96 1

样例解释
一种可能的解释：
1号课程为“数据结构”，64学时，前置课程为“程序设计基础” “面向对象程序设计基础”和“离散数学(2)”；
2号课程为“程序设计基础”，48学时，无前置课程；
3号课程为“面向对象程序设计基础”，32学时，前置课程为“程序设计基础”；
4号课程为“离散数学(1)”，48学时，无前置课程；
5号课程为“离散数学(2)”，48学时，前置课程为“离散数学(1)”。
可以证明，数据结构一定是最后学，因此它对毕业最短时间有影响；而在离散数学(1)(2)和两门程序设计课程中，离散数学需要的学时更多，从而只有减少它们的学时可以减少毕业所需时间。

数据规模与约定
对于30%的数据，1≤n≤100；
对于70%的数据，1≤n≤1000；
对于100%的数据，1≤n≤100000，∑ci≤10n
时间限制：1s
空间限制：256MB
*/

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

const int MAX_n = 100000;
const int MAX_ti = 10000;

struct Course
{
    int t, c;
    int max;
    int *pre; //prerequisites
    int *post; //courses that need it as pre
};

int ffmax(int a, int b) {
    if (a >= b) {
        return a;
    } else {
        return b;
    }
}


int find_cmax(int id, struct Course *courses) {
    int j, cmax = 0;

    if (courses[id].c == 0) {
        courses[id].max = courses[id].t;
        return courses[id].max;
    }

    //loops all the prereqs
    for (j = 0; j < courses[id].c; j++) {
        if (courses[id].max == 0) {
            cmax = ffmax(cmax, courses[id].t + find_cmax(courses[id].pre[j], courses));
        } else {
            cmax = ffmax(cmax, courses[id].max);
        }
    }
    courses[id].max = cmax;
    return cmax;
}



// main to preprocess the input
int main(void){
    struct Course courses[MAX_n];
    // struct Course course_test;

    // read the first line
    char str_n[7];
    int n, i, j;
    int tempc;
    scanf("%s", str_n);
    n = atoi(str_n);

    for (i=0; i<n; i++) {
        // ti = ci = 0;
        scanf("%d %d", &courses[i].t, &courses[i].c);
        courses[i].max = 0;
        courses[i].pre = (int*) malloc(courses[i].c * sizeof(int));

        for (j=0; j<courses[i].c; j++) {
            scanf("%d", &tempc);
            courses[i].pre[j] = tempc-1;
        }
    }

    
    int max_time[n];
    int max_class = 0;
    for (i=0; i<n; i++) {
        max_time[i] = find_cmax(i, courses);
        max_class = ffmax(max_class, max_time[i]);
    }
    
    // printf("%d", max_class);

    int c_path[n];
    int max_time_c[n];
    int max_class_c = 0;
    
    for (i=0; i<n; i++) {
        max_class_c = 0;
        c_path[i] = 0;
        courses[i].t++;

        for (j=0; j<n; j++) {
            courses[j].max = 0;
        }
        
        for (j=0; j<n; j++) {
            max_time_c[j] = find_cmax(j, courses);
            max_class_c = ffmax(max_class_c, max_time_c[j]);
        }

        if (max_class_c != max_class) {
            c_path[i] = 1;
        }
        
        courses[i].t--;
        // printf("%d %d\n", max_class, max_class_c);
    }

    for (i=0; i<n; i++) {
        printf("%d %d\n", max_time[i], c_path[i]);
    }


    for (i=0; i<n; i++) {
        free(courses[i].pre);
    }
    
    return 0;
}