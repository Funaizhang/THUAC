//  
//  Lab1a.c
//  
//  Created by Zhang Naifu 2018280351 on 01/04/2019.
//
//
/*
设施选址(1)

问题背景
在城市规划中，经常会遇到这样的问题：如果要在城市的中建一些设施，那么这些设施建在哪里是比较好的？举个例子，如果在一个城市要建五家医院，那么这五家医院要分别如何选址，才能使得从城市的任何一个位置到离其最近的医院都不会太远。也就是说，城市的某个地方突然出现了一位突发疾病的病人，那么最近的医院就能立即派救护车前往以最短的时间接到这位病人。 同样的问题存在于其他场地设计中：如何在一个游乐园建若干个卫生间，使得游客在任何地方找到最近的卫生间都尽可能得快？如何在学校中建一些食堂，使得同学在任何教学楼上课都不会离食堂太远？……

问题描述
给定一个城市的道路分布，以及通过每条道路的时间。现在想在城市中建立一个医院，该医院必须建在道路相交的路口处。请为医院选择一个位置，使得城市的任意一个路口到医院的最大时间最小。即让该城市中到医院时间最长的路口，到医院的时间尽可能得小。

输入格式
从标准输入读入数据。
输入的第一行包含两个整数 n, m，分别表示城市的路口数、道路数。同一行中相邻两数之间用一个空格隔开，下同。
接下来 m 行，每行包含三个整数 u,v,t(1≤u,v≤n,1≤t≤10000)，表示一条从路口 u 到路口 v 的双向道路，且完整经过这条道路所需的时间为 t。
我们保证，任意两个路口之间最多存在一条直接相连的道路。

输出格式
输出到标准输出。
输出仅包含一个数，表示城市的任意一个路口到医院的最大时间的最小值。

样例输入
4 4
1 2 35
3 2 15
2 4 18
3 4 30

样例输出
35

样例解释
一种可行的最佳方案是，在2号路口建立医院。

数据规模与约定
对于30%的数据，1≤n≤10；
对于70%的数据，1≤n≤50；
对于100%的数据，1≤n≤200，n−1≤m≤n^2，保证城市的任意两个路口之间存在一条路径。
时间限制：1s
空间限制：256MB
*/

#include <stdio.h>
#include <stdlib.h>

const int MAX = 1000000;

// int floyd_warshall (int *w) {
//     return 0;
// }

// returns the max value of an array of arbitrary size
// int array_max ()

// main to preprocess the input
int main(){
    // read the first line
    char str_n[5], str_m[5];
    int n, m;

    scanf("%s %s", str_n, str_m);
    n = atoi(str_n);
    m = atoi(str_m);

    int w_matrix[n][n];
    int row_max[n];
    int final_max = MAX;

    int i, j, k;
    char str_u[5], str_v[5], str_w[5];
    int u, v, t = 0;

    // make a matrix with the weights
    for (i = 0; i<n; i++) {
        for (j = 0; j<n; j++) {
            if (i == j) {
                w_matrix[i][j] = 0;
            } else {
                w_matrix[i][j] = MAX;
            }
        }
    }

    // read the subsequent m lines
    // each line represents edge from u to v, weight t
    for (i=0; i<m; i++) {
        scanf("%d %d %d", &u, &v, &t);
        w_matrix[u-1][v-1] = t;
        w_matrix[v-1][u-1] = t;
    }

    // implement Floyd-Warshall in place
    for (k=0; k<n; k++) {
        for (i=0; i<n; i++) {
            for (j=0; j<n; j++) {
                if (w_matrix[i][j] > w_matrix[i][k] + w_matrix[k][j]) w_matrix[i][j] = w_matrix[i][k] + w_matrix[k][j];
            }
        }
    }

    // find the max of each row
    for (i=0; i<n; i++) {
        row_max[i] = 0;
        for (j=0; j<n; j++) {
            if (w_matrix[i][j] > row_max[i]) row_max[i] = w_matrix[i][j];
        }
    }
    // find min wrt all vertices
    for (i=0; i<n; i++) {
        if (row_max[i] < final_max) final_max = row_max[i];
    }

    printf("%d\n", final_max);


    // // print the table
    // for (int i=0; i<n; i++) {
    //     for (int j=0; j<n; j++) {
    //         printf("%d ", w_matrix[i][j]);
    //     }
    //     printf("\n");
    // }

    // for (int i=0; i<n; i++) {
    //     printf("%d ", row_max[i]);
    // }
    // printf("\n");

    return 0;
}