//  
//  Lab2b.cpp
//  
//  Created by Zhang Naifu 2018280351 on 25/05/2019.
//  Reference acknowledgement: https://www.geeksforgeeks.org/
//
/*
B. 色数

问题描述
给定一个无向图，现在要将图上的所有结点划分为若干个集合，使得：
每个结点恰好属于一个集合；
对于无向图中的任意一条边 (u, v)，结点 u 和结点 v 在不同集合中。
对于所有满足以上性质的划分，我们称集合个数的最小值为这张无向图的色数。
现在给定一个无向图，请求出其色数。

输入格式
从标准输入读入数据。
第一行两个整数 n,m，表示该无向图有 n 个结点，m 条边。
接下来 m 行，每行两个整数 u,v，表示无向图的一条边。
我们保证：每条无向边只会被描述一次（即不存在“重边”）；不存在一条 u=v 的边（即不存在“自环”）。

输出格式
输出到标准输出。
输出仅一行，即为该无向图的色数。

样例输入
6 6
1 2
2 3
3 4
4 5
5 1
1 6

样例输出
3

数据规模与约定
对于 30% 的数据，n≤5。
对于另外 10% 的数据，m=0。
对于另外 10% 的数据，该图是完全图。
对于另外 10% 的数据，该图为完全图删去其中一条边。
对于另外 10% 的数据，该图为二分图。
对于另外 10% 的数据，该图为一棵树。
对于 100% 的数据，1≤n≤10,0≤m≤n(n−1)/2。

时间限制：1s
空间限制：256MB
提示
[考虑图中任意两个没有边相连的结点，讨论它们是否属于相同集合。如果是，则这两个结点可以合并；如果不是，则可以在图中加上这条边。]
*/

#include <iostream>
#include <list>
using namespace std;
bool graph[10][10];

// // A graph using adjacency list representation 
// class Graph { 
//     int V; // No. of vertices in Graph 
//     list<int> *adj; // Pointer to an array containing adjacency lists 
// public: 
//     Graph(int V)   { this->V = V; adj = new list<int>[V]; } // Constructor 
//     void addEdge(int u, int v); 
//     bool isSafe(int v, int vertexColor[], int c);
//     // Prints greedy coloring of the vertices 
//     int chromaticNumber(); 
// }; 
  
// void Graph::addEdge(int u, int v) { 
//     adj[u].push_back(v); // Add v to u’s list. 
//     adj[v].push_back(u); // Add u to v’s list. 
// }


// // decide if vertex can be colored with proposed color
// bool Graph::isSafe (int v, int colorList[], int c) {
//     list<int>::iterator i; 
//     for (i = adj[v].begin(); i != adj[v].end(); ++i) {
//         if (c == colorList[*i]) {
//             return false;
//         }
//     }
//     return true;
// }

bool isSafe(int v, int n, int colorList[], int c) {
    for (int i = 0; i < n; i++) {
        if (!graph[v][i]) {
            continue;
        } else if (c == colorList[i]) {
            return false;
        }
    }
    return true;
}

bool chromaticNumber(int v, int n, int max_c, int colorList[]) {
    // all vertices are colored
    if (v == n) {
        return true;
    }

    for (int c=1; c<=max_c; c++) {
        if (isSafe(v, n, colorList, c)) {
            colorList[v] = c;
            if (chromaticNumber(v+1, n, max_c, colorList)) {
                return true;
            }
            // reset color[v], as this c_max does not work
            colorList[v] = 0;
        }
    }
    // found no solution
    return false; 
}

// Main function
int main() { 
    int n, m, complete_m;
    int c, u, v;
    int i=0;

    // read n
    cin >> n >> m;
    complete_m = n*(n-1)/2;

    if (m == 0) { // empty graph
        c = 1;
    } else if (m == complete_m) { // complete graph
        c = n;
    } else if (m == complete_m - 1) { // complete graph - e
        c = n-1;
    } else {

        // // init graph
        // Graph g(n);

        // // read vertices
        // for (i=0; i<n; i++) {
        //     cin >> u >> v;
        //     g.addEdge(u-1, v-1);
        // }

        // // calculate chromatic number
        // c = g.chromaticNumber();

        // init graph matrix
        for (int i=0; i<n; i++) {
            for (int j=i; j<n; j++) {
                graph[i][j] = graph[j][i] = 0;
            }
        }
        // read vertices
        for (int i=0; i<m; i++) {
            cin >> u >> v;
            graph[u-1][v-1] = graph[v-1][u-1] = 1;
        }

        // init colorList
        int colorList[n];
        for (int i=0; i<n; i++) {
            colorList[i] = 0;
        }

        for (int max_color=2; max_color<=10; max_color++) {
            if (chromaticNumber(0, n, max_color, colorList) == true) {
                c = max_color;
                break;
            }
            // c = 10;
        }
    }
    cout << c << endl; 
    return 0; 
} 