//  
//  Lab2a.cpp
//  
//  Created by Zhang Naifu 2018280351 on 23/05/2019.
//  Reference acknowledgement: 
//  https://www.geeksforgeeks.org/find-paths-given-source-destination/
//  https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
//
/*
A. 网络

问题描述
在计算机网络中，路由是一个核心问题——它决定了一个数据包在网络上的转发路径和带宽。
我们考虑一个虚拟的计算机网络。在这个网络中，共有 nn 台计算机，它们分别位于平面上的 (xi, yi) (1≤i≤n) 处。 任意两台计算机之间都有直接的链路相连。即，这是一个“全联通”的网络。
每条“直接链路”上都有一定的带宽，为 1 / (这两台计算机的欧几里得距离平方)。
当两台计算机进行通信时，它们会选择一条路径（计算机的非空序列），并通过这条路径依次转发数据。而这条路径的带宽，即为路径上各链路带宽的最小值。
形式化地说，计算机 s 和 t 在通信时会选择一个非空序列 {l1,l2,⋯,lk}，满足 l1=s,lk=t。此时，这次通信的带宽为 min1≤i<k{1(xli−xli+1)2+(yli−yli+1)2}
现在我们想知道，对于编号为 s 的计算机和编号为 t 的计算机，它们进行通信时，最大可能的带宽是多少。
为了避免精度误差，你只需输出答案的倒数。

输入格式
从标准输入读入数据。
第一行一个整数 n，表示有 n 台计算机。
接下来 n 行，第 i 行两个整数 xi,yi, 表示第 i 台计算机的平面坐标。
接下来一行一个整数 q，表示有 q 个询问。
接下来 q 行，每行两个整数 s,t，表示一个询问。
我们保证：不存在两台不同的计算机，满足 x 坐标和 y 坐标分别相等；不存在两个询问，满足 s 和 t 分别相等。

输出格式
输出到标准输出。
输出 q 行。对于每个询问输出一行，表示对于该询问，路径的最大可能带宽的倒数。

样例输入
4
0 0
1 0
0 1
2 1
4
1 2
2 3
4 3
1 4

5
3 2
5 5
1 6
1 4
6 9
1 
3 2


样例输出
1
1
2
2

13


样例解释
对于第1个询问，一条可能的路径是 1 → 2。
对于第2个询问，一条可能的路径是 2 → 1 → 3。
对于第3个询问，一条可能的路径是 4 → 2 → 3。
对于第4个询问，一条可能的路径是 1 → 2 → 4。

数据规模与约定
对于 50% 的数据，n,q≤100。
对于 80% 的数据，n,q≤1000。
对于 100% 的数据，1≤n,q≤5000,0≤xi,yi<10000,1≤s,t≤n。

时间限制：1s
空间限制：256MB

提示
[可以先求出网络的最大生成树，并考虑生成树上的路径与答案的关系。]
*/

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <iostream>
#include <list>
#include <limits.h> 
#include <stdbool.h> 
using namespace std;

#ifndef NOMINMAX
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
#endif  /* NOMINMAX */

const int MAX = 100000000;
int graph[5000][5000];
int capacity[5000][5000];

// a structure to represent a vertex in graph 
struct Vertex {
    int x, y;
};

// a structure to represent a weighted edge in graph 
struct Edge { 
    int src, dest, weight; 
}; 

int distance(struct Vertex vertex1, struct Vertex vertex2) {
    return pow(vertex1.x - vertex2.x, 2) + pow(vertex1.y - vertex2.y, 2);
};

// A utility function to find the vertex with min key, 
// from the set of vertices not yet included in MST 
int minKey(int key[], bool mstSet[], int V) {
    // init vars
    int minimum = MAX;
    int min_idx = 0;

    // vertex i has the smallest key so far
    for (int i=0; i<V; i++) {
        if (mstSet[i] == false && key[i] < minimum) {
            minimum = key[i];
            min_idx = i;
        }
    }
    return min_idx;
};

// A directed MST using adjacency list representation 
class MST { 
    int V; // No. of vertices in MST 
    list<int> *adj; // Pointer to an array containing adjacency lists 
public:
    MST(int V); // Constructor 
    void addEdge(int u, int v); 
    void getCapacity(int s, int d); 
    // A recursive function used by getCapacity() 
    void getCapacityRecur(int s, int d, bool visited[], int path[], int &path_index); 
}; 
  
MST::MST(int V) { 
    this->V = V; 
    adj = new list<int>[V]; 
} 
  
void MST::addEdge(int u, int v) { 
    adj[u].push_back(v); // Add v to u’s list. 
    adj[v].push_back(u); // Add u to v’s list. 
} 
  
// Prints all paths from 's' to 'd' 
void MST::getCapacity(int s, int d) { 
    // cout << "getCapacity: " << s << "-" << d << endl; 

    // Mark all the vertices as not visited 
    bool *visited = new bool[V]; 
  
    // Create an array to store paths and the cost along the path
    int *path = new int[V]; 
    int *c_path = new int[V]; 
    int path_index = 0; // Initialize path[] as empty 
  
    // Initialize all vertices as not visited 
    for (int i = 0; i < V; i++) {
        visited[i] = false; 
    }

    // cout << "Initial matrix" << endl;
    // for (int i=0; i<5; i++) {
    //     for (int j=0; j<5; j++)
    //         cout << capacity[i][j] << "\t";
    //     cout << endl;
    // }

    // Call the recursive helper function to print all paths 
    getCapacityRecur(s, d, visited, path, path_index); 

    // cout << "Final matrix" << endl;
    // for (int i=0; i<5; i++) {
    //     for (int j=0; j<5; j++)
    //         cout << capacity[i][j] << "\t";
    //     cout << endl;
    // }
} 
  
// A recursive function to print the path from 'u' to 'd' along the MST.
// visited[] keeps track of vertices in current path. 
// path[] stores actual vertices and path_index is current index in path[].
void MST::getCapacityRecur(int u, int d, bool visited[], int path[], int &path_index) { 
    // Mark the current node and store it in path[] 
    visited[u] = true; 
    path[path_index] = u;
    // cout << "Path advanced: [" << path_index << "]: ";
    // for (int i =0; i<= path_index; i++)
    //     cout << path[i]+1 << " ";
    // cout << endl;

    path_index++; 
  
    // Recur for all the vertices adjacent to current vertex 
    list<int>::iterator i; 
    for (i = adj[u].begin(); i != adj[u].end(); ++i) {
        cout << "TRY: " << u+1 << "-" << *i+1 << endl;

        if (!visited[*i]) {
            if (capacity[*i][d] == -1) {
                cout << "recurse: " << u+1 << "-" << *i+1 << "-" << d+1 << endl;
                getCapacityRecur(*i, d, visited, path, path_index);
            } else {
                visited[*i] = visited[d] = true;
                path[path_index] = *i;
                path[path_index+1] = d;
                
                // update capacity matrix along the path, like dp memoization
                for (int i = path_index-1; i>=0; i--) {
                    capacity[path[i]][d] = capacity[d][path[i]] = max(capacity[path[i]][path[i+1]], capacity[path[i+1]][d]);
                }

                int c_max = 0;
                cout << "Found path: " << path[0]+1;
                for (int i = 0; i<path_index+1; i++) {
                    cout << "-" << path[i+1]+1;
                    if (capacity[path[i]][path[i+1]] > c_max)
                        c_max = capacity[path[i]][path[i+1]];
                }
                cout << endl;
                cout << "c_max: " << c_max << endl;

                // cout << "Capacity matrix updated" << endl;
                // for (int i=0; i<5; i++) {
                //     for (int j=0; j<5; j++)
                //         cout << capacity[i][j] << "\t";
                //     cout << endl;
                // }
            }
        }
    }

    // Remove current vertex from path[] and mark it as unvisited 
    path_index--; 
    visited[u] = false;
} 


// Function to construct and print MST for  
// a graph represented using adjacency matrix 'graph'
void prim(int n, int q, int start[], int end[]) { 
    // Array to store constructed MST 
    int parent[n];  
    // Key values used to pick minimum weight edge in cut 
    int key[n];  
    // To represent set of vertices not yet included in MST 
    bool mstSet[n];  

    // Init s & t
    int s, t;
  
    // Initialize all keys as MAX 
    for (int i = 0; i < n; i++) 
        key[i] = MAX, mstSet[i] = false; 
  
    // Always include first 1st vertex in MST. 
    // Make key 0 so that this vertex is picked as first vertex. 
    key[0] = 0;      
    parent[0] = -1; // First node is always root of MST  
  
    // The MST will have n-1 vertices 
    for (int count = 0; count < n-1; count++) { 
        // Pick the minimum key vertex from the  
        // set of vertices not yet included in MST 
        int u = minKey(key, mstSet, n); 
  
        // Add the picked vertex to the MST Set 
        mstSet[u] = true; 
  
        // Update key value and parent index of  
        // the adjacent vertices of the picked vertex.  
        // Consider only those vertices which are not  
        // yet included in MST 
        for (int v = 0; v < n; v++) 
  
        // graph[u][v] is non zero only for adjacent vertices of m 
        // mstSet[v] is false for vertices not yet included in MST 
        // Update the key only if graph[u][v] is smaller than key[v] 
        if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v]) 
            parent[v] = u, key[v] = graph[u][v]; 
    } 

    // build MST for search
    MST g(n);

    for (int i = 1; i < n; ++i) {
        g.addEdge(parent[i], i);
        capacity[parent[i]][i] = capacity[i][parent[i]] = graph[i][parent[i]];
        // cout << "MST edge (" << parent[i]+1 << ", " << i+1 << ") weight=" << graph[i][parent[i]] << endl;
    }

    for (int i=0; i<q; i++) {
        s = start[i];
        t = end[i]; 
        g.getCapacity(s, t); 
    }
} 


int main() { 
    int n, m, p=0, q=0;
    int x, y, s, t;
    int h=0, i=0, j=0, k=0;

    // read n
    cin >> n;
    m = n * (n-1)/2;

    struct Vertex vertices[n];

    // read vertices
    for (i=0; i<n; i++) {
        cin >> x >> y;
        vertices[i].x = x;
        vertices[i].y = y;
    }

    // read q
    cin >> q;
    int start[q];
    int end[q];
    int wrong = 0;

    // read queries
    for (i=0; i<q; i++) {
        cin >> s >> t;
        if (s<= n && t<=n) {
            start[i] = s-1;
            end[i] = t-1;
        } else {
            wrong++;
        }
    }
    q -= wrong;

    // add edges to prim complete graph
    for (i=0; i<n; i++) {
        capacity[i][i] = 0;
        for (j=i+1; j<n; j++) {
            // Initialize the cost/capacity matrix
            graph[i][j] = distance(vertices[i], vertices[j]);
            graph[j][i] = distance(vertices[i], vertices[j]);
            capacity[i][j] = capacity[j][i] = -1;
        }
    }
  
    prim(n, q, start, end);

    return 0; 
} 