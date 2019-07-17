//  
//  Lab2a.cpp
//  
//  Created by Zhang Naifu 2018280351 on 23/05/2019.
//  Reference acknowledgement: https://www.geeksforgeeks.org/
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

样例输出
1
1
2
2

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

#include <math.h> 
#include <iostream>
#include <list>
using namespace std;

const int MAX = 1000000;
int cost[5000][5000];

// a structure to represent a vertex in graph 
struct Vertex {
    int x, y;
};

// a structure to represent a weighted edge in graph 
struct Edge { 
    int src, dest, weight; 
}; 
  
// A structure to represent a subset for union-find 
struct subset { 
    int parent; 
    int rank; 
}; 

// a structure to represent a connected, undirected and weighted graph 
struct Graph { 
    // V-> Number of vertices, E-> Number of edges 
    int V, E; 
  
    // graph is represented as an array of edges.  
    // Since the graph is undirected, the edge 
    // from src to dest is also edge from dest 
    // to src. Both are counted as 1 edge here. 
    struct Edge* edge; 
}; 
  
// Creates a graph with V vertices and E edges 
struct Graph* createGraph(int V, int E) { 
    struct Graph* graph = new Graph; 
    graph->V = V;
    graph->E = E; 
  
    graph->edge = new Edge[E]; 
  
    return graph; 
} 
  
// A utility function to find set of an element i 
// (uses path compression technique) 
int find(struct subset subsets[], int i) { 
    // find root and make root as parent of i  
    // (path compression) 
    if (subsets[i].parent != i) 
        subsets[i].parent = find(subsets, subsets[i].parent); 
  
    return subsets[i].parent; 
} 
  
// A function that does union of two sets of x and y 
// (uses union by rank) 
void Union(struct subset subsets[], int x, int y) { 
    int xroot = find(subsets, x); 
    int yroot = find(subsets, y); 
  
    // Attach smaller rank tree under root of high  
    // rank tree (Union by Rank) 
    if (subsets[xroot].rank < subsets[yroot].rank) 
        subsets[xroot].parent = yroot; 
    else if (subsets[xroot].rank > subsets[yroot].rank) 
        subsets[yroot].parent = xroot; 
  
    // If ranks are same, then make one as root and  
    // increment its rank by one 
    else
    { 
        subsets[yroot].parent = xroot; 
        subsets[xroot].rank++; 
    } 
} 
  
// Compare two edges according to their weights. 
// Used in qsort() for sorting an array of edges 
int myComp(const void* a, const void* b) { 
    struct Edge* a1 = (struct Edge*)a; 
    struct Edge* b1 = (struct Edge*)b; 
    return a1->weight > b1->weight; 
};

int distance(struct Vertex vertex1, struct Vertex vertex2) {
    return pow(vertex1.x - vertex2.x, 2) + pow(vertex1.y - vertex2.y, 2);
};  



// A directed MST using adjacency list representation 
class MST { 
    int V; // No. of vertices in MST 
    list<int> *adj; // Pointer to an array containing adjacency lists 
public: 
    MST(int V)  { this->V = V; adj = new list<int>[V]; } // Constructor 
    void addEdge(int u, int v); 
    void printAllPaths(int s, int d); 
    // A recursive function used by printAllPaths() 
    void printAllPathsUtil(int s, int d, bool visited[], int path[], int &path_index); 
}; 
  
void MST::addEdge(int u, int v) { 
    adj[u].push_back(v); // Add v to u’s list. 
    adj[v].push_back(u); // Add u to v’s list. 
} 
  
// Prints all paths from 's' to 't' 
void MST::printAllPaths(int s, int t) { 

    // Mark all the vertices as not visited 
    bool *visited = new bool[V]; 
  
    // Create an array to store paths 
    int *path = new int[V]; 
    int path_index = 0; // Initialize path[] as empty 
  
    // Initialize all vertices as not visited 
    for (int i = 0; i < V; i++) 
        visited[i] = false; 
  
    // Call the recursive helper function to print all paths 
    printAllPathsUtil(s, t, visited, path, path_index); 
} 
  
// A recursive function to print all paths from 'u' to 'd'. 
// visited[] keeps track of vertices in current path. 
// path[] stores actual vertices and path_index is current 
// index in path[] 
void MST::printAllPathsUtil(int u, int d, bool visited[], int path[], int &path_index) { 
    // Mark the current node and store it in path[] 
    visited[u] = true; 
    path[path_index] = u; 
    path_index++; 
  
    // If current vertex is same as destination, then print current path[] 
    if (u == d) { 
        // keep track of lowest capacity
        int c_max = 0;

        for (int i = 0; i+1<path_index; i++) 
            if (cost[path[i]][path[i+1]] > c_max)
                c_max = cost[path[i]][path[i+1]];
            // cout << path[i] << " ";
        cout << c_max << endl; 
        return;
    } else { // If current vertex is not destination
        // Recur for all the vertices adjacent to current vertex 
        list<int>::iterator i; 
        for (i = adj[u].begin(); i != adj[u].end(); ++i) 
            if (!visited[*i]) 
                printAllPathsUtil(*i, d, visited, path, path_index); 
    } 
  
    // Remove current vertex from path[] and mark it as unvisited 
    path_index--; 
    visited[u] = false; 
} 



// The main function to construct MST using Kruskal's algorithm 
void kruskal(struct Graph* graph, int q, int start[], int end[]) { 
    int V = graph->V;  
    struct Edge result[V];  // This will store the resultant MST 
    int e = 0;  // An index variable, used for result[] 
    int i = 0;  // An index variable, used for sorted edges 
    int s, t;
  
    // Step 1:  Sort all the edges in non-decreasing  
    // order of their weight. If we are not allowed to  
    // change the given graph, we can create a copy of 
    // array of edges 
    qsort(graph->edge, graph->E, sizeof(graph->edge[0]), myComp); 
  
    // Allocate memory for creating V ssubsets 
    struct subset *subsets = (struct subset*) malloc( V * sizeof(struct subset)); 
  
    // Create V subsets with single elements 
    for (int v = 0; v < V; ++v) { 
        subsets[v].parent = v; 
        subsets[v].rank = 0; 
    } 
  
    // Number of edges to be taken is equal to V-1 
    while (e < V - 1) { 
        // Step 2: Pick the smallest edge. And increment  
        // the index for next iteration 
        struct Edge next_edge = graph->edge[i++]; 
  
        int x = find(subsets, next_edge.src); 
        int y = find(subsets, next_edge.dest); 
  
        // If including this edge does't cause cycle, 
        // include it in result and increment the index  
        // of result for next edge 
        if (x != y) 
        { 
            result[e++] = next_edge;
            Union(subsets, x, y);
        } 
        // Else discard the next_edge 
    } 
    
    // build MST for search
    MST g(V);
    // printf("Following are the edges in the constructed MST\n"); 
    for (i = 0; i < e; ++i) 
        g.addEdge(result[i].src, result[i].dest);
        // printf("%d -- %d == %d\n", result[i].src, result[i].dest, result[i].weight); 
    
    for (i=0; i<q; i++) {
        s = start[i];
        t = end[i]; 
        g.printAllPaths(s, t); 
    }
    
    return;
} 



int main() { 
    int n, m, p=0, q;
    int x, y, s, t;
    int h=0, i=0, j=0, k=0;

    // read n
    cin >> n;
    m = n * (n-1)/2;

    struct Vertex vertices[n];
    struct Graph* graph = createGraph(n, m); 

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

    // read queries
    for (i=0; i<n; i++) {
        cin >> s >> t;
        start[i] = s-1;
        end[i] = t-1;
    }

    // add edges to kruskal complete graph
    for (i=0; i<n; i++) {
        for (j=i+1; j<n; j++) {
            // add edges
            graph->edge[p].src = i; 
            graph->edge[p].dest = j; 
            graph->edge[p].weight = distance(vertices[i], vertices[j]); 
            // construct the cost matrix
            cost[i][j] = distance(vertices[i], vertices[j]);
            cost[j][i] = distance(vertices[i], vertices[j]);
            p++;
        }
    }
  
    kruskal(graph, q, start, end); 
  
    return 0; 
} 