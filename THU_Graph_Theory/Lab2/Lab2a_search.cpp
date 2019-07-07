// C++ program to print all paths from a source to destination. 
#include <iostream> 
#include <list> 
#include <vector>
using namespace std; 
  
// A directed MST using adjacency list representation 
class MST { 
    int V; // No. of vertices in MST 
    list<int> *adj; // Pointer to an array containing adjacency lists 
  
  
public: 
    MST(int V); // Constructor 
    void addEdge(int u, int v); 
    void printAllPaths(int s, int d, vector<vector<int> > &cost); 
    // A recursive function used by printAllPaths() 
    void printAllPathsUtil(int s, int d, bool visited[], int path[], int &path_index, vector<vector<int> > &cost); 
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
void MST::printAllPaths(int s, int d, vector<vector<int> > &cost) { 

    // Mark all the vertices as not visited 
    bool *visited = new bool[V]; 
  
    // Create an array to store paths 
    int *path = new int[V]; 
    int path_index = 0; // Initialize path[] as empty 
  
    // Initialize all vertices as not visited 
    for (int i = 0; i < V; i++) 
        visited[i] = false; 
  
    // Call the recursive helper function to print all paths 
    printAllPathsUtil(s, d, visited, path, path_index, cost); 
} 
  
// A recursive function to print all paths from 'u' to 'd'. 
// visited[] keeps track of vertices in current path. 
// path[] stores actual vertices and path_index is current 
// index in path[] 
void MST::printAllPathsUtil(int u, int d, bool visited[], int path[], int &path_index, vector<vector<int> > &cost) { 
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
    } else { // If current vertex is not destination
        // Recur for all the vertices adjacent to current vertex 
        list<int>::iterator i; 
        for (i = adj[u].begin(); i != adj[u].end(); ++i) 
            if (!visited[*i]) 
                printAllPathsUtil(*i, d, visited, path, path_index, cost); 
    } 
  
    // Remove current vertex from path[] and mark it as unvisited 
    path_index--; 
    visited[u] = false; 
} 
  
// Driver program 
int main() { 
    int i, j, k;
    int n=4;
    vector<vector<int> > cost(n, vector<int>(n));

    // Create a MST given in the above diagram 
    MST g(n); 
    g.addEdge(0, 1); 
    g.addEdge(0, 2); 
    g.addEdge(1, 3); 
    
    for (i=0; i<n; i++){
        for (j=0; j<n; j++) {
            cost[i][j] = i*n+j;
        }
    }

    for (j=0; j<n; j++) {
        for (k=0; k<n; k++) {
            printf("%d ", cost[j][k]);
        }
        printf("\n");
    }

    int s = 2, d = 3; 
    g.printAllPaths(s, d, cost); 
  
    return 0; 
} 