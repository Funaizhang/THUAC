//  
//  Lab2a.c
//  
//  Created by Zhang Naifu 2018280351 on 23/05/2019.
//
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef NOMINMAX
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
#endif  /* NOMINMAX */

const int MAX = 1000000;

struct Vertex {
    int x, y;
};

struct Edge {
    int src, dest, weight;
};

int distance(struct Vertex vertex1, struct Vertex vertex2) {
    return pow(vertex1.x - vertex2.x, 2) + pow(vertex1.y - vertex2.y, 2);
};

int find(int i, int parent[]) {
    // int parenti;
	while (parent[i] != -1)
	i = parent[i];
	return i;
};

int uni(int i, int j, int parent[]) {
	if (i!=j) {
		parent[j] = i;
		return 1;
	}
	return 0;
};

// main to preprocess the input
int main(){
    // read the first line
    char str_n[5], str_q[5];
    int n, q;
    int x, y;
    int s, t;
    int h=0, i=0, j=0, k=0;
    int a, b, u, v, newEdge=1;
    int min, mincost=0;

    // read first line of input
    scanf("%s", str_n);
    n = atoi(str_n);

    int cost[n][n];
    int fw[n][n];
    int max_cost[n][n];
    int parents[n];
    int edgeWeights[n-1];
    // int cost = (int*) malloc(n * n * sizeof(int));

    // read coordinates of vertices
    struct Vertex vertices[n];
    for (i=0; i<n; i++) {
        scanf("%d %d", &x, &y);
        vertices[i].x = x;
        vertices[i].y = y;
    }

    // read queries
    scanf("%s", str_q);
    q = atoi(str_q);

    // read queries
    struct Edge queries[q];
    for (i = 0; i<q; i++) {
        scanf("%d %d", &s, &t);
        queries[i].src = s-1;
        queries[i].dest = t-1;
    }

    // // find number of edges * 2, because edges are undirected
    // unsigned int m;
    // m = n * (n-1)/2;

    // // calc all the edges
    // struct Edge edges[m];
    // for (j=0; j<n; j++) {
    //     for (k=j+1; k<n; k++){
    //         if (j==k) {
    //             continue;
    //         } else {
    //             edges[h].src = j;
    //             edges[h].dest = k;
    //             edges[h].weight = distance(vertices[j], vertices[k]);
    //             h++;
    //         }
    //     }
    // }

    // init parents array
    for (j=0; j<n; j++) {
        parents[j] = -1;
    }

    // make cost matrix
    for (j=0; j<n; j++) {
        for (k=0; k<n; k++) {
            cost[j][k] = distance(vertices[j], vertices[k]);
            // printf("from v%d to v%d -- weight%d \n", j, k, cost[j][k]);
        }
    }

    // for (j=0; j<n; j++) {
    //     for (k=0; k<n; k++) {
    //         printf("%d ", cost[j][k]);
    //     }
    //     printf("\n");
    // }

    // use Floyd-Warshall algo
    for (j=0; j<n; j++) {
        for (k=0; k<n; k++) {
            if (j==k) {
                fw[j][k] = 0;
            } else {
                fw[j][k] = MAX;
            }
            max_cost[j][k] = 0;
        }
    }

    while (newEdge < n) {
        for (j=0, min=MAX; j<n; j++) {
			for (k=j; k<n; k++) {
				if (cost[j][k] < min) {
					min = cost[j][k];
					a=u=j;
					b=v=k;
                    // printf("a=u=j= %d\t min= %d\n", a, min);
                    // printf("b=v=k= %d\t min= %d\n", b, min);
				}
			}
		}
        // printf("u=%d\t find(u, parents)=%d\n", u, find(u, parents));
        // printf("v=%d\t find(v, parents)=%d\n", v, find(v, parents));
        u = find(u, parents);
		v = find(v, parents);

        // printf("Parents:\t");
        // for (i = 0; i<n; i++) {
        //     printf("%d ", parents[i]);
        // }
        // printf("\n");

		if (uni(u, v, parents)) {
			// printf("%dth edge (%d,%d)=%d\n", newEdge++, a, b, min);
            newEdge++;
            fw[a][b] = fw[b][a] = min;
            max_cost[a][b] = max_cost[b][a] = min;
			// mincost += min;
		}
		cost[a][b] = cost[b][a] = MAX;
	}
	// printf("Minimum cost = %d\n", mincost);


    // // print fw
    // printf("Printing fw table...\n");
    // for (i=0; i<n; i++) {
    //     for (j=0; j<n; j++) {
    //         printf("%d ", fw[i][j]);
    //     }
    //     printf("\n");
    // }

    // // print max_cost
    // printf("Printing max_cost table...\n");
    // for (int i=0; i<n; i++) {
    //     for (int j=0; j<n; j++) {
    //         printf("%d ", max_cost[i][j]);
    //     }
    //     printf("\n");
    // }

    // implement Floyd-Warshall in place
    for (k=0; k<n; k++) {
        for (i=0; i<n; i++) {
            for (j=0; j<n; j++) {
                if (fw[i][j] > fw[i][k] + fw[k][j]) {
                    fw[i][j] = fw[i][k] + fw[k][j];
                    if (max_cost[i][j] < max(max_cost[i][k], max_cost[k][j])) {
                        max_cost[i][j] = max(max_cost[i][k], max_cost[k][j]);
                    }
                }

            }
        }
    }

    // // print fw
    // printf("Printing fw table...\n");
    // for (i=0; i<n; i++) {
    //     for (j=0; j<n; j++) {
    //         printf("%d ", fw[i][j]);
    //     }
    //     printf("\n");
    // }

    // // print max_cost
    // printf("Printing max_cost table...\n");
    // for (i=0; i<n; i++) {
    //     for (j=0; j<n; j++) {
    //         printf("%d ", max_cost[i][j]);
    //     }
    //     printf("\n");
    // }

    for (i=0; i<q; i++) {
        s = queries[i].src;
        t = queries[i].dest;
        printf("%d\n", max_cost[s][t]);
    }

    // for (i = 0; i<m; i++) {
    //     printf("Edge %d: %d %d %d\n", i, edges[i].src, edges[i].dest, edges[i].weight);
    // }
    
    return 0;

}