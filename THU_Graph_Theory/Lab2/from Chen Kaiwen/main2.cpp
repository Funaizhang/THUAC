#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;
#define MAX 10
#define INT_MAX 2147483647
#define MIN(a, b) ((a) > (b) ? (b) : (a))

int ans = INT_MAX;
int N;
void hardCode(int id, int num, int **g, int **c, int *cnt)
{
    if (num >= ans)
    {
        return;
    }
    if (id > N)
    {
        ans = MIN(ans, num);
        return;
    }
    for (int i = 1; i <= num; i++)
    {
        int s = cnt[i];
        int js = 0;
        for (int j = 1; j <= s; j++)
        {
            if (g[id][c[i][j]] == 0)
            {
                js++;
            }
        }
        if (js == s)
        {
            c[i][++cnt[i]] = id;
            hardCode(id + 1, num, g, c, cnt);
            cnt[i]--;
        }
    }
    c[num + 1][++cnt[num + 1]] = id;
    hardCode(id + 1, num + 1, g, c, cnt);
    --cnt[num + 1];
}

int main()
{
    int M;
    cin >> N >> M;
    int **g = new int *[N + 1];
    for (int i = 0; i < N + 1; i++)
        g[i] = new int[N + 1];
    for (int i = 0; i < N + 1; i++)
        for (int j = 0; j < N + 1; j++)
            g[i][j] = 0;
    int **c = new int *[N + 1];
    for (int i = 0; i < N + 1; i++)
        c[i] = new int[N + 1];
    for (int i = 0; i < N + 1; i++)
        for (int j = 0; j < N + 1; j++)
            c[i][j] = 0;
    int *cnt = new int[N + 1];
    for (int i = 0; i < N + 1; i++)
        cnt[i] = 0;
    for (int i = 0; i < M; i++)
    {
        int a, b;
        cin >> a >> b;
        g[a][b] = g[b][a] = 1;
    }
    if (M == 0) //no edges = 1 color
    {
        printf("1\n");
        return 0;
    }
    else if (M == N * (N - 1) / 2) //complete graph = N clr
    {
        printf("%d\n", N);
        return 0;
    }
    else if (M + 1 == N * (N - 1) / 2) //complete graph - 1 = N - 1 clr
    {
        printf("%d\n", N - 1);
        return 0;
    }
    else if (N <= 10)
    {
        hardCode(1, 0, g, c, cnt);
    }
    cout << ans;
    return 0;
}