#include <stdio.h>
#include <cstring>
#include <vector>
#include <queue>
#include <utility>
#define INT_MAX 2147483647
#define MAX_N 1005
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
using namespace std;

int N;
vector<int> graph[MAX_N];
int in[MAX_N], ans[MAX_N], ac[MAX_N][MAX_N], mf[MAX_N][MAX_N], path[MAX_N], flow[MAX_N];

void cov(int i, int j)
{
    for (int k = 1; k <= N; k++)
        ac[j][k] = ac[i][k];
}

void uni(int i, int j)
{
    for (int k = 1; k <= N; k++)
        ac[j][k] = ac[i][k] | ac[j][k];
}

int longest()
{
    queue<int> q;
    for (int i = 1; i <= N; i++)
    {
        if (in[i] == 0)
        {
            q.push(i);
            ans[i] = 1;
            ac[i][i] = 1;
            mf[0][i] = 1;
        }
    }
    while (!q.empty())
    {
        int work = q.front();
        q.pop();
        for (int i = 0; i < graph[work].size(); i++)
        {
            int j = graph[work][i];
            in[j]--;
            if (ans[j] < ans[work] + 1)
            {
                ans[j] = ans[work] + 1;
                cov(work, j);
            }
            else
                uni(work, j);
            if (in[j] == 0)
                q.push(j);
        }
    }

    int res = -1;
    for (int i = 1; i <= N; i++)
        res = MAX(res, ans[i]);
    return res;
}

int bfs(int s, int t, queue<int> *q)
{
    while (!q->empty())
        q->pop();
    memset(path, -1, sizeof(path));
    flow[s] = INT_MAX;
    q->push(s);
    while (!q->empty())
    {
        int cur = q->front();
        q->pop();
        if (cur == t)
            break;
        for (int i = 1; i <= N + 1; i++)
        {
            if (path[i] == -1 && mf[cur][i] == 1)
            {
                flow[i] = MIN(flow[cur], mf[cur][i]);
                path[i] = cur;
                q->push(i);
            }
        }
    }
    if (path[t] == -1)
        return -1;
    return flow[N + 1];
}

int disjoints(int l, int s, int t)
{
    queue<int> q;
    for (int i = 1; i <= N; i++)
    {
        if (ans[i] == l)
        {
            mf[i][N + 1] = 1;
            for (int j = 1; j <= N; j++)
                if (ac[i][j])
                    mf[j][i] = 1;
        }
    }
    int maxFlow = 0, pre, now, stepin;
    while ((stepin = bfs(s, t, &q)) != -1)
    {
        maxFlow += stepin;
        now = t;
        while (now != s)
        {
            pre = path[now];
            mf[pre][now] -= stepin;
            mf[now][pre] += stepin;
            now = pre;
        }
    }
    return maxFlow;
}

int main()
{
    scanf("%d", &N);
    for (int i = 1; i <= N; i++)
    {
        int num;
        scanf("%d", &num);
        in[i] = num;
        for (int j = 1; j <= num; j++)
        {
            int pre;
            scanf("%d", &pre);
            graph[pre].push_back(i);
        }
    }
    int l = longest();
    printf("%d %d\n", l, disjoints(l, 0, N + 1));
    return 0;
}