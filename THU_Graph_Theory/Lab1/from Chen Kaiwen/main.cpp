#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#define MIN(a, b) (a) > (b) ? b : a
#define MAX(a, b) (a) > (b) ? a : b
//#define INT_MAX 2147483647
using namespace std;

struct Course
{
    int t, c;
    int max = 0;
    vector<int> pre; //prerequisites
};

int findMax(int index, Course *courses)
{
    if (courses[index].c == 0)
    {
        courses[index].max = courses[index].t;
        return courses[index].max;
    }
    int max = 0;
    for (int i = 0; i < courses[index].c; i++) //loops all the prereqs
    {
        if (courses[index].max == 0)
        {
            max = MAX(max, courses[index].t + findMax(courses[index].pre[i], courses)); //
        }
        else
        {
            max = MAX(max, courses[index].max);
        }
    }
    courses[index].max = max;
    return max;
}

int main()
{
    int n;
    cin >> n;
    Course courses[n];
    for (int i = 0; i < n; i++)
    {
        cin >> courses[i].t >> courses[i].c;
        for (int j = 0; j < courses[i].c; j++)
        {
            int p;
            cin >> p;
            courses[i].pre.push_back(p - 1);
        }
    }
    int index = 0;
    findMax(0, courses);
    for (int i = 1; i < n; i++)
    {
        findMax(i, courses);
        if (courses[i].max > courses[index].max)
        {
            index = i;
        }
    }

    int next = index;
    vector<int> taken;
    taken.push_back(next);
    while (courses[next].c != 0)
    {
        int maxIndex = courses[next].pre[0];
        int max = courses[courses[next].pre[0]].max;
        for (int i = 1; i < courses[next].c; i++)
        {
            if (max < courses[courses[next].pre[i]].max)
            {
                max = courses[courses[next].pre[i]].max;
                maxIndex = courses[next].pre[i];
            }
        }
        taken.push_back(maxIndex);
        next = maxIndex;
    }
    sort(taken.begin(), taken.end());
    index = 0;
    for (int i = 0; i < n; i++)
    {
        cout << courses[i].max << " ";
        if (index < taken.size() && i == taken[index])
        {
            index++;
            cout << 1;
        }
        else
            cout << 0;
        cout << "\n";
    }
    //cout << index << "\n";
}