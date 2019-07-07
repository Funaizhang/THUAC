//
//  Programming1.c
//  
//
//  Created by Zhang Naifu 2018280351 on 02/12/2018.
//
//
/*
 Input:
 　　3
 　　-2 5 -1
 　　-2 2
 　　Note:
 　　first line length of array
 　　second line Array values
 　　third line lower bound and upper bound
 　　one space between each value
 Output:
    3
 
 Example:
 　　Given nums = [-2, 5, -1], lower = -2, upper = 2,
 　　Return 3.
 　　The three ranges are : [0, 0], [2, 2], [0, 2] and their respective sums are: -2, -1, 2.
 The following implementation runs in O(n log n)
*/
#include <stdio.h>
#include <stdlib.h>
// helper function used in qsort()
int cmpfunc (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}
// helper function used in countRangeSum(), implements divide & conquer
int sortSums (int *sums, int start, int end, int lower, int upper) {
    int i;
    int count = 0;
    int mid = (start + end) / 2;
    
    // if there is only 1 element in the sums array, return count = 0
    if (end - start <= 1)
        return 0;
    
    // increase the count by the sortSums of the left and right subarrays
    count += (sortSums(sums, start, mid, lower, upper) + sortSums(sums, mid, end, lower, upper));
    
    // find the number of sums that cross the left and right subarrays
    int j = mid, k = mid;
    for (i = start; i < mid; i++){
        while (k < end && (sums[k] - sums[i]) <= upper) {
            k++;
        }
        while (j < end && (sums[j] - sums[i]) < lower) {
            j++;
        }
        count += k - j;
    }
    
    // Sort merged array
    qsort(sums + start, end - start, sizeof(int), cmpfunc);
    return count;
}
// main function to be called
int countRangeSum (int *nums, int numSize, int lower, int upper) {
    int *sums = malloc((numSize + 1) * sizeof(int));
    // preprocess nums into sums array of prefix sums
    int i;
    for (i = 1; i <= numSize; i++) {
        sums[i] = sums[i-1] + nums[i-1];
    }
    
    return sortSums(sums, 0, numSize + 1, lower, upper);
    free(sums);
}
// main to preprocess the input
int main(void){
    // read the first line
    int l;
    scanf("%d", &l);
    
    // read the second line
    int i;
    int nums[l];
    for (i = 0; i<l; i++){
        int n;
        char ch;
        scanf("%d%c", &n, &ch);
        nums[i] = n;
    }
    
    // read the third line
    int bounds[2];
    for (i = 0; i<2; i++){
        int n;
        char ch;
        scanf("%d%c", &n, &ch);
        bounds[i] = n;
    }
    
    // return result
    int result = countRangeSum(nums, l, bounds[0], bounds[1]);
    printf("%d", result);
}
