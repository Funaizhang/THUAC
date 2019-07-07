//
//  Programming2.c
//
//
//  Created by Zhang Naifu on 27/12/2018.
//  Runs fine on https://ideone.com/mGGOuY and https://onlinegdb.com/Hyl5V_fW4
/*
 Input:
 7 10
 word
 like
 first
 As
 the
 the
 complete
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int print_neatly(int *l, int n, int M, int *p, int *c) {
    int (*extras)[n+1] = malloc(sizeof(int[n+1][n+1]));
    int (*lc)[n+1] = malloc(sizeof(int[n+1][n+1]));
    
    int i;
    int j;
    
    // Compute extras[i, j] for 1 ≤ i ≤ j ≤ n.
    for (i = 1; i <= n; i++) {
        extras[i][i] = M - l[i];
        for (j = i+1; j <= n; j++) {
            extras[i][j] = extras[i][j-1] - l[j] - 1;
        }
    }
    
    // Compute lc[i, j] for 1 ≤ i ≤ j ≤ n.
    for (i = 1; i <= n ; i++) {
        for (j = i; j <= n; j++) {
            if (extras[i][j] < 0) {
                lc[i][j] = 100000;
            } else if (j == n && extras[i][j] >= 0) {
                lc[i][j] = 0;
            } else {
                lc[i][j] = pow(extras[i][j], 3);
            }
        }
    }
    
    // Compute c[j] and p[j] for 1 ≤ j ≤ n.
    c[0] = 0;
    for (j = 1; j <= n; j++) {
        c[j] = 100000;
        for (i = 1; i <= j; i++) {
            if (c[i-1] + lc[i][j] < c[j]) {
                c[j] = c[i-1] + lc[i][j];
                p[j] = i;
            }
        }
    }
    
    free(extras);
    free(lc);
    
    return 0;
}
// function to print output given c and p tables
int print_line(int *p, int j, int n, char *wordsList, int *lcumu) {
    int i = p[j];
    if (i != 1) {
        print_line(p, i - 1, n, wordsList, lcumu);
    }
    
    int k;
    int h;
    
    // print the k lines
    for (k = i; k <= j; k++) {
        for (h = lcumu[k-1]; h < lcumu[k]; h++) {
            printf("%c",wordsList[h]);
        }
        if (k != j)
            printf(" ");
    }
    if (j != n)
        printf("\n");
    
    return 0;
}
// main to preprocess the input
int main(void) {
    // read the first line
    int n;
    int M;
    scanf("%i %i", &n, &M);
    
    // read the subsequent lines
    char *wordsList = malloc(n*M*sizeof(char));
    char *word = malloc(M*sizeof(char));
    int *l = malloc((n+1)*sizeof(int));
    int *lcumu = malloc((n+1)*sizeof(int));
    int counter = 1;
    
    while(scanf("%s", word) != EOF) {
        // append word to wordsList
        strcat(wordsList, word);
        // append length of word to list l, and another list of cumulative length that is used for printf later
        l[counter] = strlen(word);
        lcumu[counter] = lcumu[counter-1] + strlen(word);
        counter ++;
    }
    
    // set up to run main function
    int *c = malloc(sizeof(int)*(n+1));
    int *p = malloc(sizeof(int)*(n+1));
    
    // compute the c and p tables
    print_neatly(l, n, M, p, c);
    printf("%d\n", c[n]);
    
    // print out the lines
    print_line(p, n, n, wordsList, lcumu);
    
    free(c);
    free(p);
    
    free(wordsList);
    free(word);
    free(l);
    free(lcumu);
}