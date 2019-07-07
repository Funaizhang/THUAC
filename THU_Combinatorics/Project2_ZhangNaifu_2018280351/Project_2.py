# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 01:59:40 2018

@author: zhangnaifu
"""

# Import external libraries
import time
import math

# The main fucntion
def heaps_recursive(a,n):
    # Additional print statement to print out the original input a    
    if n == len(a):
        print(a)
    
    # This is the actual Heap's algorithm that does the bulk of the work, and prints n!-1 permutations
    # Initialize counter i, which ensures heaps_recursive(a,n-1) is recursed n times
    i = 0
    while True:
        # n = 2 is our base case, where we simply print the 2 permutations
        if n>2:
            heaps_recursive(a,n-1)
        # Check the recursion has not happened n-1 times yet
        if n<=i+1:
            break
        # If n is odd, swap first element with last element
        elif (n%2) == 1:
            a[0],a[n-1]=a[n-1],a[0]
        # If n is even, swap ith element with last element
        else:
            a[i],a[n-1]=a[n-1],a[i]
        print(a)
        i+=1


# Just a function asking for the right user input
def wrapper():  
    sizeCheck = False
    
    # This block prompts user for a positive integer to be used for Heap's algorithm
    while not sizeCheck:
        n = None
        # Ensure input is a positive integer
        while (type(n) is not int or n<=0):
            try:
                n = int(input("Please enter a positive integer: "))
            except ValueError:
               continue
    
        # Double check with user if input > 10
        if n>10:
            y = input('Permutating '+str(n)+' elements might take some time.\nEnter Y to continue, any other key to re-enter integer.')
            if (y == 'Y' or y == 'y'):
                sizeCheck = True
        else:
            sizeCheck = True
    
    # Executes Heap's algorithm, enumerating each permutation
    start_time = time.time()
    heaps_recursive(list(range(1,n+1)),n)
    end_time = time.time()
    # Print out the total number of permutations
    print(str(math.factorial(n))+' permutations from '+str(n)+' elements')
    # Find time taken
    print("{0:.2f} seconds taken".format(end_time - start_time))
    
    
wrapper()
