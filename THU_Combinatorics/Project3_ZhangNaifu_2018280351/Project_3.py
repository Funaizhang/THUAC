# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:05:47 2018

@author: Zhang Naifu 2018280351
"""
  
def calc_partition_number(N):
    '''
    main function to return the partition number p(N) given input N
    '''
    # handles the case where N<0
    if N<0:
        return 0
    
    else:
        # initalize the list of length N+1 for p(n), first element is p(0) = 1
        partitionNumList = [0] * (N+1)
        partitionNumList[0] = 1
        
        # loop over n to generate n partition numbers 
        for n in range(1,N+1):
            # loop over k in the recurrence relation
            # this corresponds to the sum function
            for k in range(1,N+1):
                # calculate the two pentagonal numbers, i.e. g(k) = k*(3*k-1)/2, g(-k) = k*(3*k+1)/2
                coefficient = (-1)**(k+1)
                pentagonalNumFormer = int(k*(3*k-1)/2)
                pentagonalNumLatter = int(k*(3*k+1)/2)
                
                # calculate the two RHS terms in the recurrence relation, and sum them up
                # we check that 
                if n >= pentagonalNumFormer:
                    partitionNumList[n] += coefficient*partitionNumList[n-pentagonalNumFormer]
                if n >= pentagonalNumLatter:
                    partitionNumList[n] += coefficient*partitionNumList[n-pentagonalNumLatter]
        
        return partitionNumList[-1]


def wrapper():  
    '''
    Just a function asking for the right user input
    '''
    sizeCheck = False
    
    # This block prompts user for an integer
    while not sizeCheck:
        N = None
        while (type(N) is not int):
            try:
                N = int(input("Please enter a natural number: "))
            except ValueError:
               continue
              
        # Double check with user if input > 10000
        if N>=10000:
            y = input('Computing p('+str(N)+') might take a long time.\nEnter Y to continue, any other key to re-enter integer.')
            if (y == 'Y' or y == 'y'):
                sizeCheck = True
        else:
            sizeCheck = True
        
    print('p('+ str(N) + ') = ' + str(calc_partition_number(N)))
    
wrapper()

