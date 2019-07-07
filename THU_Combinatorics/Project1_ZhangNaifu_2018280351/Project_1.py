# -*- coding: utf-8 -*-
"""
Created on Wed Oct 3 02:02:41 2018

@author: Zhang Naifu 2018280351
"""

# Import external libraries
from itertools import permutations
import time

# invalidJumps enumerates all the possible invalid 'jumps'
# representativeCases is used by semiBruteForce, the list contains a corner digit '1', a side digit '2' and the centre digit '5'
# These lists are in strings for easier manipulation later
digits = ['1','2','3','4','5','6','7','8','9']
invalidJumps = ['13','31','46','64','79','97','17','71','28','82','39','93','19','91','37','73']
representativeCases = ['1','2','5']


# Function returns False if the unlock sequence contains any invalid jump, True otherwise
def isValid(unlockSequence, invalidJumps):
    
    # Ensure there is no repetition in the digits
    for j in digits:
        if unlockSequence.count(j) > 1:
            return False
        
    for i in invalidJumps:
        # If invalid jump is in the unlock sequence
        jumpIndex = unlockSequence.find(i)
        if jumpIndex >= 0:
            # Observe that the digit being jumped over is the average of the digits at the start and end of the invalid jump
            digitJumped = str(int((int(i[0]) + int(i[1]))/2))
            jumpedIndex = unlockSequence.find(digitJumped)
            
            # Return False if jumped digit has NOT been touched before
            if jumpedIndex < 0 or jumpedIndex < jumpIndex:
                return False
                
    # Return True if NO invalid jump found in the unlock sequence OR if the jumped digit has been touched before
    return True
    
  
# Generate all valid permutations of length n by brute force
def bruteForce(n):
    # Ensure password is 4-9 digits long
    if not isinstance(n,int) or n<4 or n>9:
        raise ValueError("Only passwords of length 4-9 allowed")
        
    # Generate all permutations of length n as strings, regardless of validity
    permAll = [''.join(digit) for digit in permutations(digits, n)]
    
    counter = 0
    # Check if each permutation is valid, count the valid ones
    for unlockSequence in permAll:
        if isValid(unlockSequence, invalidJumps):
            counter += 1
    return counter

# Tally number of valid permutations of length 4-9
start_time = time.time()
print("bruteForce: ")
print("{} different unlock sequences of length 4 to 9".format(sum(bruteForce(x) for x in range(4,10))))
print("{0:.2f} seconds taken".format(time.time() - start_time))


# Generate all valid permutations of length n
def semiBruteForce(n):
    # Ensure password is 4-9 digits long
    if not isinstance(n,int) or n<1 or n>9:
        raise ValueError("Only passwords of length 4-9 allowed")

    # Generate all permutations of length n-1 as strings, regardless of validity
    permAll = [''.join(digit) for digit in permutations(digits, n-1)]
    # Append all the permutations of length n-1 to the digits '1', '2' and '5'
    permRepresentative = [j+k for j in representativeCases for k in permAll]
    
    counter = 0
    # Check if each permutation is valid, count the valid ones
    for unlockSequence in permRepresentative:
        if isValid(unlockSequence, invalidJumps):
            # Count the unlock sequences starting from a corner or side (i.e. '1' or '2') 4 times
            if unlockSequence[0] == '1' or unlockSequence[0] == '2':
                counter += 4
            # Count the unlock sequences starting from the centre (i.e. '5') 1 time
            elif unlockSequence[0] == '5':
                counter += 1
            # Flag error if unlock sequence starts with any digit other than '1', '2' or '5'
            else:
                raise ValueError("Please only use 1, 2, 5 as the first digit")   
    return counter

# Tally number of valid permutations of length 4-9
start_time = time.time()
# Observe that semiBruteForce(8) = semiBruteForce(9) so no point running it twice
print("semiBruteForce: ")
print("{} different unlock sequences of length 4 to 9".format(sum(semiBruteForce(x) for x in range(4,8)) + semiBruteForce(8)*2))
print("{0:.2f} seconds taken".format(time.time() - start_time))