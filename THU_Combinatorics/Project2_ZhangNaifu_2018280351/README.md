# Project 2 Generating Permutations
This projects implements the algorithm which counts and lists all the permutations and of a user-given length without repetition or loss.

## Getting Started
The zip folder contains a Project_2_Report.pdf and a Project_2.py.

You can run Project_2.py in any Python 3.5 (or later) environment. The script would first prompt the user to enter a non-zero natural number, then proceed to count and enumerate the permutations of this length.
The results and code are discussed in Project_2_Report.pdf.

On Linux, Windows or OS X, with Python installed (see below otherwise), you can alternatively run the following in Terminal at the correct directory.
```sh
$ python Project_2.py
```

If you are working with Ubantu, run
```sh
$ sudo python3 Project_2.py
```
You will see the following prompt till you correctly enter a non-zero natural number
```sh
Please enter a positive integer:
```

The author does not recommend entering any number larger than 10, as the run could take up to minutes. If you do, the program makes sure this is intentional. For e.g.:
```sh
Please enter a positive integer: 15

Permutating 15 elements might take some time.
Enter Y to continue, any other key to re-enter integer.
```

Proceed accordingly. You should end up with an output like this:
```sh
Please enter a positive integer: 2
[1, 2]
[2, 1]
2 permutations from 2 elements
0.00 seconds taken
```

## Installing Python 3
*Skip this section if you have a working Python 3.*

If you are working with Linux, OS X and Windows, you can download Python 3 at [python.org](https://www.python.org/getit/).

The installation procedure over command prompt is not as straight forward, so please refer to this [online guide](https://realpython.com/installing-python/) for details of your system.

The entire process should will take a few minutes, or up to something like half an hour if starting from scratch.

## Installing IDE
*Skip this section if you have a working Python 3 IDE, or if you choose to run the script in Terminal.*

Author does not recommend Anaconda, as it runs into I/O errors for large inputs. Atom seems to work fine but requires installation of additional community add-ons for Python 3 - further details on Atom website.
* [PyCharm](https://www.jetbrains.com/help/pycharm/quick-start-guide.html)
* [Anaconda](http://docs.continuum.io/anaconda/install/#)
* [Emacs](https://www.gnu.org/software/emacs/download.html)
* [Atom](https://flight-manual.atom.io/getting-started/sections/installing-atom/)


## Author System Specifications
These specifications are not needed to run the script. If user is using any other OS or specifications, actual runtime might differ (but not hugely). Other results would not change.

**OS X** version 10.11.6  
**Processor** 2.5 GHz Intel Core i5  
**Memory** 8 GB 1600 MHz DDR3  
**Graphics** Intel HD Graphics 4000 1536 MB

## Author
* **Zhang Naifu** - *2018280351* - [Fork it on Github](https://github.com/Funaizhang/thuac/tree/master/Combinatorics%20Algorithms/Project2_ZhangNaifu_2018280351)

## References
* [Real Python](https://realpython.com/installing-python/)
