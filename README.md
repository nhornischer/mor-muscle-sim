# BA_Code

This repository contains the code and files corresponding to the bachelorthesis "Model order reduction with transformed modes for electrophysiological simulations" by Niklas Hornischer.

The main concept behind the code is storing all the conducted experiments under unique hashes (combination of characters and numbers), a list of all hashes can be found in "hashes.txt". Each experiment gets such a unique hash and writes its parameters into the log-file "logs.txt". All used experiments where executed based on the commands in "experiments.py". A list with all these experiments and short explanations can be found in "experiments_logs.txt". 

All bare-results are stored in the folder "solutions", the folder "plots" is an empty placeholder for new plots. 


Explanation of the python files:

"base.py":          includes all the methods for storing, plotting, loading. Can be seen as the backbone of the whole code. To delete specific hashes, look for the matching method in this file.
   
"decomp.py":        includes all methods to construct the reduced ansatz space
   
"experiments.py":   all used numerical experiments
   
"fibre_0...":       initial value files directly from Felix Huber
  
"minidihu.py":      FOM simulation from Felix Huber

"solver.py":        includes the methods to solve the FOM and ROM
