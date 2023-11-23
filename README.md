# Genetic algorithm for decryption

## This genetic algorithm was developed as part of my coursework for Udemy's Lazy Programmer Machine Learning course. It takes a piece of text that has been encrypted by assigning each letter of the alphabet with a number between one and twenty-six, like the CodeWord puzzles you find in newspapers and magazines. 

## The genetic algorithm tries to find the most likely solution to the encrypted text. This algorithm works most of the time to successfully decrypt the text. Occasionally, it gets 'trapped' in a local minima and its best solution is way off. If this happens, re-run it and it should generate a good solution. It still tends to get two letters incorrect because they are so rarely occurring that there is insufficient material to allow the algorithm to find the optimal solution. 
