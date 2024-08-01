from sympy import E
import time



class iter_testing():
    def __init__(self, num_iters = 10):
        self.num_iters = num_iters
    
    def __iter__(self):
        list_of_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        
        for letter in list_of_letters:
            for i in range(self.num_iters):
                time.sleep(1)
                yield i, letter 
                
if __name__ == "__main__":
    it = iter_testing()
    for i, letter in it:
        print(f'Iteration: {letter}')
        print(i)
        