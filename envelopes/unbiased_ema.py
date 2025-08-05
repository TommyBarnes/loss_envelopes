#Simple EMA with unbiased constant step size trick (Exercise 2.7 in Sutton & Barto)
class UnbiasedEMA():
    def __init__(self, step_size=0.01) -> None:
        self.bias_trace = 0
        self.step_size = step_size
        self.value = 0
    
    def update(self, new_value):
        self.bias_trace += self.step_size * (1 - self.bias_trace)
        unbiased_step_size = self.step_size / self.bias_trace
        self.value += unbiased_step_size * (new_value - self.value)
        return self.value
