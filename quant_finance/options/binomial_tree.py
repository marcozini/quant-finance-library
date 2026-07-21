#### Binomial Tree Option Pricing ####

import numpy as np

class BinomialTreeOption:
    def __init__(
        self,
        S0,
        K,
        T,
        r,
        sigma,
        N,
        q=0.0,
        option_type="call",
    ):
        # Store contract and model parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.q = q
        self.option_type = option_type

        # Validate inputs
        if self.S0 <= 0:
            raise ValueError("S0 must be positive.")

        if self.K <= 0:
            raise ValueError("K must be positive.")

        if self.T <= 0:
            raise ValueError("T must be positive.")

        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

        if not isinstance(self.N, (int, np.integer)) or isinstance(self.N, bool):
            raise TypeError("N must be an integer.")

        if self.N <= 0:
            raise ValueError("N must be positive.")

        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")

        # Model & Tree parameters
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = np.exp(-self.sigma * np.sqrt(self.dt))
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)

        if not 0.0 <= self.p <= 1.0:
            raise ValueError("Risk-neutral probability must lie between 0 and 1.")
            
            
        
    def build_tree(self):
        
        # Build and return the stock-price tree:
        stock_tree = []
    
        for step in range(self.N + 1):
            stock_prices = []
    
            for node in range(step + 1):
                #Each node index represents the number of upward moves
                stock_price = (self.S0* self.u**node* self.d**(step - node))
                stock_prices.append(stock_price)
    
            stock_tree.append(np.array(stock_prices))
    
        return stock_tree
        
        

    def payoff(self, stock_price):
        # Return intrinsic value for call or put
        if self.option_type == "call":
            return np.maximum(stock_price - self.K, 0.0)
    
        elif self.option_type == "put":
            return np.maximum(self.K - stock_price, 0.0)
    
        raise ValueError("option_type must be 'call' or 'put'")
    
    
            
    def can_exercise(self, step):
        raise NotImplementedError(
            "Subclasses must define the exercise rule."
        )



    def price(self):

        stock_tree = self.build_tree()
    
        # The terminal payoff provides the boundary condition
        option_values = self.payoff(stock_tree[-1])
    
        # Store terminal layer first
        option_value_tree = [option_values.copy()]
    
        # Backward induction
        # Roll the option values backward from maturity to today
        for step in range(self.N - 1, -1, -1):
    
            continuation_values = []
    
            for node in range(step + 1):
                # Adjacent values are the down and up children of the current node
                continuation_value = self.discount * (
                    (1.0 - self.p) * option_values[node]
                    + self.p * option_values[node + 1])
    
                continuation_values.append(continuation_value)
    
            option_values = np.array(continuation_values)
    
            # Early exercise for American and Bermudan options
            if self.can_exercise(step):
                intrinsic_values = self.payoff(stock_tree[step])
                option_values = np.maximum(
                    option_values,
                    intrinsic_values,
                )
    
            option_value_tree.append(option_values.copy())
    
        option_value_tree.reverse()
    
        return option_values[0], option_value_tree



# European Option
class EuropeanOption(BinomialTreeOption):
    def can_exercise(self, step):
        return False


# American Option
class AmericanOption(BinomialTreeOption):
    def can_exercise(self, step):
        return True
    
    
# Bermudan Option
class BermudanOption(BinomialTreeOption):
    def __init__(
        self,
        S0,
        K,
        T,
        r,
        sigma,
        N,
        exercise_steps, # Instaed of calender dates
        q=0.0,
        option_type="call",
    ):
        # Run the constructor of BinomialTreeOption
        super().__init__(
            S0,
            K,
            T,
            r,
            sigma,
            N,
            q,
            option_type,
        )

        # Store the allowed exercise steps
        self.exercise_steps = set(exercise_steps)
        
        # Validate Bermudan exercise steps
        for step in self.exercise_steps:
            if not isinstance(step, (int, np.integer)) or isinstance(step, bool):
                raise TypeError("Exercise steps must be integers.")
        
            if not 0 <= step < self.N:
                raise ValueError("Exercise steps must be between 0 and N - 1.")

    def can_exercise(self, step):
        return step in self.exercise_steps
    
