import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def original_function(x):
    """
    Define the original function here.
    This example uses: f(x) = x^3 - 2x^2 + 2
    """
    return (1-np.exp(-x))/(1+np.exp(-x))

def calculate_derivative(func, x):
    """
    Calculate the derivative of the function at point x
    using scipy's derivative function
    """
    return derivative(func, x, dx=1e-6)

def df2(func, x):
    return derivative(func, x, dx=1e-6, n=2)

def df3(func, x):
    return derivative(func, x, dx=1e-6, n=3)

def df4(func, x):
    return derivative(func, x, dx=1e-6, n=4)

def df5(func, x):
    return derivative(func, x, dx=1e-6, n=5)

def plot_function_and_derivative(x_range=(-3, 3), num_points=1000):
    """
    Plot the original function and its derivative
    """
    # Create points for x axis
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate y values for original function
    y_original = original_function(x)
    
    # Calculate derivative values
    y_derivative = [calculate_derivative(original_function, xi) for xi in x]

    # Calculate second-derivative values
    y_derivative_2 = [df2(original_function, xi) for xi in x]
    y_derivative_3 = [df3(original_function, xi) for xi in x]
    y_derivative_4 = [df4(original_function, xi) for xi in x]
    y_derivative_5 = [df5(original_function, xi) for xi in x]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot original function
    plt.plot(x, y_original, 'b-', label='f(x)')
    
    # Plot derivative
    plt.plot(x, y_derivative, 'r-', label="f'(x)")

    # Plot second derivative
    plt.plot(x, y_derivative_2, 'g-', label="f''(x)")
    plt.plot(x, y_derivative_3, 'c-', label="f'''(x)")
    plt.plot(x, y_derivative_4, 'm-', label="f''''(x)")
    plt.plot(x, y_derivative_5, 'y-', label="f'''''(x)")
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add title and labels
    plt.title("f(x), f'(x), and f''(x)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Add x and y axis lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Show the plot
    plt.show()

# Execute the plotting function
plot_function_and_derivative()