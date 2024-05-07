from pyomo.environ import *
import numpy as np

# Constants
S = 2 # the number of scenarios
p_s = 1/2 # the density of each scenario
n = 8 # the number of products
m = 5 # the number of parts to be ordered before production

# Define a function to randomly generate values
def get_data():
    np.random.seed(0)
    vector_b = np.random.uniform(20, 30, m) # the pre-ordered cost of each part
    vector_l = np.random.uniform(100, 200, n) # the production cost of the product
    vector_q = np.random.uniform(500, 1000, n) # the selling price of the product
    vector_s = np.random.uniform(10, 15, m) # the salvage value of leftover parts
    matrix_A = np.random.uniform(0, 10, (n, m)) # the matrix of requirements
    vector_D = np.random.binomial(10, 0.5, size=(2, 8)) # random demand vector follows the binomial distribution Bin (10, 1/2)
    return vector_b, vector_l, vector_q, vector_s, matrix_A, vector_D

vector_b, vector_l, vector_q, vector_s, matrix_A, vector_D = get_data()

# Print the randomly generated values
print("Preorder cost b:")
print(vector_b)
print("Production cost l:")
print(vector_l)
print("Selling price q:")
print(vector_q)
print("Salvage value s:")
print(vector_s)
print("Requirement matrix A:")
print(matrix_A)
print("Demand Vector D: ")
print(vector_D)
# Initialize a model for the First-Stage Problem
model = ConcreteModel()
# Initialize decision variables for the First-Stage Problem
model.x = Var(range(m), domain=NonNegativeReals)
model.z = Var([1,2], range(n), domain=NonNegativeReals)
# Create the objective function
Obj_function = sum(vector_b[j] * model.x[j] for j in range(m)) + 0.5 * (sum((vector_l[i] - vector_q[i]) * model.z[1, i] for i in range(n))) + 0.5 * (sum((vector_l[i] - vector_q[i]) * model.z[2, i] for i in range(n))) 
model.obj = Objective(expr=Obj_function, sense=minimize)
# Create and add the model constraints
model.constraints = ConstraintList()
for j in range(m):
    model.constraints.add(0 <= model.x[j] - sum(matrix_A[i, j] * model.z[1, i] for i in range(n)))  
for j in range(m):
    model.constraints.add(0 <= model.x[j] - sum(matrix_A[i, j] * model.z[2, i] for i in range(n)))   
for i in range(n): 
    model.constraints.add(model.z[1, i] <= vector_D[0, i])    
for i in range(n): 
    model.constraints.add(model.z[2, i] <= vector_D[1, i])   
# Solve the model
solver = SolverFactory('glpk')
result = solver.solve(model)
# Print the optimal solution
x_optimal_solution = [value(model.x[j]) for j in range(m)]
z_optimal_solution = [[value(model.z[i, j]) for j in range(n)] for i in [1, 2]] 
print("\nOptimal Solution for the First Stage:")
print("x:", x_optimal_solution)
print("z in the First Scenario:", z_optimal_solution[0])
print("z in the Second Scenario:", z_optimal_solution[1])
print("Cost of the First Stage:", model.obj())
# Initialize a model for the Second-Stage Problem in the First Scenario
x = x_optimal_solution
model_21 = ConcreteModel()
# Initialize decision variables
model_21.y = Var(range(m), domain=NonNegativeReals)
model_21.z = Var(range(n), domain=NonNegativeReals)
# Create the objective function 
Obj_function_1 = sum((vector_l[i] - vector_q[i]) * model_21.z[i] for i in range(n)) - sum(vector_s[j] * model_21.y[j] for j in range(m))
model_21.obj_1 = Objective(expr=Obj_function_1, sense=minimize)
# Create and add the model constraints
model_21.constraints_1 = ConstraintList()
for j in range(m):
    model_21.constraints_1.add(model_21.y[j] == x[j] - sum(matrix_A[i, j] * model_21.z[i] for i in range(n)))  
for i in range(n): 
    model_21.constraints_1.add(model_21.z[i] <= vector_D[0, i])    
# Solve the model
solver = SolverFactory('glpk')
result = solver.solve(model_21)  
# Print the optimal solution
y_optimal_solution_1 = [value(model_21.y[j]) for j in range(m)] 
z_optimal_solution_1 = [value(model_21.z[i]) for i in range(n)]
print("\nOptimal Solution for the Second Stage in the First Scenario:")
print("y in the First Scenario:", y_optimal_solution_1)
print("z in the First Scenario:", z_optimal_solution_1)
print("Cost of the Second Stage in the First Scenario:", model_21.obj_1())
# Initialize a model for the Second-Stage Problem in the Second Scenario
model_22 = ConcreteModel()
# Initialize decision variables
model_22.y = Var(range(m), domain=NonNegativeReals)
model_22.z = Var(range(n), domain=NonNegativeReals)
# Create the objective function 
Obj_function_2 = sum((vector_l[i] - vector_q[i]) * model_22.z[i] for i in range(n)) - sum(vector_s[j] * model_22.y[j] for j in range(m))
model_22.obj_2 = Objective(expr=Obj_function_2, sense=minimize)
# Create and add the model constraints
model_22.constraints_2 = ConstraintList()
for j in range(m):
    model_22.constraints_2.add(model_22.y[j] == x[j] - sum(matrix_A[i, j] * model_22.z[i] for i in range(n)))  
for i in range(n): 
    model_22.constraints_2.add(model_22.z[i] <= vector_D[1, i])    
# Solve the model
solver = SolverFactory('glpk')
result = solver.solve(model_22)  
# Print the optimal solution
y_optimal_solution_2 = [value(model_22.y[j]) for j in range(m)] 
z_optimal_solution_2 = [value(model_22.z[i]) for i in range(n)]
print("\nOptimal Solution for the Second Stage in the Second Scenario:")
print("y in the Second Scenario:", y_optimal_solution_2)
print("z in the Second Scenario:", z_optimal_solution_2)
print("Cost of the Second Stage in the Second Scenario:", model_22.obj_2())

