import pandas as pd
import numpy as np
from itertools import product


def read_excel(file_path):
    data = pd.read_excel(file_path, header=None)
    costs = data.iloc[:3, :3].values
    supply = data.iloc[:3, 3].values
    demand = data.iloc[3, :3].values
    return costs, supply, demand

def northwest_corner(costs, supply, demand):
    rows, cols = costs.shape
    allocation = np.zeros((rows, cols))
    i, j = 0, 0
    while i < rows and j < cols:
        alloc = min(supply[i], demand[j])
        allocation[i, j] = alloc
        supply[i] -= alloc
        demand[j] -= alloc
        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1
    return allocation
# Minimum Cost Method
def minimum_cost_method(costs, supply, demand):
    rows, cols = costs.shape
    allocation = np.zeros((rows, cols))
    cost_indices = sorted(product(range(rows), range(cols)), key=lambda x: costs[x])
    for i, j in cost_indices:
        if supply[i] == 0 or demand[j] == 0:
            continue
        alloc = min(supply[i], demand[j])
        allocation[i, j] = alloc
        supply[i] -= alloc
        demand[j] -= alloc
    return allocation

def vogels_method(costs, supply, demand):
    rows, cols = costs.shape
    allocation = np.zeros((rows, cols))
    
    # Create working copies
    supply_left = supply.copy()
    demand_left = demand.copy()
    
    # Create masks for remaining supply and demand points
    supply_points = list(range(rows))
    demand_points = list(range(cols))
    
    while supply_points and demand_points:
        penalties = []
        
        # Calculate row penalties
        for i in supply_points:
            row_costs = [costs[i][j] for j in demand_points]
            if len(row_costs) >= 2:
                sorted_costs = sorted(row_costs)
                penalties.append((sorted_costs[1] - sorted_costs[0], i, 'row'))
            elif len(row_costs) == 1:
                penalties.append((row_costs[0], i, 'row'))
        
        # Calculate column penalties
        for j in demand_points:
            col_costs = [costs[i][j] for i in supply_points]
            if len(col_costs) >= 2:
                sorted_costs = sorted(col_costs)
                penalties.append((sorted_costs[1] - sorted_costs[0], j, 'col'))
            elif len(col_costs) == 1:
                penalties.append((col_costs[0], j, 'col'))
        
        if not penalties:
            break
            
        # Find highest penalty
        penalty, index, kind = max(penalties, key=lambda x: x[0])
        
        # Find cell with minimum cost
        if kind == 'row':
            i = index
            j = demand_points[np.argmin([costs[i][j] for j in demand_points])]
        else:
            j = index
            i = supply_points[np.argmin([costs[i][j] for i in supply_points])]
        
        # Allocate
        alloc = min(supply_left[i], demand_left[j])
        allocation[i, j] = alloc
        supply_left[i] -= alloc
        demand_left[j] -= alloc
        
        # Remove satisfied points
        if supply_left[i] < 1e-10:  # Use small threshold for float comparison
            supply_points.remove(i)
        if demand_left[j] < 1e-10:  # Use small threshold for float comparison
            demand_points.remove(j)
    
    return allocation

def calculate_total_cost(allocation, costs):
    return np.sum(allocation * costs)

def find_basic_variables(allocation):
    """Find positions of basic variables (non-zero allocations)"""
    rows, cols = allocation.shape
    return [(i, j) for i, j in product(range(rows), range(cols)) if allocation[i, j] > 0]

def compute_dual_variables(costs, basic_vars):
    """Compute dual variables u and v using basic variables"""
    rows = max(i for i, _ in basic_vars) + 1
    cols = max(j for _, j in basic_vars) + 1
    
    # Initialize dual variables
    u = [None] * rows
    v = [None] * cols
    u[0] = 0  # Set first u to 0 as reference
    
    # Iterate until all dual variables are computed
    while any(x is None for x in u + v):
        found_new = False
        for i, j in basic_vars:
            if u[i] is not None and v[j] is None:
                v[j] = costs[i, j] - u[i]
                found_new = True
            elif u[i] is None and v[j] is not None:
                u[i] = costs[i, j] - v[j]
                found_new = True
        if not found_new:
            break
            
    # Fill any remaining None values with 0
    u = [0 if x is None else x for x in u]
    v = [0 if x is None else x for x in v]
    
    return u, v

def compute_reduced_costs(costs, u, v):
    """Compute reduced costs for all variables"""
    rows, cols = costs.shape
    reduced_costs = np.zeros((rows, cols))
    
    for i, j in product(range(rows), range(cols)):
        reduced_costs[i, j] = costs[i, j] - u[i] - v[j]
    
    return reduced_costs

def find_entering_variable(reduced_costs, basic_vars):
    """Find the entering variable with the most negative reduced cost"""
    min_cost = 0
    entering = None
    
    rows, cols = reduced_costs.shape
    for i, j in product(range(rows), range(cols)):
        if (i, j) not in basic_vars and reduced_costs[i, j] < min_cost:
            min_cost = reduced_costs[i, j]
            entering = (i, j)
            
    return entering

def find_cycle(entering, basic_vars, rows, cols):
    """Find a cycle starting from the entering variable"""
    def find_next_in_cycle(current, visited, direction):
        i, j = current
        candidates = []
        
        if direction == 'horizontal':
            # Look for vertical moves
            for row in range(rows):
                if row != i and (row, j) in basic_vars and (row, j) not in visited:
                    candidates.append((row, j))
        else:  # vertical
            # Look for horizontal moves
            for col in range(cols):
                if col != j and (i, col) in basic_vars and (i, col) not in visited:
                    candidates.append((i, col))
                    
        return candidates

    def build_cycle(start, current, visited, direction, cycle):
        if len(cycle) > 1 and current == start:
            return cycle
            
        next_points = find_next_in_cycle(current, visited, direction)
        for next_point in next_points:
            new_direction = 'vertical' if direction == 'horizontal' else 'horizontal'
            new_cycle = build_cycle(start, next_point, visited + [current], new_direction, cycle + [current])
            if new_cycle:
                return new_cycle
        return None

    # Try to build cycle starting horizontally and vertically
    cycle = build_cycle(entering, entering, [], 'horizontal', [])
    if not cycle:
        cycle = build_cycle(entering, entering, [], 'vertical', [])
        
    return cycle + [entering] if cycle else None

def update_allocation(allocation, cycle):
    """Update the allocation matrix based on the cycle"""
    # Determine the maximum amount that can be shifted
    theta = float('inf')
    for i in range(1, len(cycle), 2):  # Check odd-indexed positions (minus positions)
        theta = min(theta, allocation[cycle[i]])
        
    # Update allocations along the cycle
    for i in range(len(cycle)):
        pos = cycle[i]
        if i % 2 == 0:  # Plus position
            allocation[pos] += theta
        else:  # Minus position
            allocation[pos] -= theta
            
    return allocation

def transportation_simplex(costs: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    """Implement the transportation simplex algorithm"""
    rows, cols = costs.shape
    max_iterations = 100  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        # Find basic variables
        basic_vars = find_basic_variables(allocation)
        
        # Compute dual variables
        u, v = compute_dual_variables(costs, basic_vars)
        
        # Compute reduced costs
        reduced_costs = compute_reduced_costs(costs, u, v)
        
        # Find entering variable
        entering = find_entering_variable(reduced_costs, basic_vars)
        
        if entering is None:
            break  # Optimal solution found
        
        # Find cycle
        cycle = find_cycle(entering, basic_vars, rows, cols)
        
        if not cycle:
            break  # No valid cycle found
        
        # Update allocation
        allocation = update_allocation(allocation, cycle)
        
        iteration += 1
        
    return allocation

def main():
    # Test with the provided data
    file_path = "C:\\Users\\jelen\\OneDrive\\Documents\\UVic\\Classeur Opti.xlsx"  # Replace with your file path
    costs, supply, demand = read_excel(file_path)
    print(costs)
    print(supply)
    print(demand)
    
    
    
    print("Northwest Corner Method:")
    nw_allocation = northwest_corner(costs, supply.copy(), demand.copy())
    print(nw_allocation)
    print("Minimum Cost Method:")
    min_cost_allocation = minimum_cost_method(costs, supply.copy(), demand.copy())
    print(min_cost_allocation)
    print("Vogel's Method:")
    vogels_allocation = vogels_method(costs, supply.copy(), demand.copy())
    print(vogels_allocation)
    print("Transportation Simplex Algorithm:")
    final_allocation = transportation_simplex(costs, vogels_allocation)
    print(final_allocation)

main()
