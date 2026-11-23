import heapq
from search.ucs import uniform_cost_search

def heuristic(a, b):
    """Heuristic function using Manhattan Distance for A* Search."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(env, start, goal):
    """2. Implement A* Search algorithm for efficient routing."""
    pq = [(heuristic(start, goal), 0, start, [start])]
    g_scores = {start: 0}
    visited = set()
    
    while pq:
        f, g, current, path = heapq.heappop(pq)
        if current == goal:
            return path, g
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor in env.get_neighbors(current):
            new_g = g + 1
            if neighbor not in g_scores or new_g < g_scores[neighbor]:
                g_scores[neighbor] = new_g
                heapq.heappush(pq, (new_g + heuristic(neighbor, goal), new_g, neighbor, path + [neighbor]))
    return None, float('inf')

def build_distance_matrix(env):
    """3. Build Distance Matrix between the hive and flowers for other project modules."""
    locations = [env.hive] + list(env.flowers.keys())
    matrix = {}
    for start in locations:
        for goal in locations:
            if start == goal:
                matrix[(start, goal)] = ([], 0)
            else:
                path, cost = a_star_search(env, start, goal)
                matrix[(start, goal)] = (path, cost)
    return matrix

def search_greedy_strategy(env):
    """4. Greedy Strategy with Energy Safety Check (Ensures the bee can return safely)."""
    current_pos = env.hive
    current_energy = env.energy_limit
    total_nectar = 0
    visited_flowers = set()
    
    print("--- 🐝 Starting Search-Based Brain Simulation (Greedy + Energy Check) ---")
    
    while len(visited_flowers) < len(env.flowers):
        best_flower = None
        best_path = None
        min_cost = float('inf')
        
        for flower in env.flowers:
            if flower not in visited_flowers:
                path_to_flower, cost_to_flower = a_star_search(env, current_pos, flower)
                path_to_hive, cost_to_hive = a_star_search(env, flower, env.hive)
                
                # Safety Condition: Energy must cover trip to flower AND trip back to hive
                if path_to_flower and path_to_hive:
                    if current_energy >= (cost_to_flower + cost_to_hive):
                        if cost_to_flower < min_cost:
                            min_cost = cost_to_flower
                            best_flower = flower
                            best_path = path_to_flower
                            
        if not best_flower:
            print("⚠️ Remaining energy is not enough to visit a new flower and return safely.")
            break
            
        current_energy -= min_cost
        total_nectar += env.flowers[best_flower]
        visited_flowers.add(best_flower)
        current_pos = best_flower
        print(f"🌸 Visited flower at {best_flower} | Total Nectar: {total_nectar} | Remaining Energy: {current_energy}")

    # Final return to the hive
    path_back, cost_back = a_star_search(env, current_pos, env.hive)
    if path_back and current_energy >= cost_back:
        current_energy -= cost_back
        print(f"🏠 Bee successfully returned to the hive! Final Total Nectar: {total_nectar} | Remaining Energy: {current_energy}")
    else:
        print("❌ The bee failed to return back to the hive!")
if __name__ == "__main__":
    from environment.garden import BeeGarden

    env = BeeGarden()
    
    path, cost = a_star_search(env, env.hive, (7, 7))
    print("A* Path to (7, 7):", path)
    print("A* Cost to (7, 7):", cost)
    print("-" * 40)

    search_greedy_strategy(env)        
