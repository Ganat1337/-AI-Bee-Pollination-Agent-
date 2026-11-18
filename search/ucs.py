import heapq

def uniform_cost_search(env, start, goal):
    """1. Implement Uniform Cost Search (UCS) algorithm."""
    pq = [(0, start, [start])]
    visited = set()
    
    while pq:
        cost, current, path = heapq.heappop(pq)
        if current == goal:
            return path, cost
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor in env.get_neighbors(current):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + 1, neighbor, path + [neighbor]))
    return None, float('inf')
if __name__ == "__main__":
    from environment.garden import BeeGarden
    env = BeeGarden()
    path, cost = uniform_cost_search(env, env.hive, (7, 7))
    print("UCS Path:", path)
    print("UCS Cost:", cost)