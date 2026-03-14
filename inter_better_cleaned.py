import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def is_dendrimer_in_bounds(dendrimer, boxlength):
    for point in dendrimer:
        if point[0] < 0 or point[0] > boxlength or point[1] < 0 or point[1] > boxlength:
            return False
    return True


def calculate_coverage_fraction_grid(coords, dist, boxlength, grid_resolution=1.0):
    # Create a grid
    grid_size = int(boxlength / grid_resolution)
    if grid_size <= 0:
        return 0.0

    covered_grid = np.zeros((grid_size, grid_size), dtype=bool)

    radius = dist / 2.0

    # For each particle, mark all grid points within its radius as covered
    for coord in coords:
        x_center, y_center = coord

        # Find the range of grid points to check
        x_min = max(0, int((x_center - radius) / grid_resolution))
        x_max = min(grid_size, int((x_center + radius) / grid_resolution) + 1)
        y_min = max(0, int((y_center - radius) / grid_resolution))
        y_max = min(grid_size, int((y_center + radius) / grid_resolution) + 1)

        # Check each grid point in the range
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                # Convert grid indices to actual coordinates
                x_grid = (i + 0.5) * grid_resolution
                y_grid = (j + 0.5) * grid_resolution

                # Check if this grid point is within the particle's radius
                distance_sq = (x_grid - x_center)**2 + (y_grid - y_center)**2
                if distance_sq <= radius**2:
                    covered_grid[i, j] = True

    # Calculate coverage fraction
    total_grid_points = grid_size * grid_size
    covered_points = np.sum(covered_grid)
    coverage_fraction = covered_points / total_grid_points
    return coverage_fraction




def dendrimercreator(boxlength=100000, dist=4.27, branches=3, branchingnumber=2, generations=10):
    dendrimercoords = []
    level_ends = [[] for _ in range(generations+1)]

    # Create core
    core = [random.uniform(0, boxlength), random.uniform(0, boxlength)]
    dendrimercoords.append(core)

    # Create first level branches from core
    for _ in range(branches):
        coreangle = random.uniform(0, 360)
        branchcoord1 = core[0] + 2*dist*np.cos(np.radians(coreangle))
        branchcoord2 = core[1] + 2*dist*np.sin(np.radians(coreangle))
        branchcoord = [branchcoord1, branchcoord2]
        # No distance check - allow self-intersection
        dendrimercoords.append(branchcoord)
        level_ends[0].append(branchcoord)

    # Create remaining levels of branches
    for level in range(1, generations):
        prev_level_ends = level_ends[level-1]
        for parent_branch in prev_level_ends:
            for _ in range(branchingnumber):
                branchangle = random.uniform(0, 360)
                branchcoord1 = parent_branch[0] + 2*dist*np.cos(np.radians(branchangle))
                branchcoord2 = parent_branch[1] + 2*dist*np.sin(np.radians(branchangle))
                branchcoord = [branchcoord1, branchcoord2]
                # No distance check - allow self-intersection
                dendrimercoords.append(branchcoord)
                level_ends[level].append(branchcoord)

    # Return both the full dendrimer and just the last generation points
    return {'full': dendrimercoords, 'last_gen': level_ends[generations-1]}

def create_grid_index(coords, boxlength, cell_size):
    # Initialize grid
    grid = defaultdict(list)

    # Assign each point to a grid cell
    for i, point in enumerate(coords):
        cell_x = int(point[0] / cell_size)
        cell_y = int(point[1] / cell_size)
        grid[(cell_x, cell_y)].append(i)

    return grid

def get_nearby_points(point, grid, coords, cell_size, dist):
    x, y = point
    cell_x = int(x / cell_size)
    cell_y = int(y / cell_size)

    cells_to_check = int(dist / cell_size) + 1

    nearby_indices = []
    for i in range(cell_x - cells_to_check, cell_x + cells_to_check + 1):
        for j in range(cell_y - cells_to_check, cell_y + cells_to_check + 1):
            if (i, j) in grid:
                nearby_indices.extend(grid[(i, j)])

    return nearby_indices


def calculate_true_coverage_fraction_fast(coords, bead_radius, boxlength, grid_resolution=None):
    if not coords:
        return 0.0

    if grid_resolution is None:
        base_resolution = min(bead_radius, boxlength / 200)  # Max 200x200 grid
        grid_resolution = max(base_resolution, bead_radius / 2)

    max_grid_points = 100000
    estimated_points = (boxlength / grid_resolution) ** 2

    if estimated_points > max_grid_points:
        grid_resolution = boxlength / np.sqrt(max_grid_points)

    x_points = np.arange(grid_resolution/2, boxlength, grid_resolution)
    y_points = np.arange(grid_resolution/2, boxlength, grid_resolution)

    print(f"Coverage calculation: {len(x_points)}x{len(y_points)} grid ({len(x_points)*len(y_points)} points)")

    X, Y = np.meshgrid(x_points, y_points)
    covered = np.zeros_like(X, dtype=bool)

    bead_radius_sq = bead_radius**2

    batch_size = min(50, len(coords))

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]

        for coord in batch_coords:
            dx = X - coord[0]
            dy = Y - coord[1]
            distances_sq = dx*dx + dy*dy
            covered |= (distances_sq <= bead_radius_sq)

    return np.sum(covered) / covered.size

def dend_run_rsa(iterations=5000, dist=5, boxlength=4000, max_consecutive_fails=5000000, generations=5, branches=3, branchingnumber=2, grid_resolution=2.0):
    adsorbcounter = 0
    adsorbtions = []
    attemptcounter = 1
    attempts = []
    grid_coverage_fractions = []
    coords = []
    consecutive_fails = 0

    cell_size = dist * 1.5

    bead_radius = dist / 2

    print(f"Starting RSA simulation with:")
    print(f"- {generations} generations")
    print(f"- {branches} initial branches")
    print(f"- {branchingnumber} branches per subsequent node")
    print(f"- {iterations} max iterations")
    print(f"- {boxlength}x{boxlength} box size")
    print(f"- {dist} minimum distance between particles")
    print(f"- {bead_radius} bead radius for coverage calculation")
    print(f"- Grid resolution: {grid_resolution}")
    print(f"- Self-intersection allowed within dendrimers")
    print(f"- Optimized with spatial indexing (cell size: {cell_size})")

    # Track timing for performance analysis
    start_time = time.time()
    last_time = start_time

    # Initialize first dendrimer
    dend_result = dendrimercreator(boxlength=boxlength, dist=dist,
                                  generations=generations, branches=branches,
                                  branchingnumber=branchingnumber)

    # Check if the dendrimer is within bounds
    first_dend = None
    if is_dendrimer_in_bounds(dend_result['full'], boxlength):
        first_dend = dend_result['full']
    else:
        # Try again with a more centered starting position
        print("Initial dendrimer out of bounds, retrying with centered position...")
        # Create a new dendrimer with core at center
        dend_result = dendrimercreator(boxlength=boxlength, dist=dist,
                                      generations=generations, branches=branches,
                                      branchingnumber=branchingnumber)
        first_dend = dend_result['full']


    coords.extend(first_dend)

    particle_area = np.pi*dist**2
    particles_per_dendrimer = len(first_dend)
    dendrimer_area = particles_per_dendrimer * np.pi * (dist/2)**2
    box_area = boxlength**2

    print(f"Success! Created initial dendrimer with {particles_per_dendrimer} particles")
    print(f"Dendrimer area: {dendrimer_area:.2f}")
    print(f"Box area: {box_area}")
    print(f"Starting RSA simulation...")

    adsorbcounter = 1


    grid_coverage = calculate_coverage_fraction_grid(coords, dist, boxlength, grid_resolution)

    grid_coverage_fractions.append(grid_coverage)
    attempts.append(attemptcounter)
    adsorbtions.append(adsorbcounter)

    # Create the initial spatial index
    grid = create_grid_index(coords, boxlength, cell_size)

    while attemptcounter < iterations:
        dend_result = dendrimercreator(boxlength=boxlength, dist=dist,
                                   generations=generations, branches=branches,
                                   branchingnumber=branchingnumber)

        new_dend = dend_result['full']
        last_gen_points = dend_result['last_gen']

        if not is_dendrimer_in_bounds(new_dend, boxlength):
            attemptcounter += 1
            consecutive_fails += 1
            adsorbtions.append(adsorbcounter)
            attempts.append(attemptcounter)
            grid_coverage_fractions.append(grid_coverage_fractions[-1])  # No change in coverage
            continue

        # Take random half of the last generation points for checking
        if len(last_gen_points) > 1:
            num_points_to_check = max(1, len(last_gen_points) // 2)  # At least check 1 point
            sample_points = random.sample(last_gen_points, num_points_to_check)
        else:
            sample_points = last_gen_points  # If there's only one point, check it

        intersect = False
        min_dist_squared = dist**2

        # Use spatial indexing for faster collision detection
        for point in sample_points:
            nearby_indices = get_nearby_points(point, grid, coords, cell_size, dist)
            for idx in nearby_indices:
                coord = coords[idx]
                dx = point[0] - coord[0]
                dy = point[1] - coord[1]
                if (dx*dx + dy*dy) <= min_dist_squared:
                    intersect = True
                    break
            if intersect:
                break

        if not intersect:
            # This dendrimer can be placed - rebuild the grid index
            start_idx = len(coords)
            coords.extend(new_dend)

            # Update the grid with new points
            for i, point in enumerate(new_dend):
                cell_x = int(point[0] / cell_size)
                cell_y = int(point[1] / cell_size)
                grid[(cell_x, cell_y)].append(start_idx + i)

            adsorbcounter += 1
            consecutive_fails = 0

            # Calculate grid-based coverage (only when a dendrimer is successfully placed)
            grid_coverage = calculate_coverage_fraction_grid(coords, dist, boxlength, grid_resolution)
        else:
            consecutive_fails += 1
            # Use previous grid coverage value
            grid_coverage = grid_coverage_fractions[-1] if grid_coverage_fractions else 0.0

        attemptcounter += 1
        adsorbtions.append(adsorbcounter)
        attempts.append(attemptcounter)
        grid_coverage_fractions.append(grid_coverage)

        if consecutive_fails >= max_consecutive_fails:
            print(f"Reached maximum consecutive failures ({max_consecutive_fails})")
            print(f"Final grid coverage fraction: {grid_coverage_fractions[-1]:.4f}")
            break

        if attemptcounter % 100 == 0:
            current_time = time.time()
            elapsed = current_time - last_time
            total_elapsed = current_time - start_time
            print(f"Attempt {attemptcounter}, Dendrimers placed: {adsorbcounter}, "
                  f"Grid coverage: {grid_coverage_fractions[-1]:.4f}, "
                  f"Consecutive fails: {consecutive_fails}, "
                  f"Time for last 100: {elapsed:.2f}s, Total time: {total_elapsed:.2f}s")
            last_time = current_time

    end_time = time.time()
    total_time = end_time - start_time
    print("\nFinal Statistics:")
    print(f"Total attempts: {attemptcounter}")
    print(f"Successfully placed dendrimers: {adsorbcounter}")
    print(f"Final grid coverage fraction: {grid_coverage_fractions[-1]:.4f}")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Average time per attempt: {total_time/attemptcounter:.4f} seconds")

    return attempts, adsorbtions, grid_coverage_fractions

# Run simulation with specified parameters
if __name__ == "__main__":
    generations = 3
    branches = 3         # Core branching
    branchingnumber = 2  # Branching number for subsequent generations
    iterations = 10000    # Maximum number of placement attempts
    boxlength = 300    # Size of the simulation box
    dist = 2.5   # Minimum distance between particles
    grid_resolution = 2.0  # Grid resolution for coverage calculation

    attempts, adsorbtions, grid_coverage_fractions = dend_run_rsa(
        iterations=iterations,
        dist=dist,
        boxlength=boxlength,
        generations=generations,
        branches=branches,
        branchingnumber=branchingnumber,
        grid_resolution=grid_resolution
    )

    if len(attempts) > 0:
        plt.plot([i for i in range(1,10001)], grid_coverage_fractions, 'g-', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Time')
        plt.ylabel('Coverage Fraction')
        plt.title(f'{generations} Generation Dendrimer RSA')
        plt.grid(True, alpha=0.3)
        plt.show()


        print(f"\nParameters used:")
        print(f"Grid resolution: {grid_resolution}")
        print(f"Box size: {boxlength} x {boxlength}")
        print(f"Particle distance: {dist}")
        print(f"Generations: {generations}")
        print(f"Branches: {branches}")
        print(f"Branching number: {branchingnumber}")

print(len(grid_coverage_fractions[::20]))
