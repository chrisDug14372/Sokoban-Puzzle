"""

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script.
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail
for the test of your code. This is not negotiable!

You have to make sure that your code works with the files provided
(search.py and sokoban.py) as your code will be tested
with the original copies of these files.

Last modified by 2020-08-09  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

"""

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search
import sokoban
from itertools import combinations
from itertools import chain


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    """
    return [(5437164, 'Nicholas', 'Scott'), (10572988, 'Christopher', 'Dugdale')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def add_tuple(t1, t2):
    return tuple(v1 + v2 for v1, v2 in zip(t1, t2))


def subtract_tuple(t1, t2):
    return tuple(v1 - v2 for v1, v2 in zip(t1, t2))


def abs_subtract_tuple(t1, t2):
    return tuple(abs(v1 - v2) for v1, v2 in zip(t1, t2))


def manhattan_dist(t1, t2):
    tuple_dist = abs_subtract_tuple(t1, t2)
    cum_dist = 0
    for value in tuple_dist:
        cum_dist += value
    return cum_dist


TRANSLATION_ACTION = {
    (-1, 0): "left",
    "left": (-1, 0),
    (0, 1): "down",
    "down": (0, 1),
    (1, 0): "right",
    "right": (1, 0),
    (0, -1): "up",
    "up": (0, -1),
    (-1, 1): "downleft",
    "downleft": (-1, 1),
    (-1, -1): "upleft",
    "upleft": (-1, -1),
    "downright": (1, 1),
    (1, 1): "downright",
    (1, -1): "upright",
    "upright": (1, -1)
}


def stringify_cells(warehouse, cells):
    """Creates a string containing the rows of cells containing walls as # and
    the specified cells as X.

    @param warehouse:
        a Warehouse object with a worker inside the warehouse

    @param cells:
        a tuple of (x,y) coordinate tuples
    """
    X, Y = zip(*warehouse.walls)
    x_size, y_size = 1 + max(X), 1 + max(Y)

    vis = [[" "] * x_size for y in range(y_size)]
    for (x, y) in warehouse.walls:
        vis[y][x] = "#"
    for (x, y) in cells:
        vis[y][x] = "X"
    return "\n".join(["".join(line) for line in vis])


def inside_warehouse(warehouse):
    """ Creates a tuple of (x,y) tuples of cells located inside the walls of a
    warehouse.

    @param warehouse:
        a Warehouse object with a worker inside the warehouse.

    """
    #This purpose of this function is to determine all of the cells inside the warehouse i.e. reachable cells
    def reachable(current, boxes, walls, path,nexti):
        if current in path or current in walls:
            return path[:len(path)]
        path[nexti] = current
        return (reachable( (current[0], current[1] + 1), boxes, walls, path, nexti + 1),
                reachable( (current[0], current[1] - 1), boxes, walls, path, nexti + 1),
                reachable( (current[0] + 1, current[1]), boxes, walls, path, nexti + 1),
                reachable( (current[0] - 1, current[1]), boxes, walls, path, nexti + 1))

    path = [0 for i in range(warehouse.nrows * warehouse.ncols)]
    valid_paths=reachable(warehouse.worker,warehouse.boxes, warehouse.walls, path,0)

    def print_nested_tuple(T, cells):
        if type(T) == list:
            for b in T:
                cells.append(b)
            return cells
        else:  # scan the top level
            for t in T:
                    # print_nested_tuple(t,cells)
                return tuple(
                    print_nested_tuple(t, cells)
                    for t in T)
    list_cells = print_nested_tuple(valid_paths, [])
        
        #the function returns the contents of the deepest list which contains all reachable cells
    def deepest(cells):
        if type(cells) == list:
            return cells
        return deepest(cells[len(cells) - 1])

    list_cells_final = deepest(list_cells)
    #print(list_cells_final)
    cells_set = set()
            #the function below removes all  doubles ups
    for i in list_cells_final:
        cells_set.add(i)
        
    cells_set.discard(0)
    reachable_cell_list = sorted(cells_set, key=lambda k: [k[1], k[0]])
    return reachable_cell_list




def taboo_cells(warehouse):
    """
    Identify the taboo cells of a warehouse. A "taboo cell" is by definition
    a cell inside a warehouse such that whenever a box get pushed on such
    a cell then the puzzle becomes unsolvable.

    Cells outside the warehouse are not taboo. It is a fail to tag one as taboo.

    When determining the taboo cells, you must ignore all the existing boxes,
    only consider the walls and the target  cells.
    Use only the following rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of
             these cells is a target.

    @param warehouse:
        a Warehouse object with a worker inside the warehouse

    @return
       A string representing the warehouse with only the wall cells marked with
       a '#' and the taboo cells marked with a 'X'.
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.
    """

    reachable_cells = inside_warehouse(warehouse)
    corners = []
    #The below function returns all corners within a warehouse
    for current_cell in reachable_cells:
        if (current_cell[0] + 1, current_cell[1]) in warehouse.walls and (
                current_cell[0], current_cell[1] + 1) in warehouse.walls or (
                current_cell[0], current_cell[1] - 1) in warehouse.walls and (
                current_cell[0] - 1, current_cell[1]) in warehouse.walls or (
                current_cell[0], current_cell[1] - 1) in warehouse.walls and (
                current_cell[0] + 1, current_cell[1]) in warehouse.walls or (
                current_cell[0] - 1, current_cell[1]) in warehouse.walls and (
                current_cell[0], current_cell[1] + 1) in warehouse.walls:
            corners = corners + [current_cell]

    corner_pairs = combinations(corners, 2)
    corner_pairs = list(corner_pairs)

    test_taboo1 = []
    #The below loops returns viable corner pairs, i.e. if they're on the same x or y plane
    for pairs in corner_pairs:
        if pairs[0][0] == pairs[1][0] or pairs[0][1] == pairs[1][1]:
            test_taboo1 = test_taboo1 + [pairs]

    taboo_paths = []
    taboo_paths2 = []

    #The below code creates potential taboo cells by creating the paths between the corners
    for corncomb in range(0, len(test_taboo1)):
        if test_taboo1[corncomb][0][0] == test_taboo1[corncomb][1][0]:
            for b in range(test_taboo1[corncomb][0][1], test_taboo1[corncomb][1][1] + 1):
                taboo_paths = taboo_paths + [(test_taboo1[corncomb][0][0], b)]
        if test_taboo1[corncomb][0][1] == test_taboo1[corncomb][1][1]:
            for b in range(test_taboo1[corncomb][0][0], test_taboo1[corncomb][1][0] + 1):
                taboo_paths = taboo_paths + [(b, test_taboo1[corncomb][0][1])]
        taboo_paths2 = taboo_paths2 + [taboo_paths]
        taboo_paths = []

    count = 0
    taboo_paths3 = []
    #The below code tests to see if the potential paths include targets, if so then only the corners in the paths are taboo (unless the corners are targets)
    for b in taboo_paths2:
        for i in warehouse.targets:
            if i not in b:
                count = count + 1
        if count == len(warehouse.targets):
            taboo_paths3 = taboo_paths3 + [b]
        count = 0

    path_left = []
    path_right = []
    path_up = []
    path_down = []
    in_walls_u = True
    in_walls_d = True
    in_walls_l = True
    in_walls_r = True
    taboo_paths_el = []
    taboo_paths = []
    #The loops below create shifts of the potential taboo paths, either up, and down if they're on the same y plane or left and right if they're on the same x plane and tests to see
    #if all elements in the path exist in walls, if so, then they are taboo cells
    for potential_paths in taboo_paths3:
        if potential_paths[0][0] == potential_paths[1][0]:
            for path_length_index in range(0, len(potential_paths)):
                path_left = path_left + [(potential_paths[0][0] - 1, potential_paths[path_length_index][1])]
                path_right = path_right + [(potential_paths[0][0] + 1, potential_paths[path_length_index][1])]
            for path_left_cells in path_left:
                if path_left_cells in warehouse.walls and in_walls_l is True:
                    in_walls_l = True
                else:
                    in_walls_l = False
            if in_walls_l is True:
                taboo_paths_el = taboo_paths_el + potential_paths
            for path_right_cells in path_right:
                if path_right_cells in warehouse.walls and in_walls_r is True:
                    in_walls_r = True
                else:
                    in_walls_r = False
            if in_walls_r is True:
                taboo_paths_el = taboo_paths_el + potential_paths
            in_walls_r = True
            in_walls_l = True
            path_left = []
            path_right = []

        if potential_paths[0][1] == potential_paths[1][1]:
            for path_length_index in range(0, len(potential_paths)):
                path_up = path_up + [(potential_paths[path_length_index][0], potential_paths[0][1] - 1)]
                path_down = path_down + [(potential_paths[path_length_index][0], potential_paths[0][1] + 1)]
            for path_up_cell in path_up:
                if path_up_cell in warehouse.walls and in_walls_u is True:
                    in_walls_u = True
                else:
                    in_walls_u = False
            if in_walls_u is True:
                taboo_paths_el = taboo_paths_el + potential_paths
            for path_down_cell in path_down:
                if path_down_cell in warehouse.walls and in_walls_d is True:
                    in_walls_d = True
                else:
                    in_walls_d = False
            if in_walls_d == True:
                taboo_paths_el = taboo_paths_el + potential_paths
            path_up = []
            path_down = []
            in_walls_d = True
            in_walls_u = True
        taboo_paths = taboo_paths + taboo_paths_el
        taboo_paths_el = []

    taboo_paths2 = []
    #The below loop checks the current taboo paths and removes the paths which include targets
    #print(taboo_paths)
    for path_cel in taboo_paths:
        if path_cel not in warehouse.targets and path_cel not in warehouse.walls:
            taboo_paths2 = taboo_paths2 + [path_cel]
    #Corners are added if the are not targets        
    for corner in corners:
        if corner not in warehouse.targets:
            taboo_paths2 = taboo_paths2 + [corner]
    #Set of taboo cells created        
    taboo_paths2 = set(taboo_paths2)
    taboo_paths2 = sorted(taboo_paths2, key=lambda k: [k[1], k[0]])
    return stringify_cells(warehouse, taboo_paths2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class LessWarehouse(sokoban.Warehouse):
    """This is a subclass of Warehouse that creates a duplicate of the Warehouse
    object passed to it, but with the __lt__ function defined as always True. This
    should allow it to be used in a heap queue."""

    def __init__(self, warehouse):
        super().from_string(warehouse.__str__())

    def __lt__(self, warehouse):
        return True

    def copy(self, worker=None, boxes=None):
        return LessWarehouse(super().copy(worker, boxes))


class SokobanPuzzle(search.Problem):
    """
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of
    the provided module 'search.py'.

    """

    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' method is needed
    #     to satisfy the interface of 'search.Problem'.
    #
    #     You are allowed (and encouraged) to use auxiliary functions and classes

    def __init__(self, warehouse):
        super().__init__(LessWarehouse(warehouse), None)
        self.allow_taboo_push = True
        self.macro = False
        self.weighted = False
        self.taboo_cells = []
        self.box_costs = []

    def set_taboo_cells(self):
        self.allow_taboo_push = False
        taboo_cells_string = taboo_cells(self.initial)
        taboo_lines = taboo_cells_string.split(sep='\n')
        for y in range(self.initial.nrows):
            for x in range(self.initial.ncols):
                if taboo_lines[y][x] == "X":
                    self.taboo_cells.append((x, y))

    def set_macro(self):
        self.macro = True

    def set_path_cost(self,box_costs):
        self.weighted = True
        self.box_costs = box_costs

    def path_cost(self, c, state1, action, state2):
        if self.weighted is False:
            return c + 1
        else:
            if state1.boxes == state2.boxes:
                return c + 1
            else:
                for i in range(len(state1.boxes)):
                    if state2.worker == state1.boxes[i]:
                        return c + self.box_costs[i]

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        valid_actions = []

        if self.macro is False:
            for action in ["left", "right", "up", "down"]:
                worker_target = add_tuple(state.worker, TRANSLATION_ACTION[action])
                if worker_target in state.walls:
                    continue
                else:
                    if worker_target in state.boxes:
                        box_target = add_tuple(worker_target, TRANSLATION_ACTION[action])
                        if (box_target in state.boxes or
                                box_target in state.walls):
                            continue
                        else:
                            if box_target in self.taboo_cells and self.allow_taboo_push is False:
                                continue
                            else:
                                valid_actions.append(action)
                    else:
                        valid_actions.append(action)
        else:
            for box in state.boxes:
                for action in ["left", "right", "up", "down"]:
                    
                    if can_go_there(state, subtract_tuple(box, TRANSLATION_ACTION[action])):
                        box_target = add_tuple(box, TRANSLATION_ACTION[action])
                        if box_target in state.boxes or box_target in state.walls:
                            continue
                        else:
                            if box_target in self.taboo_cells and self.allow_taboo_push is False:
                                continue
                            else:
                                valid_actions.append((box, action))
        return valid_actions

    def result(self, state, action):
        if self.macro is False:
            worker = add_tuple(state.worker, TRANSLATION_ACTION[action])
            boxes = state.boxes.copy()
            if worker in boxes:
                boxes[boxes.index(worker)] = add_tuple(worker, TRANSLATION_ACTION[action])
            return state.copy(worker, boxes)
        else:
            worker = action[0]
            boxes = state.boxes.copy()
            boxes[boxes.index(worker)] = add_tuple(worker, TRANSLATION_ACTION[action[1]])
        return state.copy(worker, boxes)

    def goal_test(self, state):
        for box in state.boxes:
            if box not in state.targets:
                return False
            #else:
              #  return True
        return True

    def h(self, node):
        target_distance = 0
        for box in node.state.boxes:
            box_target_distance = []
            for target in node.state.targets:
                box_target_distance.append(manhattan_dist(box, target))
            target_distance += min(box_target_distance)
        return target_distance


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    """

    Determine if the sequence of actions listed in 'action_seq' is legal or not.

    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.

    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']

    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    """

    ##         "INSERT YOUR CODE HERE"

    for action in action_seq:
        worker_target = add_tuple(warehouse.worker, TRANSLATION_ACTION[action])
        if worker_target in warehouse.walls:
            return "Impossible"
        elif worker_target in warehouse.boxes:
            box_target = add_tuple(worker_target, TRANSLATION_ACTION[action])
            if box_target in warehouse.boxes or box_target in warehouse.walls:
                return "Impossible"
            else:
                warehouse.worker = worker_target
                while worker_target in warehouse.boxes: warehouse.boxes.remove(worker_target)
                warehouse.boxes.append(box_target)
        else:
            warehouse.worker = worker_target
    return warehouse.str()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def action_extract(node):
    action_list = []

    while node.parent is not None:
        action_list.append(node.action)
        node = node.parent
    action_list = action_list[::-1]
    return action_list

def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using A* graph search algorithm and elementary actions
    the puzzle defined in the parameter 'warehouse'.
    
    In this scenario, the cost of all (elementary) actions is one unit.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''

    ##         "INSERT YOUR CODE HERE"

    puzzle = SokobanPuzzle(warehouse)
    puzzle.set_taboo_cells()
    astar_search = search.astar_graph_search(puzzle, lambda n: n.path_cost + puzzle.h(n))
    if astar_search is None:
        return "Impossible"
    else:
        return action_extract(astar_search)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# class CellWalkability(CellReachability):
#    '''Class for determining if a worker can walk to a given location inside
#    the warehouse without pushing any boxes. Subclass of CellReachability with
#    different allowable actions.'''
#
#    def actions(self, state):
#        valid_actions = []
#        for action in ["left","right","up","down"]:
#            if (self.result(state,action) not in self.warehouse.walls and
#            self.result(state,action) not in self.warehouse.boxes):
#                valid_actions.append(action)
#        return valid_actions

def can_go_there(warehouse, dst):
    """
    Determine whether the worker can walk to the cell dst=(row,column)
    without pushing any box.

    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
      
    """
    if dst == warehouse.worker:
        return True
    #The below recursive functions returns a boolean of the possisble paths in a warehouse starting
    #at the worker and aiming for a destination, false is provided if a wall or box is hit in the path, 
    #true is returned if we get to the destination without hitting a box or wall
    def can_go(boxes, current, target, path, nexti, cgt, walls):
        if current in path or current in walls or current in boxes:
            cgt = False
            return cgt
        if current == target:
            cgt = True
            return cgt
        path[nexti] = current
        
        return (can_go(boxes, (current[0], current[1] + 1), target, path, nexti + 1, cgt,walls),
                can_go(boxes, (current[0], current[1] - 1), target, path, nexti + 1, cgt,walls),
                can_go(boxes, (current[0] + 1, current[1]), target, path, nexti + 1, cgt,walls),
                can_go(boxes, (current[0] - 1, current[1]), target, path, nexti + 1, cgt,walls))

    def print_cgt(r, c, boxes, current, target, walls):
        path = [0 for i in range(r * c)]
        cgt = False
        return can_go(boxes, current, target, path, 0, cgt, walls)
    cgt_lists = print_cgt(warehouse.nrows, warehouse.ncols, warehouse.boxes, warehouse.worker, dst, warehouse.walls)

    def print_nested_tuple(T, cells):
        if (type(T) == bool):
            cells.append(T)
            return cells
        else:  # scan the top level
            return tuple(
                print_nested_tuple(t, cells)
                for t in T)

    cgt_list = chain(*print_nested_tuple(cgt_lists, []))
    if True in cgt_list:
        return True
    else:
        return False





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    """
    Solve using using A* algorithm and macro actions the puzzle defined in
    the parameter 'warehouse'.

    A sequence of macro actions should be represented by a list M of the form

            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]

    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ]
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.

    In this scenario, the cost of all (macro) actions is one unit.

    @param warehouse: a valid Warehouse object

    @return
        If the puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    """

    #         "INSERT YOUR CODE HERE"

    puzzle = SokobanPuzzle(warehouse)
    puzzle.set_taboo_cells()
    puzzle.set_macro()
    astar_search = search.astar_graph_search(puzzle, lambda n: n.path_cost + puzzle.h(n))
    if astar_search is None:
        return "Impossible"
    else:
        return action_extract(astar_search)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban_elem(warehouse, push_costs):
    """
    In this scenario, we assign a pushing cost to each box, whereas for the
    functions 'solve_sokoban_elem' and 'solve_sokoban_macro', we were
    simply counting the number of actions (either elementary or macro) executed.

    When the worker is moving without pushing a box, we incur a
    cost of one unit per step. Pushing the ith box to an adjacent cell
    now costs 'push_costs[i]'.

    The ith box is initially at position 'warehouse.boxes[i]'.

    This function should solve using A* algorithm and elementary actions
    the puzzle 'warehouse' while minimizing the total cost described above.

    @param
     warehouse: a valid Warehouse object
     push_costs: list of the weights of the boxes (pushing cost)

    @return
        If puzzle cannot be solved return 'Impossible'
        If a solution exists, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    """

    puzzle = SokobanPuzzle(warehouse)
    puzzle.set_taboo_cells()
    puzzle.set_path_cost(push_costs)
    astar_search = search.astar_graph_search(puzzle, lambda n: n.path_cost + puzzle.h(n))
    if astar_search is None:
        return "Impossible"
    else:

        return action_extract(astar_search)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

warehouse = sokoban.Warehouse()
warehouse.load_warehouse("./warehouses/warehouse_03.txt")
#print(taboo_cells(warehouse))
# print(can_go_there(warehouse,(7,2))

import time

t0 = time.time()
test_solve = solve_sokoban_elem(warehouse)
print(test_solve)
t1 = time.time()
print(t1 - t0)



#test_solve2 = solve_weighted_sokoban_elem(warehouse,(1,10))
#print(test_solve2)#
#
#if test_solve == test_solve2:
#    print ("SAME")
#else:
#    print("DIFF")
#test_macro = solve_sokoban_macro(warehouse)
#print(test_macro)

#print(SokobanPuzzle(warehouse))