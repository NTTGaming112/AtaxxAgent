import math
import random
from heuristics import evaluate
from constants import DEFAULT_MCTS_ITERATIONS

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = state.get_legal_moves()
        self.visits = 0
        self.wins = 0
        self.uct_c = 1.41
        self.node_player = state.current_player  

    def ucb1(self, child):
        if child.visits == 0:
            return float('inf')
        return (child.wins / child.visits) + self.uct_c * math.sqrt(math.log(self.visits) / child.visits)
    
    def select_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda c: self.ucb1(c))

    def expand(self):
        move = random.choice(self.untried_moves)
        next_state = self.state.copy()
        next_state.make_move(move)
        child = MCTSNode(state=next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def rollout(self, root_player):
        sim_state = self.state.copy()
        while not sim_state.is_game_over():
            moves = sim_state.get_legal_moves()
            if not moves:
                sim_state.current_player = -sim_state.current_player
                continue
            move = random.choice(moves)
            sim_state.make_move(move)

        return evaluate(sim_state, root_player)

    def backpropagate(self, result, root_player):
        while self is not None:
            self.visits += 1
            if self.node_player == root_player:
                self.wins += result
            else:
                self.wins += 1 - result
            self = self.parent

class MCTSAgent:
    def __init__(self, iterations=DEFAULT_MCTS_ITERATIONS):
        self.iterations = iterations
        self.root = None

    def get_move(self, state):
        self.root = MCTSNode(state)
        root_player = state.current_player

        for _ in range(self.iterations):
            node = self._select(self.root)
            if node.untried_moves:
                node = node.expand()
            result = node.rollout(root_player)
            node.backpropagate(result, root_player)

        if not self.root.children:
            return None
        return max(self.root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0).move

    def _select(self, node):
        while node.untried_moves == [] and node.children:
            child = node.select_child()
            if child is None:
                break
            node = child
        return node