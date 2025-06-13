import time
from heuristics import evaluate
from constants import DEFAULT_MINIMAX_DEPTH, BOARD_SIZE
from move_scores import move_score_manager

class MinimaxAgent:
    def __init__(self, max_depth=DEFAULT_MINIMAX_DEPTH, time_limit=None):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.root_player = None
    
    def ordering_moves(self, state):
        moves = state.get_legal_moves()
        scored_moves = []
        for move in moves:
            new_state = state.copy()
            taken = new_state.make_move(move)  
            if taken is None:
                taken = new_state.count_stones(-state.current_player) - state.count_stones(-state.current_player)
            scored_moves.append((taken, move))
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]

    def minimax(self, state, depth, alpha, beta, maximizingPlayer, max_depth):
        if depth >= max_depth or state.is_game_over():
            return evaluate(state, self.root_player), None

        moves = self.ordering_moves(state)
        if not moves:
            new_state = state.copy()
            new_state.current_player = -new_state.current_player
            return self.minimax(new_state, depth + 1, alpha, beta, not maximizingPlayer, max_depth)

        if maximizingPlayer:
            maxEval = float('-inf')
            best_move = None
            
            if depth == 0 and move_score_manager.is_enabled():
                move_scores = {}
            
            for move in moves:
                new_state = state.copy()
                new_state.make_move(move)
                eval, _ = self.minimax(new_state, depth + 1, alpha, beta, False, max_depth)
                
                if depth == 0 and move_score_manager.is_enabled():
                    move_scores[move] = eval
                
                if eval > maxEval:
                    maxEval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            
            if depth == 0 and move_score_manager.is_enabled():
                move_score_manager.store_move_scores(move_scores, "Minimax")
                
            return maxEval, best_move
        else:
            minEval = float('inf')
            best_move = None
            for move in moves:
                new_state = state.copy()
                new_state.make_move(move)
                eval, _ = self.minimax(new_state, depth + 1, alpha, beta, True, max_depth)
                if eval < minEval:
                    minEval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval, best_move

    def get_move(self, state):
        if not state.get_legal_moves():
            return None
        
        self.root_player = state.current_player
        start_time = time.time()
        best_move = None
        
        for depth in range(1, self.max_depth + 1):
            if self.time_limit is not None:
                if time.time() - start_time > self.time_limit:
                    break
            
            _, move = self.minimax(state, 0, float('-inf'), float('inf'), True, depth)
            if move:
                best_move = move
        return best_move