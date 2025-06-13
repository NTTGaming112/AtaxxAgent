import random
import numpy as np
from heuristics import evaluate, heuristic
from constants import DEFAULT_MCTS_DOMAIN_ITERATIONS
from mcts_agent import MCTSNode
from move_scores import move_score_manager
import math

class MCTSDomainNode(MCTSNode):
    def __init__(self, state, parent=None, move=None):
        super().__init__(state, parent, move)
    
    def select_child(self):
        return super().select_child()

    def heuristic_expand(self): 
        if not self.untried_moves:
            return None

        scored_moves = [(m, heuristic(m, self.state, self.state.current_player)) for m in self.untried_moves]
        probabilities = self.softmax([score for _, score in scored_moves], temperature=0.8)
        move = random.choices([m for m, _ in scored_moves], weights=probabilities)[0]
        self.untried_moves.remove(move)
        next_state = self.state.copy()
        next_state.make_move(move)
        child = MCTSDomainNode(state=next_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    
    def heuristic_rollout(self, root_player):
        sim_state = self.state.copy()
        while not sim_state.is_game_over():
            moves = sim_state.get_legal_moves()
            if not moves:
                sim_state.current_player = -sim_state.current_player
                continue

            scored_moves = [(m, heuristic(m, sim_state, sim_state.current_player)) for m in moves]
            scores = [s for _, s in scored_moves]
            probs = self.softmax(scores, temperature=0.4)
            move = random.choices([m for m, _ in scored_moves], weights=probs)[0]
            sim_state.make_move(move)

        return evaluate(sim_state, root_player)

    
    def softmax(self, scores, temperature=1.0):
        scores = np.array(scores)
        scores = scores - np.max(scores)  
        exp_scores = np.exp(scores / temperature)
        sum_exp = np.sum(exp_scores)
        if sum_exp == 0:  
            return np.ones_like(scores) / len(scores) 
        return exp_scores / sum_exp

    def tournament_rollout(self, root_player, tournament_params):
        moves = self.state.get_legal_moves()
        if not moves:
            return evaluate(self.state, root_player)
        
        scores = [heuristic(move, self.state, self.state.current_player) for move in moves]
        threshold_r1 = 0.4 if len(moves) > 10 else 0.2
        max_idx = np.argmax(scores)
        if len(moves) > 1 and scores[max_idx] - sorted(scores)[-2] > threshold_r1:
            return scores[max_idx]
        
        top_k = min(tournament_params[0]['top_k'], len(moves))
        probs = self.softmax(scores, temperature=0.8) 
        top_indices = np.random.choice(len(moves), size=top_k, p=probs, replace=False)
        top_moves_r1 = [moves[i] for i in top_indices]
        
        move_scores_r2 = []
        sim_state = self.state.copy()  
        for i, move in enumerate(top_moves_r1):
            adaptive_sims = tournament_params[1]['num_sim'] + (len(top_moves_r1) - i - 1) * 10
            total_score = 0
            simulations_run = 0
            scores_list = []
            
            for sim_idx in range(adaptive_sims):
                sim_state_copy = sim_state.copy()
                sim_state_copy.make_move(move)
                
                while not sim_state_copy.is_game_over():
                    legal_moves = sim_state_copy.get_legal_moves()
                    if not legal_moves:
                        sim_state_copy.current_player = -sim_state_copy.current_player
                        continue
                    scores = [heuristic(m, sim_state_copy, sim_state_copy.current_player) for m in legal_moves]
                    probs = self.softmax(scores, temperature=0.3)  
                    best_move = legal_moves[np.random.choice(len(legal_moves), p=probs)]
                    sim_state_copy.make_move(best_move)
                
                score = evaluate(sim_state_copy, root_player)
                total_score += score
                scores_list.append(score)
                simulations_run += 1
                
                if sim_idx >= 8 and len(scores_list) > 1:
                    current_avg = total_score / simulations_run
                    prev_avg = (total_score - score) / (simulations_run - 1) if simulations_run > 1 else current_avg
                    if sim_idx >= 20 and abs(current_avg - prev_avg) < 0.05:
                        break
            
            avg_score = total_score / simulations_run if simulations_run > 0 else 0
            move_scores_r2.append((move, avg_score, simulations_run))
        
        move_scores_r2.sort(key=lambda x: x[1], reverse=True)
        if len(move_scores_r2) > 1 and move_scores_r2[0][1] - move_scores_r2[1][1] > 0.3:
            return move_scores_r2[0][1]
        
        top_k_r2 = min(tournament_params[1]['top_k'], len(move_scores_r2))
        top_moves_r2 = [move for move, _, _ in move_scores_r2[:top_k_r2]]
        move_scores_r3 = []
        
        for move in top_moves_r2:
            total_score = 0
            simulations_run = 0
            scores_list = []
            sim_state = self.state.copy()
            
            for sim_idx in range(tournament_params[2]['num_sim']):
                sim_state_copy = sim_state.copy()
                sim_state_copy.make_move(move)
                
                while not sim_state_copy.is_game_over():
                    legal_moves = sim_state_copy.get_legal_moves()
                    if not legal_moves:
                        sim_state_copy.current_player = -sim_state_copy.current_player
                        continue
                    scores = [heuristic(m, sim_state_copy, sim_state_copy.current_player) for m in legal_moves]
                    probs = self.softmax(scores, temperature=0.3) 
                    best_move = legal_moves[np.random.choice(len(legal_moves), p=probs)]
                    sim_state_copy.make_move(best_move)
                
                score = evaluate(sim_state_copy, root_player)
                total_score += score
                scores_list.append(score)
                simulations_run += 1
            
            avg_score = total_score / simulations_run if simulations_run > 0 else 0
            move_scores_r3.append((move, avg_score, simulations_run))
        
        if move_scores_r3:
            return max(move_scores_r3, key=lambda x: x[1])[1]
        elif move_scores_r2:
            return max(move_scores_r2, key=lambda x: x[1])[1]
        else:
            return max(scores)

class MCTSDomainAgent:
    def __init__(self, iterations=DEFAULT_MCTS_DOMAIN_ITERATIONS, tournament_params=None, tournament=False):
        self.iterations_per_round = iterations * 6 // 10 if tournament else iterations
        tournament_total = iterations * 4 // 10
        
        self.tournament_params = tournament_params if tournament_params else [
            {'num_sim': tournament_total, 'top_k': 5},      
            {'num_sim': tournament_total * 3 // 10, 'top_k': 3},     
            {'num_sim': tournament_total * 7 // 10, 'top_k': 1}  
        ]
        self.tournament = tournament

    def get_move(self, state):
        self.root = MCTSDomainNode(state)
        root_player = state.current_player
        
        for iteration in range(self.iterations_per_round):
            node = self._select(self.root)
            if node.untried_moves:
                node = node.heuristic_expand()
            if self.tournament:
                result = node.tournament_rollout(root_player, self.tournament_params)
            else:
                result = node.heuristic_rollout(root_player)
            node.backpropagate(result, root_player)

        if not self.root.children:
            return None
        
        best_move = max(self.root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0).move
        
        if move_score_manager.is_enabled():
            move_scores = {}
            for child in self.root.children:
                if child.visits > 0:
                    win_rate = child.wins / child.visits
                    move_scores[child.move] = win_rate
            move_score_manager.store_move_scores(move_scores, "MCTS_Domain")

        return best_move
    
    def _select(self, node):
        while node.untried_moves == [] and node.children:
            child = node.select_child()
            if child is None:
                break
            node = child
        return node