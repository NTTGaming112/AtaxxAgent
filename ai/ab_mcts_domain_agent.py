from mcts_domain_agent import MCTSDomainAgent
from minimax_agent import MinimaxAgent
from move_scores import move_score_manager

class ABMCTSDomainAgent:
    def __init__(self, iterations=600, ab_depth=4, transition_threshold=13, tournament=True):
        self.mcts_agent = MCTSDomainAgent(iterations=iterations, tournament=tournament)
        self.ab_agent = MinimaxAgent(max_depth=ab_depth)
        self.transition_threshold = transition_threshold

    def get_move(self, state):
        empty_cells = state.get_empty_cells()
        if empty_cells > self.transition_threshold:
            if move_score_manager.is_enabled():
                move_score_manager._agent_name = f"AB+MCTS_Domain(using_Minimax)"
            return self.ab_agent.get_move(state)
        else:
            if move_score_manager.is_enabled():
                move_score_manager._agent_name = f"AB+MCTS_Domain(using_MCTS)"
            return self.mcts_agent.get_move(state) 