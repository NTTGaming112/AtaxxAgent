from mcts_domain_agent import MCTSDomainAgent
from minimax_agent import MinimaxAgent

class ABMCTSDomainAgent:
    def __init__(self, iterations=600, ab_depth=4, transition_threshold=13, tournament=False):
        self.mcts_agent = MCTSDomainAgent(iterations=iterations, tournament=tournament)
        self.ab_agent = MinimaxAgent(max_depth=ab_depth)
        self.transition_threshold = transition_threshold

    def get_move(self, state):
        empty_cells = state.get_empty_cells()
        
        if empty_cells > self.transition_threshold:
            return self.ab_agent.get_move(state)
        else:
            return self.mcts_agent.get_move(state) 