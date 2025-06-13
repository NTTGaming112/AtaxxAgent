#!/usr/bin/env python3

import asyncio
import argparse
import platform
import pandas as pd
import numpy as np
import os
from datetime import datetime

from ataxx_state import AtaxxState
from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
from mcts_domain_agent import MCTSDomainAgent
from ab_mcts_domain_agent import ABMCTSDomainAgent
from move_scores import move_score_manager


class TournamentRunner:
    def __init__(self, map_file=None, games_per_match=5, iterations=300, 
                 algo1="MCTS_Domain_600", algo2="Minimax+AB", 
                 delay=0.5, first_player="W", use_tournament=False,
                 transition_threshold=13, depths=4):
        
        self.map_file = map_file
        self.games_per_match = games_per_match
        self.iterations = iterations
        self.algo1 = algo1
        self.algo2 = algo2
        self.delay = delay
        self.first_player = 1 if first_player == "B" else -1
        self.use_tournament = use_tournament
        self.transition_threshold = transition_threshold
        self.depths = depths
        
        self.running = True
        self.paused = False
        
        self.setup_game()
        
        self.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                       for name in [self.algo1, self.algo2]}
        
        print(f"🎮 Tournament Setup Complete")
        print(f"   Map: {self.map_file or 'Default'}")
        print(f"   Games per match: {self.games_per_match}")
        print(f"   First player: {'Black (X)' if self.first_player == 1 else 'White (O)'}")
        print(f"   Agents: {self.algo1} vs {self.algo2}")

    def setup_game(self):
        map_dir = "map"
        if os.path.exists(map_dir):
            self.available_maps = [f for f in os.listdir(map_dir) if f.endswith('.txt')]
        
        if self.map_file and self.map_file in self.available_maps:
            self.initial_board = self.load_map_from_file(self.map_file)
            print(f"📍 Loaded map: {self.map_file}")
        else:
            self.initial_board = self.get_default_board()
            print("📍 Using default map")
        
        self.agents = {}
        
        for algo_name in [self.algo1, self.algo2]:
            if algo_name == "Minimax" or algo_name == "Minimax+AB":
                self.agents[algo_name] = MinimaxAgent(max_depth=self.depths)
            elif algo_name == "MCTS":
                self.agents[algo_name] = MCTSAgent(iterations=self.iterations)
            elif "AB+MCTS_Domain" in algo_name:
                iter_count = int(algo_name.split('_')[-1]) if '_' in algo_name else self.iterations
                self.agents[algo_name] = ABMCTSDomainAgent(
                    iterations=iter_count,
                    transition_threshold=self.transition_threshold,
                    ab_depth=self.depths,
                    tournament=self.use_tournament
                )
            elif "MCTS_Domain" in algo_name:
                iter_count = int(algo_name.split('_')[-1]) if '_' in algo_name else self.iterations
                self.agents[algo_name] = MCTSDomainAgent(
                    iterations=iter_count, 
                    tournament=self.use_tournament
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")

    def load_map_from_file(self, filename):
        try:
            map_path = os.path.join("map", filename)
            with open(map_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            
            board = np.zeros((7, 7), dtype=int)
            for r, line in enumerate(lines[:7]):
                for c, char in enumerate(line.strip()[:7]):
                    if char == 'B':
                        board[r][c] = 1
                    elif char == 'W':
                        board[r][c] = -1
                    elif char == '#':
                        board[r][c] = 0
            
            return board
            
        except FileNotFoundError:
            print(f"Warning: Map file 'map/{filename}' not found, using default map")
            return self.get_default_board()
        except Exception as e:
            print(f"Warning: Error reading map file: {e}, using default map")
            return self.get_default_board()
    
    def get_default_board(self):
        """Create default Ataxx board layout"""
        board = np.zeros((7, 7), dtype=int)
        board[0, 0] = 1   
        board[6, 6] = 1   
        board[0, 6] = -1  
        board[6, 0] = -1  
        
        return board

    async def play_game(self, agent1_name, agent2_name, forward=True):
        """Play a single game between two agents"""
        move_score_manager.clear_scores()
        state = AtaxxState(initial_board=self.initial_board, current_player=self.first_player)
        
        if forward:
            current_x_player = agent1_name
            current_o_player = agent2_name
        else:
            current_x_player = agent2_name
            current_o_player = agent1_name
        
        if self.map_file and self.map_file in self.available_maps:
            map_idx = self.available_maps.index(self.map_file)
            map_name = f"Map {map_idx}: {self.map_file.replace('.txt', '')}"
        else:
            map_name = "Map 1: Default"
            
        print(f"\nGame ({'Forward' if forward else 'Reverse'}) on {map_name}")
        print(f"X (Red): {current_x_player} | O (Blue): {current_o_player}")
        
        print(f"\nInitial board:")
        state.display_board()
        
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            print(f"Warning: No initial legal moves for player {state.current_player}")
        
        move_count = 0
        x_pieces, o_pieces = 0, 0

        while not state.is_game_over() and self.running:
            if not self.running:
                break
            
            if self.paused:
                await asyncio.sleep(0.1)
                continue
                
            legal_moves = state.get_legal_moves()
            
            if not legal_moves:
                print(f"No legal moves for player {state.current_player} - PASS")
                state.current_player = -state.current_player
                continue
            
            if state.current_player == 1: 
                current_agent_name = current_x_player
            else:  
                current_agent_name = current_o_player
            
            agent = self.agents[current_agent_name]
            
            move_score_manager.enable_score_collection(current_agent_name)
            try:
                move = agent.get_move(state)
            finally:
                move_score_manager.disable_score_collection()

            if not self.running:
                return None

            if move:
                r, c, nr, nc = move
                is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1
                move_type = "Clone" if is_clone else "Jump"
                player_symbol = "X" if state.current_player == 1 else "O"
                print(f"\nMove {move_count + 1}: {current_agent_name} ({player_symbol}) moves from ({r},{c}) to ({nr},{nc}) ({move_type})")
                
                state.make_move(move)
                move_count += 1
                
                print(f"\nBoard after move {move_count}:")
                state.display_board()
                
                x_pieces = np.sum(state.board == 1)
                o_pieces = np.sum(state.board == -1)
                print(f"Pieces - X: {x_pieces}, O: {o_pieces}")
                
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
            else:
                print(f"\n{current_agent_name} has no legal moves - PASS")
                state.current_player = -state.current_player
        
        if not self.running:
            return None
            
        winner = state.get_winner()
        
        x_agent = current_x_player  
        o_agent = current_o_player
        
        self.results[x_agent]["avg_pieces"] += x_pieces
        self.results[o_agent]["avg_pieces"] += o_pieces
        self.results[x_agent]["games_played"] += 1
        self.results[o_agent]["games_played"] += 1
        
        if winner == 1:  
            winner_name = x_agent
            loser_name = o_agent
            self.results[x_agent]["wins"] += 1
            self.results[o_agent]["losses"] += 1
            print(f"Winner: {winner_name} (X)")
        elif winner == -1:  
            winner_name = o_agent
            loser_name = x_agent
            self.results[o_agent]["wins"] += 1
            self.results[x_agent]["losses"] += 1
            print(f"Winner: {winner_name} (O)")
        else:
            self.results[x_agent]["draws"] += 1
            self.results[o_agent]["draws"] += 1
            winner_name = None
            loser_name = None
            print("Draw")
        
        move_score_manager.clear_scores()

        return {
            'winner': winner,
            'winner_name': winner_name,
            'loser_name': loser_name,
            'x_pieces': x_pieces,
            'o_pieces': o_pieces,
            'move_count': move_count
        }

    async def run_tournament(self):
        """Run complete tournament"""
        move_score_manager.clear_scores()
        print(f"\n🏆 Starting Tournament: {self.algo1} vs {self.algo2}")
        print(f"📋 {self.games_per_match} games each way (total: {self.games_per_match * 2} games)")
        
        self.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                       for name in [self.algo1, self.algo2]}
        
        print(f"\n🔴 Round 1: {self.algo1} (X) vs {self.algo2} (O)")
        for game_num in range(self.games_per_match):
            if not self.running:
                break
            print(f"Game {game_num + 1}/{self.games_per_match}")
            move_score_manager.clear_scores()
            result = await self.play_game(self.algo1, self.algo2, forward=True)
            if not self.running or result is None:
                break
        
        if not self.running:
            return
                
        print(f"\n🔵 Round 2: {self.algo2} (X) vs {self.algo1} (O)")
        for game_num in range(self.games_per_match):
            if not self.running:
                break
            print(f"Game {game_num + 1}/{self.games_per_match}")
            move_score_manager.clear_scores()
            result = await self.play_game(self.algo1, self.algo2, forward=False)
            if not self.running or result is None:
                break
        
        if not self.running:
            return
                
        print(f"\n🏁 Tournament Results ({self.algo1} vs {self.algo2}):")
        
        self.validate_results()
        
        for name in self.results:
            if self.results[name]["games_played"] > 0:
                wins = self.results[name]['wins']
                losses = self.results[name]['losses']
                draws = self.results[name]['draws']
                avg_pieces = self.results[name]['avg_pieces'] / self.results[name]['games_played']
                total_games = wins + losses + draws
                win_rate = (wins / total_games * 100) if total_games > 0 else 0
                
                result_text = (f"{name}: {wins}W-{losses}L-{draws}D "
                             f"({win_rate:.1f}% win rate, {avg_pieces:.2f} avg pieces)")
                print(f"  {result_text}")
        
        move_score_manager.clear_scores()
        self.save_results()

    def validate_results(self):
        """Validate tournament results for consistency"""
        total_games_played = sum(self.results[agent]["games_played"] for agent in self.results)
        expected_games = self.games_per_match * 2 * len(self.results)
        
        if total_games_played != expected_games:
            print(f"⚠️  Warning: Expected {expected_games} total games, but recorded {total_games_played}")
        
        for agent in self.results:
            agent_total = (self.results[agent]["wins"] + 
                          self.results[agent]["losses"] + 
                          self.results[agent]["draws"])
            if agent_total != self.results[agent]["games_played"]:
                print(f"⚠️  Warning: {agent} results inconsistent - {agent_total} vs {self.results[agent]['games_played']}")
        
        return True

    def save_results(self):
        """Save tournament results to CSV file"""
        data = []
        
        if self.map_file and self.map_file in self.available_maps:
            map_idx = self.available_maps.index(self.map_file)
            map_name = f"Map_{map_idx}_{self.map_file.replace('.txt', '')}"
        else:
            map_name = "Map_1_Default"
            
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for name in self.results:
            if self.results[name]["games_played"] > 0:
                avg_pieces = self.results[name]['avg_pieces'] / self.results[name]['games_played']
                data.append({
                    'Agent': name,
                    'Opponent': self.algo2 if name == self.algo1 else self.algo1,
                    'Wins': self.results[name]['wins'],
                    'Losses': self.results[name]['losses'],
                    'Draws': self.results[name]['draws'],
                    'TotalGames': self.results[name]['games_played'],
                    'Map': map_name,
                    'Depth': self.depths,
                    'Iterations': self.iterations,
                })
        
        new_df = pd.DataFrame(data)
        output_path = '/kaggle/working/results.csv' if os.path.exists('/kaggle/working') else 'results.csv'
        
        if platform.system() == "Emscripten":
            print("Pyodide: Cannot save CSV. Results:")
            print(new_df.to_string(index=False))
            return new_df
        
        try:
            if os.path.exists(output_path):
                print(f"📖 Reading existing results from {output_path}")
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"📊 Added {len(new_df)} new records to existing {len(existing_df)} records")
            else:
                print(f"📝 Creating new results file {output_path}")
                combined_df = new_df
            
            combined_df.to_csv(output_path, index=False)
            print(f"💾 Saved {len(combined_df)} total records to {output_path}")
            
            print(f"\n📈 Latest Match Results ({map_name}):")
            for _, row in new_df.iterrows():
                win_rate = (row['Wins'] / row['TotalGames'] * 100) if row['TotalGames'] > 0 else 0
                print(f"  {row['Agent']}: {row['Wins']}W-{row['Losses']}L-{row['Draws']}D "
                    f"({win_rate:.1f}% win rate, {row['AvgPieces']} avg pieces)")
            
            if len(combined_df) > len(new_df):
                print(f"\n📊 Historical Summary (All Matches):")
                historical_summary = combined_df.groupby('Agent').agg({
                    'Wins': 'sum',
                    'Losses': 'sum', 
                    'Draws': 'sum',
                    'TotalGames': 'sum',
                    'AvgPieces': 'mean'
                }).round(2)
                
                for agent, stats in historical_summary.iterrows():
                    total_games = stats['TotalGames']
                    win_rate = (stats['Wins'] / total_games * 100) if total_games > 0 else 0
                    print(f"  {agent}: {int(stats['Wins'])}W-{int(stats['Losses'])}L-{int(stats['Draws'])}D "
                        f"({win_rate:.1f}% win rate, {stats['AvgPieces']:.2f} avg pieces, {int(total_games)} total games)")
            
            return combined_df
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print("Fallback - Results printed to console:")
            print(new_df.to_string(index=False))
            return new_df


def parse_args():
    parser = argparse.ArgumentParser(description="Ataxx Tournament Runner")
    parser.add_argument("--map_file", type=str, default=None, help="Map file in 'map/' directory")
    parser.add_argument("--games", type=int, default=5, help="Games per match")
    parser.add_argument("--iterations", type=int, default=300, help="MCTS iterations")
    parser.add_argument("--algo1", type=str, default="MCTS_Domain_600", help="First agent")
    parser.add_argument("--algo2", type=str, default="Minimax+AB", help="Second agent")
    parser.add_argument("--delay", type=float, default=0, help="Delay per move (seconds)")
    parser.add_argument("--first_player", type=str, default="W", help="First player (W=White/O or B=Black/X)")
    parser.add_argument("--use_tournament", type=bool, default=False, help="Use tournament selection for MCTS Domain")
    parser.add_argument("--transition_threshold", type=int, default=13, help="Transition threshold for AB+MCTS Domain")
    parser.add_argument("--depths", type=int, default=4, help="Search depths for Minimax and AB")
    return parser.parse_args()


async def main():
    args = parse_args()
    try:
        runner = TournamentRunner(
            map_file=args.map_file,
            games_per_match=args.games,
            iterations=args.iterations,
            algo1=args.algo1,
            algo2=args.algo2,
            delay=args.delay,
            first_player=args.first_player,
            use_tournament=args.use_tournament,
            transition_threshold=args.transition_threshold,
            depths=args.depths
        )
        await runner.run_tournament()
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Tournament interrupted by user")
        exit(0)


if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
