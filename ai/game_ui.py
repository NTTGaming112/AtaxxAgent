#!/usr/bin/env python3
"""
Game UI for Ataxx Tournament
Handles pygame interface, terminal display, and result visualization
"""

import asyncio
import argparse
import platform
import numpy as np
import os

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available - terminal mode only")

from ataxx_state import AtaxxState
from tournament_runner import TournamentRunner

# Colors for pygame interface
COLORS = {
    'bg': (35, 39, 42),
    'panel': (52, 58, 64),
    'panel_light': (73, 80, 87),
    'grid': (108, 117, 125),
    'text': (248, 249, 250),
    'text_dark': (33, 37, 41),
    'text_black': (0, 0, 0),
    'accent': (13, 110, 253),
    'success': (25, 135, 84),
    'warning': (255, 193, 7),
    'player_x': (220, 53, 69),
    'player_o': (13, 202, 240),
    'empty': (173, 181, 189),
    'blocked': (52, 58, 64)
}


class AtaxxGameUI:
    def __init__(self, map_file=None, games_per_match=5, iterations=300, 
                 algo1="MCTS_Domain_600", algo2="Minimax+AB", 
                 display="pygame", delay=0.5, first_player="W", 
                 use_tournament=False, transition_threshold=13, depths=4):
        
        self.display = display
        self.delay = delay
        
        # Initialize tournament runner
        self.tournament = TournamentRunner(
            map_file=map_file,
            games_per_match=games_per_match,
            iterations=iterations,
            algo1=algo1,
            algo2=algo2,
            delay=delay,
            first_player=first_player,
            use_tournament=use_tournament,
            transition_threshold=transition_threshold,
            depths=depths
        )
        
        # UI state
        self.menu_active = True
        self.running = True
        self.paused = False
        
        # Menu configuration state
        self.selected_algo1 = algo1
        self.selected_algo2 = algo2
        self.selected_games = games_per_match
        self.selected_iterations = iterations
        self.selected_delay = delay
        self.selected_first_player = 1 if first_player == "X" else -1
        self.selected_depths = depths
        self.selected_transition = transition_threshold
        self.selected_map = "position_00"
        self.selected_use_tournament = use_tournament  # Add tournament selection setting
        
        # Available options
        self.available_agents = [
            "Minimax+AB", "MCTS", "MCTS_Domain_600", "AB+MCTS_Domain_600"
        ]
        # Load available maps from map directory
        self.available_maps = get_available_maps()
        self.available_games = [1, 3, 5, 10, 20, 50]
        self.available_iterations = [100, 300, 600, 1200, 2400]
        self.available_delays = [0, 0.1, 0.5, 1.0, 2.0]
        self.available_depths = [2, 3, 4]
        self.available_transitions = [10, 13, 18, 20, 23]
        
        # UI state
        self.fullscreen = False
        
        # Input field state
        self.iterations_input_active = False
        self.iterations_input_text = str(self.selected_iterations)
        
        # Pygame specific
        if self.display == 'pygame' and PYGAME_AVAILABLE:
            self.init_pygame()
        elif self.display == 'pygame' and not PYGAME_AVAILABLE:
            print("⚠️  Pygame not available, switching to terminal mode")
            self.display = 'terminal'

    def init_pygame(self):
        """Initialize pygame interface"""
        if not PYGAME_AVAILABLE:
            return
            
        pygame.init()
        pygame.font.init()
        
        # Screen setup
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ataxx Tournament")
        
        # Fonts
        self.font_large = pygame.font.SysFont('Segoe UI', 24, bold=True)
        self.font = pygame.font.SysFont('Segoe UI', 18)
        self.font_small = pygame.font.SysFont('Segoe UI', 14)
        
        # Game board setup
        self.board_size = min(self.screen_width, self.screen_height) - 200
        self.cell_size = self.board_size // 7
        self.board_x = (self.screen_width - self.board_size) // 2
        self.board_y = 80
        
        # Initialize particles for background
        self.particles = []
        for _ in range(50):
            self.particles.append({
                'x': np.random.randint(0, self.screen_width),
                'y': np.random.randint(0, self.screen_height),
                'vx': np.random.uniform(-1, 1),
                'vy': np.random.uniform(-1, 1),
                'alpha': np.random.randint(50, 100)
            })
        
        print("🎮 Pygame initialized successfully")

    def draw_gradient_rect(self, surface, rect, color1, color2, vertical=False):
        """Draw a gradient rectangle"""
        if vertical:
            for i in range(rect.height):
                ratio = i / rect.height
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(surface, (r, g, b), 
                               (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))
        else:
            for i in range(rect.width):
                ratio = i / rect.width
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(surface, (r, g, b), 
                               (rect.x + i, rect.y), (rect.x + i, rect.y + rect.height))

    def draw_shadow(self, surface, rect, offset=5):
        """Draw shadow effect"""
        shadow_rect = pygame.Rect(rect.x + offset, rect.y + offset, rect.width, rect.height)
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height))
        shadow_surface.set_alpha(80)
        shadow_surface.fill(COLORS['text_dark'])
        surface.blit(shadow_surface, shadow_rect)

    def draw_particles_background(self):
        """Draw animated particles in background"""
        if not hasattr(self, 'particles'):
            return
            
        for particle in self.particles:
            # Draw particle
            particle_surface = pygame.Surface((3, 3))
            particle_surface.set_alpha(particle['alpha'])
            particle_surface.fill(COLORS['accent'])
            self.screen.blit(particle_surface, (int(particle['x']), int(particle['y'])))

    def update_particles(self):
        """Update particle positions and properties"""
        if not hasattr(self, 'particles'):
            return
            
        screen_width, screen_height = self.screen.get_size()
        for particle in self.particles:
            # Update particle position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Wrap around screen
            if particle['x'] < 0:
                particle['x'] = screen_width
            elif particle['x'] > screen_width:
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = screen_height
            elif particle['y'] > screen_height:
                particle['y'] = 0

    def draw_board(self, state=None):
        """Draw the game board"""
        if not PYGAME_AVAILABLE or self.display != 'pygame':
            return None
            
        if state is None:
            state = AtaxxState(initial_board=self.tournament.initial_board, 
                             current_player=self.tournament.first_player)
        
        # Clear screen with background
        self.draw_gradient_rect(self.screen, pygame.Rect(0, 0, self.screen_width, self.screen_height),
                              COLORS['bg'], (COLORS['bg'][0]+10, COLORS['bg'][1]+10, COLORS['bg'][2]+10))
        
        board = state.board
        
        # Draw tournament info header
        header_rect = pygame.Rect(0, 0, self.screen_width, 70)
        pygame.draw.rect(self.screen, COLORS['panel'], header_rect)
        
        # Tournament title
        title = f"{self.tournament.algo1} vs {self.tournament.algo2}"
        title_surface = self.font_large.render(title, True, COLORS['text'])
        title_rect = title_surface.get_rect(center=(self.screen_width//2, 25))
        self.screen.blit(title_surface, title_rect)
        
        # Current game info
        current_player_name = "X" if state.current_player == 1 else "O"
        info_text = f"Current Player: {current_player_name}"
        info_surface = self.font.render(info_text, True, COLORS['text'])
        info_rect = info_surface.get_rect(center=(self.screen_width//2, 50))
        self.screen.blit(info_surface, info_rect)
        
        # Draw board background
        board_rect = pygame.Rect(self.board_x - 10, self.board_y - 10, 
                                self.board_size + 20, self.board_size + 20)
        self.draw_shadow(self.screen, board_rect, 8)
        self.draw_gradient_rect(self.screen, board_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], board_rect, border_radius=15)
        pygame.draw.rect(self.screen, COLORS['accent'], board_rect, 3, border_radius=15)
        
        # Draw grid and pieces
        for row in range(7):
            for col in range(7):
                x = self.board_x + col * self.cell_size
                y = self.board_y + row * self.cell_size
                cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Cell background
                if board[row][col] == -2:  # Blocked cell
                    pygame.draw.rect(self.screen, COLORS['blocked'], cell_rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['empty'], cell_rect)
                
                # Grid lines
                pygame.draw.rect(self.screen, COLORS['grid'], cell_rect, 1)
                
                # Pieces
                if board[row][col] == 1:  # X piece (Red)
                    center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    pygame.draw.circle(self.screen, COLORS['player_x'], center, self.cell_size // 3)
                    pygame.draw.circle(self.screen, COLORS['text'], center, self.cell_size // 3, 2)
                    # Draw X symbol
                    font_symbol = pygame.font.SysFont('Arial', self.cell_size // 4, bold=True)
                    x_text = font_symbol.render("X", True, COLORS['text'])
                    x_rect = x_text.get_rect(center=center)
                    self.screen.blit(x_text, x_rect)
                elif board[row][col] == -1:  # O piece (Blue)
                    center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    pygame.draw.circle(self.screen, COLORS['player_o'], center, self.cell_size // 3)
                    pygame.draw.circle(self.screen, COLORS['text'], center, self.cell_size // 3, 2)
                    # Draw O symbol
                    font_symbol = pygame.font.SysFont('Arial', self.cell_size // 4, bold=True)
                    o_text = font_symbol.render("O", True, COLORS['text'])
                    o_rect = o_text.get_rect(center=center)
                    self.screen.blit(o_text, o_rect)
        
        # Draw game statistics sidebar
        sidebar_x = self.board_x + self.board_size + 30
        sidebar_width = self.screen_width - sidebar_x - 20
        
        if sidebar_width > 200:  # Only draw sidebar if there's enough space
            # Piece counts
            x_pieces = np.sum(state.board == 1)
            o_pieces = np.sum(state.board == -1)
            empty_cells = np.sum(state.board == 0)
            
            stats_y = self.board_y
            stats = [
                f"X Pieces: {x_pieces}",
                f"O Pieces: {o_pieces}",
                f"Empty Cells: {empty_cells}",
                "",
                "Tournament Progress:",
                f"{self.tournament.algo1}:",
                f"  W: {self.tournament.results[self.tournament.algo1]['wins']}",
                f"  L: {self.tournament.results[self.tournament.algo1]['losses']}",
                f"  D: {self.tournament.results[self.tournament.algo1]['draws']}",
                f"{self.tournament.algo2}:",
                f"  W: {self.tournament.results[self.tournament.algo2]['wins']}",
                f"  L: {self.tournament.results[self.tournament.algo2]['losses']}",
                f"  D: {self.tournament.results[self.tournament.algo2]['draws']}"
            ]
            
            for i, stat in enumerate(stats):
                if stat:  # Skip empty strings
                    color = COLORS['text']
                    if stat.startswith(self.tournament.algo1) or stat.startswith(self.tournament.algo2):
                        color = COLORS['accent']
                    stat_surface = self.font_small.render(stat, True, color)
                    self.screen.blit(stat_surface, (sidebar_x, stats_y + i * 25))
        
        pygame.display.flip()

    def draw_game_result(self, result):
        """Draw game result overlay"""
        if not PYGAME_AVAILABLE or self.display != 'pygame':
            return
            
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill(COLORS['text_dark'])
        self.screen.blit(overlay, (0, 0))
        
        box_width = min(self.screen_width//2, 400)
        box_height = 200
        result_rect = pygame.Rect(self.screen_width//2 - box_width//2, 
                                self.screen_height//2 - box_height//2, 
                                box_width, box_height)
        
        self.draw_shadow(self.screen, result_rect, 5)
        self.draw_gradient_rect(self.screen, result_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], result_rect, border_radius=20)
        pygame.draw.rect(self.screen, COLORS['accent'], result_rect, 4, border_radius=20)
        
        if result['winner'] == 0:
            result_text = "DRAW!"
            color = COLORS['warning']
        else:
            if result['winner_name']:
                player_symbol = "X" if result['winner'] == 1 else "O"
                result_text = f"{result['winner_name']} ({player_symbol}) WINS!"
            else:
                result_text = f"Player {'X' if result['winner'] == 1 else 'O'} WINS!"
            color = COLORS['success']
        
        text_surface = self.font_large.render(result_text, True, color)
        text_rect = text_surface.get_rect(center=result_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def draw_final_results(self):
        """Draw final tournament results"""
        if not PYGAME_AVAILABLE or self.display != 'pygame':
            return {}
            
        self.draw_gradient_rect(self.screen, pygame.Rect(0, 0, self.screen_width, self.screen_height),
                              COLORS['bg'], (COLORS['bg'][0]+10, COLORS['bg'][1]+10, COLORS['bg'][2]+10))
        
        margin = 50
        result_rect = pygame.Rect(margin, margin, self.screen_width - 2*margin, self.screen_height - 2*margin)
        self.draw_shadow(self.screen, result_rect, 5)
        self.draw_gradient_rect(self.screen, result_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], result_rect, border_radius=20)
        pygame.draw.rect(self.screen, COLORS['accent'], result_rect, 4, border_radius=20)
        
        # Title
        title = self.font_large.render("Tournament Results", True, COLORS['text_black'])
        title_rect = title.get_rect(center=(self.screen_width//2, result_rect.y + 50))
        self.screen.blit(title, title_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        y_offset = result_rect.y + 120;
        
        # Display results for each agent
        for name in self.tournament.results:
            if self.tournament.results[name]["games_played"] > 0:
                avg_pieces = self.tournament.results[name]['avg_pieces'] / self.tournament.results[name]['games_played']
                wins = self.tournament.results[name]['wins']
                losses = self.tournament.results[name]['losses']
                draws = self.tournament.results[name]['draws']
                
                # Agent result card
                agent_rect = pygame.Rect(result_rect.x + 30, y_offset - 10, result_rect.width - 60, 80)
                pygame.draw.rect(self.screen, (255, 255, 255), agent_rect, border_radius=12)
                pygame.draw.rect(self.screen, COLORS['grid'], agent_rect, 2, border_radius=12)
                
                # Agent name
                agent_text = self.font_large.render(name, True, COLORS['text_black'])
                self.screen.blit(agent_text, (agent_rect.x + 20, y_offset))
                
                # Stats
                stats_text = f"{wins}W  {losses}L  {draws}D  {avg_pieces:.1f} avg pieces"
                stats_surface = self.font.render(stats_text, True, COLORS['text_black'])
                self.screen.blit(stats_surface, (agent_rect.x + 20, y_offset + 35))
                
                # Win rate bar
                total_games = wins + losses + draws
                if total_games > 0:
                    win_rate = wins / total_games
                    bar_width = min(300, agent_rect.width - 40)
                    bar_rect = pygame.Rect(agent_rect.x + 20, y_offset + 60, bar_width, 8)
                    pygame.draw.rect(self.screen, COLORS['grid'], bar_rect, border_radius=4)
                    fill_width = int(bar_width * win_rate)
                    if fill_width > 0:
                        fill_rect = pygame.Rect(agent_rect.x + 20, y_offset + 60, fill_width, 8)
                        color = COLORS['success'] if win_rate > 0.5 else COLORS['warning'] if win_rate == 0.5 else COLORS['player_x']
                        pygame.draw.rect(self.screen, color, fill_rect, border_radius=4)
                    
                    # Win rate percentage
                    win_rate_text = f"{win_rate*100:.1f}%"
                    win_rate_surface = self.font_small.render(win_rate_text, True, COLORS['text_black'])
                    self.screen.blit(win_rate_surface, (agent_rect.x + bar_width + 35, y_offset + 55))
                
                y_offset += 120
        
        # Control buttons
        button_y = result_rect.bottom - 80
        button_width = 200
        button_height = 50
        
        menu_button_x = self.screen_width//2 - button_width - 10
        menu_rect = pygame.Rect(menu_button_x, button_y, button_width, button_height)
        
        new_button_x = self.screen_width//2 + 10
        new_rect = pygame.Rect(new_button_x, button_y, button_width, button_height)
        
        # Draw buttons
        self.draw_button(self.screen, menu_rect, "🏠 RETURN TO MENU", mouse_pos, COLORS['accent'])
        self.draw_button(self.screen, new_rect, "🔄 NEW TOURNAMENT", mouse_pos, COLORS['success'])
        
        pygame.display.flip()
        
        return {'menu_button': menu_rect, 'new_button': new_rect}
    
    def draw_menu(self):
        """Draw the main menu interface"""
        if not PYGAME_AVAILABLE or self.display != 'pygame':
            return None
            
        # Clear screen with animated background
        self.draw_gradient_rect(self.screen, pygame.Rect(0, 0, self.screen_width, self.screen_height),
                              COLORS['bg'], (COLORS['bg'][0]+15, COLORS['bg'][1]+15, COLORS['bg'][2]+15))
        
        # Draw animated particles
        self.draw_particles_background()
        
        # Main title
        title_font = pygame.font.SysFont('Segoe UI', 48, bold=True)
        title_surface = title_font.render("ATAXX TOURNAMENT", True, COLORS['text'])
        title_rect = title_surface.get_rect(center=(self.screen_width//2, 150))
        
        # Title shadow
        shadow_surface = title_font.render("ATAXX TOURNAMENT", True, COLORS['text_dark'])
        shadow_rect = shadow_surface.get_rect(center=(self.screen_width//2 + 3, 153))
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(title_surface, title_rect)
        
        # Subtitle
        subtitle_text = f"{self.tournament.algo1} vs {self.tournament.algo2}"
        subtitle_surface = self.font_large.render(subtitle_text, True, COLORS['accent'])
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_width//2, 200))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Tournament settings panel
        panel_width = 600
        panel_height = 300
        panel_x = self.screen_width//2 - panel_width//2
        panel_y = 250
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        self.draw_shadow(self.screen, panel_rect, 8)
        self.draw_gradient_rect(self.screen, panel_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect, border_radius=20)
        pygame.draw.rect(self.screen, COLORS['accent'], panel_rect, 3, border_radius=20)
        
        # Settings info
        settings_y = panel_y + 30
        settings = [
            f"🎮 Games per Match: {self.tournament.games_per_match}",
            f"🧠 MCTS Iterations: {self.tournament.iterations}",
            f"⏱️ Move Delay: {self.tournament.delay}s",
            f"🎯 First Player: {'White (O)' if self.tournament.first_player == 'W' else 'Black (X)'}",
            f"🔍 Minimax Depth: {self.tournament.depths}",
            f"🔄 Transition Threshold: {self.tournament.transition_threshold}"
        ]
        
        for i, setting in enumerate(settings):
            setting_surface = self.font.render(setting, True, COLORS['text'])
            self.screen.blit(setting_surface, (panel_x + 30, settings_y + i * 35))
        
        # Start button
        button_width = 300
        button_height = 60
        button_x = self.screen_width//2 - button_width//2
        button_y = panel_y + panel_height + 40
        start_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        
        mouse_pos = pygame.mouse.get_pos()
        self.draw_button(self.screen, start_rect, "🚀 START TOURNAMENT", mouse_pos, COLORS['success'])
        
        # Control instructions
        instructions = [
            "🎮 Controls:",
            "ESC - Exit",
            "F11 - Toggle Fullscreen",
            "← → - Navigate Maps",
            "SPACE - Pause/Resume (during game)"
        ]
        
        instr_y = button_y + 100
        for i, instruction in enumerate(instructions):
            color = COLORS['accent'] if i == 0 else COLORS['text']
            font = self.font if i == 0 else self.font_small
            instr_surface = font.render(instruction, True, color)
            instr_rect = instr_surface.get_rect(center=(self.screen_width//2, instr_y + i * 25))
            self.screen.blit(instr_surface, instr_rect)
        
        pygame.display.flip()
        return start_rect

    def draw_button(self, surface, rect, text, mouse_pos, base_color):
        """Draw a button with hover effects"""
        is_hovered = rect.collidepoint(mouse_pos)
        
        # Shadow
        shadow_offset = 6 if is_hovered else 4
        shadow_rect = pygame.Rect(rect.x + shadow_offset, rect.y + shadow_offset, rect.width, rect.height)
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height))
        shadow_surface.set_alpha(100 if is_hovered else 70)
        shadow_surface.fill(COLORS['text_dark'])
        surface.blit(shadow_surface, shadow_rect)
        
        # Button background - no border radius
        if is_hovered:
            bg_color = tuple(min(255, c + 30) for c in base_color)
        else:
            bg_color = base_color
        
        pygame.draw.rect(surface, bg_color, rect)
        
        # Border - no border radius
        border_color = COLORS['text'] if is_hovered else COLORS['grid']
        border_width = 2
        pygame.draw.rect(surface, border_color, rect, border_width)
        
        # Text
        button_font = pygame.font.SysFont('Segoe UI', 15, bold=True)
        text_surface = button_font.render(text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)
        
        return rect

    def draw_interactive_menu(self):
        """Draw interactive menu with configurable options using navigation buttons"""
        if not PYGAME_AVAILABLE or self.display != 'pygame':
            return None
            
        # Clear screen
        self.draw_gradient_rect(self.screen, pygame.Rect(0, 0, self.screen_width, self.screen_height),
                              COLORS['bg'], (COLORS['bg'][0]+15, COLORS['bg'][1]+15, COLORS['bg'][2]+15))
        
        # Draw animated particles
        self.draw_particles_background()
        
        # Title
        title_font = pygame.font.SysFont('Segoe UI', 42, bold=True)
        title_surface = title_font.render("ATAXX TOURNAMENT SETUP", True, COLORS['text'])
        title_rect = title_surface.get_rect(center=(self.screen_width//2, 70))
        self.screen.blit(title_surface, title_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        
        # Main settings panel - compact single column
        panel_width = min(650, self.screen_width - 100)
        panel_height = min(420, self.screen_height - 180)
        panel_x = self.screen_width//2 - panel_width//2
        panel_y = 120
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        self.draw_shadow(self.screen, panel_rect, 8)
        self.draw_gradient_rect(self.screen, panel_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect, border_radius=15)
        pygame.draw.rect(self.screen, COLORS['accent'], panel_rect, 3, border_radius=15)
        
        # Settings layout - single column with navigation buttons
        settings_x = panel_x + 30
        col_width = panel_width - 60
        
        settings_y = panel_y + 30
        row_height = 42
        value_width = 180
        button_width = 25
        
        # Settings data
        settings_data = [
            ("🤖 Player 1:", self.selected_algo1, self.available_agents, 'algo1'),
            ("🤖 Player 2:", self.selected_algo2, self.available_agents, 'algo2'),
            ("🗺️  Map:", self.selected_map, self.available_maps, 'map'),
            ("🎮 Games:", self.selected_games, self.available_games, 'games'),
            ("🎯 First Player:", "X" if self.selected_first_player == 1 else "O", ["X", "O"], 'first_player'),
            ("🔍 Minimax Depth:", self.selected_depths, self.available_depths, 'depths'),
            ("🔄 Transition Threshold:", self.selected_transition, self.available_transitions, 'transition'),
            # ("🏆 Use Tournament:", "Yes" if self.selected_use_tournament else "No", ["Yes", "No"], 'use_tournament'),
            ("🧠 MCTS Iter:", self.selected_iterations, 'INPUT', 'iterations'),
        ]
        
        clickable_areas = {}
        
        # Draw settings with navigation buttons
        for i, (label, value, options, key) in enumerate(settings_data):
            x = settings_x
            y = settings_y + i * row_height
            
            # Label
            label_surface = self.font.render(label, True, COLORS['text'])
            self.screen.blit(label_surface, (x, y))
            
            # Control area
            control_x = x + 180
            
            # Special handling for iterations input field
            if options == 'INPUT' and key == 'iterations':
                input_rect = pygame.Rect(control_x, y - 3, value_width, 30)
                is_active = self.iterations_input_active
                
                # Input field background
                bg_color = COLORS['panel_light'] if is_active else COLORS['panel']
                border_color = COLORS['accent'] if is_active else COLORS['grid']
                pygame.draw.rect(self.screen, bg_color, input_rect, border_radius=8)
                pygame.draw.rect(self.screen, border_color, input_rect, 2, border_radius=8)
                
                # Input text
                display_text = self.iterations_input_text if is_active else str(value)
                text_surface = self.font_small.render(display_text, True, COLORS['text'])
                text_rect = text_surface.get_rect(midleft=(input_rect.x + 10, input_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # Cursor for active input
                if is_active and pygame.time.get_ticks() % 1000 < 500:
                    cursor_x = text_rect.right + 2
                    pygame.draw.line(self.screen, COLORS['text'], 
                                   (cursor_x, input_rect.y + 6), 
                                   (cursor_x, input_rect.bottom - 6), 2)
                
                clickable_areas[key] = {
                    'main_rect': input_rect,
                    'left_button': None,
                    'right_button': None,
                    'options': 'INPUT'
                }
            else:
                # Navigation buttons for other settings
                left_button = pygame.Rect(control_x - 35, y - 3, button_width, 30)
                value_rect = pygame.Rect(control_x, y - 3, value_width, 30)
                right_button = pygame.Rect(control_x + value_width + 10, y - 3, button_width, 30)
                
                # Left button
                left_color = COLORS['accent'] if left_button.collidepoint(mouse_pos) else COLORS['panel_light']
                pygame.draw.rect(self.screen, left_color, left_button, border_radius=6)
                pygame.draw.rect(self.screen, COLORS['grid'], left_button, 2, border_radius=6)
                left_arrow = self.font_small.render("◀", True, COLORS['text'])
                left_arrow_rect = left_arrow.get_rect(center=left_button.center)
                self.screen.blit(left_arrow, left_arrow_rect)
                
                # Value display
                value_color = COLORS['panel_light'] if value_rect.collidepoint(mouse_pos) else COLORS['panel']
                pygame.draw.rect(self.screen, value_color, value_rect, border_radius=8)
                pygame.draw.rect(self.screen, COLORS['accent'], value_rect, 2, border_radius=8)
                
                # Value text
                text_surface = self.font_small.render(str(value), True, COLORS['text'])
                text_rect = text_surface.get_rect(center=value_rect.center)
                self.screen.blit(text_surface, text_rect)
                
                # Right button
                right_color = COLORS['accent'] if right_button.collidepoint(mouse_pos) else COLORS['panel_light']
                pygame.draw.rect(self.screen, right_color, right_button, border_radius=6)
                pygame.draw.rect(self.screen, COLORS['grid'], right_button, 2, border_radius=6)
                right_arrow = self.font_small.render("▶", True, COLORS['text'])
                right_arrow_rect = right_arrow.get_rect(center=right_button.center)
                self.screen.blit(right_arrow, right_arrow_rect)
                
                clickable_areas[key] = {
                    'main_rect': value_rect,
                    'left_button': left_button,
                    'right_button': right_button,
                    'options': options
                }
        
        # Control buttons
        button_width = 140
        button_height = 45
        button_y = panel_rect.bottom + 20
        
        # Center the buttons
        total_width = button_width * 3 + 40  # 3 buttons + 2 gaps
        start_x = self.screen_width//2 - total_width//2
        
        reset_rect = pygame.Rect(start_x, button_y, button_width, button_height)
        start_rect = pygame.Rect(start_x + button_width + 20, button_y, button_width, button_height)
        back_rect = pygame.Rect(start_x + 2*button_width + 40, button_y, button_width, button_height)
        
        self.draw_button(self.screen, reset_rect, "🔄 RESET", mouse_pos, COLORS['warning'])
        self.draw_button(self.screen, start_rect, "🚀 START", mouse_pos, COLORS['success'])
        self.draw_button(self.screen, back_rect, "⬅️ BACK", mouse_pos, COLORS['accent'])
        
        # Control instructions
        instr_y = button_y + 60
        instructions = [
            "🎮 Controls: ESC - Exit  |  F11 - Fullscreen  |  ← → Navigation Buttons  |  Click to Change Settings"
        ]
        
        for instruction in instructions:
            instr_surface = self.font_small.render(instruction, True, COLORS['text'])
            instr_rect = instr_surface.get_rect(center=(self.screen_width//2, instr_y))
            self.screen.blit(instr_surface, instr_rect)
        
        pygame.display.flip()
        
        return {
            'clickable_areas': clickable_areas,
            'start_button': start_rect,
            'reset_button': reset_rect,
            'back_button': back_rect
        }
    
    # Note: draw_settings_menu was removed as it was an incomplete duplicate 
    # of draw_interactive_menu. All UI functionality is properly implemented 
    # in draw_interactive_menu method.

    def set_setting(self, key, value):
        """Update a setting value"""
        if key == 'algo1':
            self.selected_algo1 = value
        elif key == 'algo2':
            self.selected_algo2 = value
        elif key == 'map':
            self.selected_map = value
        elif key == 'games':
            self.selected_games = value
        elif key == 'iterations':
            self.selected_iterations = value
        elif key == 'delay':
            self.selected_delay = float(str(value).replace('s', ''))
        elif key == 'first_player':
            self.selected_first_player = 1 if value == 'X' else -1
        elif key == 'depths':
            self.selected_depths = value
        elif key == 'use_tournament':
            self.selected_use_tournament = True if value == "Yes" else False

    def get_map_layout(self, map_name):
        """Get map layout for the specified map name"""
        # All maps are now file-based - use the standalone load_map_from_file function
        return load_map_from_file(map_name)

    def apply_settings(self):
        """Apply current settings to tournament"""
        print(f"🔧 Applying tournament settings:")
        print(f"   Player 1: {self.selected_algo1}")
        print(f"   Player 2: {self.selected_algo2}")
        print(f"   Map: {self.selected_map}")
        print(f"   Games: {self.selected_games}")
        print(f"   Iterations: {self.selected_iterations}")
        print(f"   Depth: {self.selected_depths}")
        print(f"   Transition Threshold: {self.selected_transition}")
        print(f"   Use Tournament: {'Yes' if self.selected_use_tournament else 'No'}")
        
        # Clear previous tournament data
        print("🗑️  Clearing previous tournament data...")
        self.tournament.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                                 for name in [self.selected_algo1, self.selected_algo2]}
        
        self.tournament.algo1 = self.selected_algo1
        self.tournament.algo2 = self.selected_algo2
        self.tournament.games_per_match = self.selected_games
        self.tournament.iterations = self.selected_iterations
        self.tournament.delay = self.selected_delay
        self.tournament.first_player = self.selected_first_player  # Already 1 for X, -1 for O
        self.tournament.depths = self.selected_depths
        self.tournament.transition_threshold = self.selected_transition
        self.tournament.use_tournament = self.selected_use_tournament
        
        # Apply map - set both map_file and initial_board
        # Map from file - use the actual filename with .txt extension
        map_filename = f"{self.selected_map}.txt"
        self.tournament.map_file = map_filename
        self.tournament.initial_board = self.get_map_layout(self.selected_map)
        print(f"📍 Using file map: {map_filename}")
        
        # Re-setup the tournament with new settings
        self.tournament.setup_game()
        
        # Reset tournament state
        self.tournament.running = True
        self.tournament.paused = False
        
        print(f"✅ Tournament settings applied successfully")
        print(f"🔄 Ready to start fresh tournament")

    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.selected_algo1 = "MCTS_Domain_600"
        self.selected_algo2 = "Minimax+AB"
        self.selected_games = 5
        self.selected_iterations = 600
        self.selected_delay = 0.5
        self.selected_first_player = 1
        self.selected_depths = 4
        self.selected_transition = 13
        self.selected_map = "position_00"
        self.selected_use_tournament = False  # Reset tournament selection

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if not PYGAME_AVAILABLE:
            return
            
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # Update screen dimensions
        self.screen_width, self.screen_height = self.screen.get_size()
        
        # Recalculate board positioning
        self.board_size = min(self.screen_width, self.screen_height) - 200
        self.cell_size = self.board_size // 7
        self.board_x = (self.screen_width - self.board_size) // 2
        self.board_y = 80

    async def run_menu(self):
        """Run the menu interface"""
        if self.display != 'pygame' or not PYGAME_AVAILABLE:
            return
            
        # Use the interactive menu instead of the old static menu
        await self.run_interactive_menu()

    async def run_with_ui(self):
        """Run tournament with UI"""
        if self.display == 'pygame' and PYGAME_AVAILABLE:
            await self.run_menu()
            if not self.running:
                return
        
        # Override tournament's play_game method to include UI updates
        original_play_game = self.tournament.play_game
        
        async def play_game_with_ui(*args, **kwargs):
            # Store original state for UI updates
            game_state = AtaxxState(initial_board=self.tournament.initial_board, 
                                  current_player=self.tournament.first_player)
            
            # Show initial board
            if self.display == 'pygame' and PYGAME_AVAILABLE:
                self.draw_board(game_state)
                await asyncio.sleep(1)  # Show initial state
            
            # Run the actual game with periodic UI updates
            result = await self.play_game_with_live_updates(*args, **kwargs)
            
            if result and self.display == 'pygame' and PYGAME_AVAILABLE:
                self.draw_game_result(result)
                for i in range(30):  # Show result for 3 seconds
                    await asyncio.sleep(0.1)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.tournament.running = False
                            return result
            return result
        
        async def play_game_with_live_updates(agent1_name, agent2_name, forward=True):
            """Play game with live pygame updates"""
            state = AtaxxState(initial_board=self.tournament.initial_board, 
                             current_player=self.tournament.first_player)
            
            # Set current player names for display
            if forward:
                current_x_player = agent1_name
                current_o_player = agent2_name
            else:
                current_x_player = agent2_name
                current_o_player = agent1_name
            
            move_count = 0
            x_pieces, o_pieces = 0, 0

            while not state.is_game_over() and self.tournament.running:
                if not self.tournament.running:
                    break
                
                # Handle pygame events
                if self.display == 'pygame' and PYGAME_AVAILABLE:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.tournament.running = False
                            return None
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.tournament.running = False
                                return None
                            elif event.key == pygame.K_SPACE:
                                self.tournament.paused = not self.tournament.paused
                
                if self.tournament.paused:
                    await asyncio.sleep(0.1)
                    continue
                    
                legal_moves = state.get_legal_moves()
                
                if not legal_moves:
                    state.current_player = -state.current_player
                    continue
                
                # Determine which agent should play
                if state.current_player == 1:  # X player
                    current_agent_name = current_x_player
                else:  # O player
                    current_agent_name = current_o_player
                
                agent = self.tournament.agents[current_agent_name]
                move = agent.get_move(state)

                if not self.tournament.running:
                    return None

                if move:
                    state.make_move(move)
                    move_count += 1
                    
                    # Update UI with new board state
                    if self.display == 'pygame' and PYGAME_AVAILABLE:
                        self.draw_board(state)
                    
                    x_pieces = np.sum(state.board == 1)
                    o_pieces = np.sum(state.board == -1)
                    
                    # Add delay if specified
                    if self.tournament.delay > 0:
                        await asyncio.sleep(self.tournament.delay)
                else:
                    state.current_player = -state.current_player
            
            if not self.tournament.running:
                return None
                
            # Calculate final result
            winner = state.get_winner()
            
            x_agent = current_x_player
            o_agent = current_o_player
            
            # Update results
            self.tournament.results[x_agent]["avg_pieces"] += x_pieces
            self.tournament.results[o_agent]["avg_pieces"] += o_pieces
            self.tournament.results[x_agent]["games_played"] += 1
            self.tournament.results[o_agent]["games_played"] += 1
            
            if winner == 1:  # X wins
                winner_name = x_agent
                loser_name = o_agent
                self.tournament.results[x_agent]["wins"] += 1
                self.tournament.results[o_agent]["losses"] += 1
            elif winner == -1:  # O wins
                winner_name = o_agent
                loser_name = x_agent
                self.tournament.results[o_agent]["wins"] += 1
                self.tournament.results[x_agent]["losses"] += 1
            else:
                self.tournament.results[x_agent]["draws"] += 1
                self.tournament.results[o_agent]["draws"] += 1
                winner_name = None
                loser_name = None
            
            return {
                'winner': winner,
                'winner_name': winner_name,
                'loser_name': loser_name,
                'x_pieces': x_pieces,
                'o_pieces': o_pieces,
                'move_count': move_count
            }
        
        # Replace the method
        self.play_game_with_live_updates = play_game_with_live_updates
        self.tournament.play_game = play_game_with_ui
        
        # Run tournament
        await self.tournament.run_tournament()
        
        # Show final results
        if self.display == 'pygame' and PYGAME_AVAILABLE and self.running:
            results_active = True
            while results_active and self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        results_active = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            results_active = False
                        elif event.key == pygame.K_SPACE:
                            # Reset and start new tournament
                            self.tournament.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                                                     for name in [self.tournament.algo1, self.tournament.algo2]}
                            self.menu_active = True
                            results_active = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        buttons = self.draw_final_results()
                        if 'menu_button' in buttons and buttons['menu_button'].collidepoint(event.pos):
                            self.menu_active = True
                            results_active = False
                        elif 'new_button' in buttons and buttons['new_button'].collidepoint(event.pos):
                            self.tournament.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                                                     for name in [self.tournament.algo1, self.tournament.algo2]}
                            self.menu_active = True
                            results_active = False
                
                self.draw_final_results()
                await asyncio.sleep(0.016)

    async def run_interactive_menu(self):
        """Run the interactive menu with event handling"""
        clock = pygame.time.Clock()
        
        while self.running and self.menu_active:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self.fullscreen = not self.fullscreen
                        if self.fullscreen:
                            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                        self.screen_width, self.screen_height = self.screen.get_size()
                    elif event.key == pygame.K_ESCAPE:
                        if self.fullscreen:
                            self.fullscreen = False
                            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                        else:
                            self.running = False
                            return
                    elif self.iterations_input_active:
                        # Handle text input for iterations field
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            # Apply the input
                            try:
                                new_iterations = int(self.iterations_input_text)
                                if new_iterations > 0:
                                    self.selected_iterations = new_iterations
                                    self.iterations_input_active = False
                                else:
                                    # Reset to current value if invalid
                                    self.iterations_input_text = str(self.selected_iterations)
                            except ValueError:
                                # Reset to current value if invalid
                                self.iterations_input_text = str(self.selected_iterations)
                        elif event.key == pygame.K_ESCAPE:
                            # Cancel input
                            self.iterations_input_text = str(self.selected_iterations)
                            self.iterations_input_active = False
                        elif event.key == pygame.K_BACKSPACE:
                            # Remove last character
                            self.iterations_input_text = self.iterations_input_text[:-1]
                        elif event.unicode.isdigit() and len(self.iterations_input_text) < 6:
                            # Add digit (max 6 digits)
                            self.iterations_input_text += event.unicode
                    else:
                        # Arrow key navigation for map selection
                        if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                            current_index = 0
                            if self.selected_map in self.available_maps:
                                current_index = self.available_maps.index(self.selected_map)
                            
                            if event.key == pygame.K_LEFT:
                                # Previous map
                                new_index = (current_index - 1) % len(self.available_maps)
                            else:
                                # Next map
                                new_index = (current_index + 1) % len(self.available_maps)
                            
                            self.selected_map = self.available_maps[new_index]
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        menu_data = self.draw_interactive_menu()
                        if menu_data:
                            self.handle_menu_click(event.pos, menu_data)
            
            # Draw menu
            self.draw_interactive_menu()
            
            # Update particles animation
            self.update_particles()
            
            clock.tick(60)
            await asyncio.sleep(0.001)

    def handle_menu_click(self, pos, menu_data):
        """Handle clicks on the interactive menu with navigation buttons"""
        # Check start button
        if menu_data['start_button'].collidepoint(pos):
            # Apply all settings before starting tournament
            self.apply_settings()
            self.menu_active = False
            return
        
        # Check reset button
        if menu_data['reset_button'].collidepoint(pos):
            self.reset_settings()
            return
        
        # Check back button
        if menu_data['back_button'].collidepoint(pos):
            self.running = False
            return
        
        # Check navigation buttons and input fields
        for key, data in menu_data['clickable_areas'].items():
            main_rect = data['main_rect']
            left_button = data.get('left_button')
            right_button = data.get('right_button')
            options = data['options']
            
            # Handle left navigation button first
            if left_button and left_button.collidepoint(pos):
                self.navigate_setting(key, -1)  # Go to previous option
                return
            
            # Handle right navigation button  
            if right_button and right_button.collidepoint(pos):
                self.navigate_setting(key, 1)  # Go to next option
                return
            
            # Special handling for iterations input field
            if key == 'iterations' and options == 'INPUT':
                if main_rect.collidepoint(pos):
                    self.iterations_input_active = True
                return
        
        # Close any active input
        self.iterations_input_active = False

    def navigate_setting(self, key, direction):
        """Navigate to the next/previous option for a setting"""
        print(f"🔧 Navigating {key} in direction {direction}")
        
        if key == 'algo1':
            current_index = self.available_agents.index(self.selected_algo1)
            new_index = (current_index + direction) % len(self.available_agents)
            self.selected_algo1 = self.available_agents[new_index]
            print(f"  New algo1: {self.selected_algo1}")
        
        elif key == 'algo2':
            current_index = self.available_agents.index(self.selected_algo2)
            new_index = (current_index + direction) % len(self.available_agents)
            self.selected_algo2 = self.available_agents[new_index]
            print(f"  New algo2: {self.selected_algo2}")
        
        elif key == 'map':
            current_index = self.available_maps.index(self.selected_map)
            new_index = (current_index + direction) % len(self.available_maps)
            self.selected_map = self.available_maps[new_index]
            print(f"  New map: {self.selected_map}")
        
        elif key == 'games':
            current_index = self.available_games.index(self.selected_games)
            new_index = (current_index + direction) % len(self.available_games)
            self.selected_games = self.available_games[new_index]
            print(f"  New games: {self.selected_games}")
        
        elif key == 'first_player':
            # Toggle between X and O
            self.selected_first_player = 1 if self.selected_first_player == -1 else -1
            print(f"  New first_player: {'X' if self.selected_first_player == 1 else 'O'}")
        
        elif key == 'depths':
            current_index = self.available_depths.index(self.selected_depths)
            new_index = (current_index + direction) % len(self.available_depths)
            self.selected_depths = self.available_depths[new_index]
            print(f"  New depths: {self.selected_depths}")
        
        elif key == 'transition':
            current_index = self.available_transitions.index(self.selected_transition)
            new_index = (current_index + direction) % len(self.available_transitions)
            self.selected_transition = self.available_transitions[new_index]
            print(f"  New transition_threshold: {self.selected_transition}")
        
        elif key == 'use_tournament':
            # Toggle between Yes and No
            self.selected_use_tournament = not self.selected_use_tournament
            print(f"  New use_tournament: {'Yes' if self.selected_use_tournament else 'No'}")

        

    def reset_settings(self):
        """Reset all settings to default values"""
        self.selected_algo1 = "MCTS_Domain_600"
        self.selected_algo2 = "Minimax+AB"
        self.selected_map = "position_00"
        self.selected_games = 5
        self.selected_iterations = 600
        self.selected_delay = 0.5
        self.selected_first_player = 1
        self.selected_depths = 4
        self.selected_use_tournament = False  # Reset tournament selection

    async def run(self):
        """Main execution loop for the game UI"""
        if self.display == 'pygame' and PYGAME_AVAILABLE:
            while self.running:
                if self.menu_active:
                    await self.run_interactive_menu()
                    if not self.running:
                        break
                else:
                    await self.run_with_ui()
                    self.menu_active = True  # Return to menu after tournament
        else:
            # Terminal mode
            await self.tournament.run_tournament()

def load_map_from_file(map_name):
    """Load a map from the map directory"""
    map_dir = os.path.join(os.path.dirname(__file__), 'map')
    map_file = os.path.join(map_dir, f"{map_name}.txt")
    
    if not os.path.exists(map_file):
        return None
    
    try:
        with open(map_file, 'r') as f:
            lines = f.readlines()
        
        # Convert text format to numpy array
        # W = 1 (white), B = -1 (black), # = -2 (blocked), empty = 0
        board = []
        for line in lines:
            line = line.strip()
            if line:
                row = []
                for char in line:
                    if char == 'W':
                        row.append(1)
                    elif char == 'B':
                        row.append(-1)
                    elif char == '#':
                        row.append(0)
                board.append(row)
        
        return np.array(board)
    except Exception as e:
        print(f"Error loading map {map_name}: {e}")
        return None

def get_available_maps():
    """Get list of available maps from the map directory"""
    map_dir = os.path.join(os.path.dirname(__file__), 'map')
    if not os.path.exists(map_dir):
        return []  # Return empty list if map directory doesn't exist
    
    maps = []
    for filename in os.listdir(map_dir):
        if filename.endswith('.txt'):
            map_name = filename[:-4]  # Remove .txt extension
            maps.append(map_name)

    print(f"Available maps: {maps}")
    
    # No built-in maps - only use file-based maps
    return sorted(maps)

async def main():
    """Main entry point for the game"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ataxx Tournament Runner')
    parser.add_argument('--map-file', type=str, help='Path to map file')
    parser.add_argument('--games', type=int, default=5, help='Number of games per match')
    parser.add_argument('--iterations', type=int, default=600, help='MCTS iterations')
    parser.add_argument('--algo1', type=str, default="MCTS_Domain_600", help='First algorithm')
    parser.add_argument('--algo2', type=str, default="Minimax+AB", help='Second algorithm')
    parser.add_argument('--display', type=str, default='pygame', choices=['pygame', 'terminal'], help='Display mode')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between moves')
    parser.add_argument('--first-player', type=str, default='X', choices=['X', 'W'], help='First player')
    parser.add_argument('--use-tournament', action='store_true', help='Use tournament mode')
    parser.add_argument('--transition-threshold', type=int, default=13, help='Transition threshold')
    parser.add_argument('--depths', type=int, default=4, help='Minimax depth')
    
    args = parser.parse_args()
    
    # Create and run the game UI
    game_ui = AtaxxGameUI(
        map_file=args.map_file,
        games_per_match=args.games,
        iterations=args.iterations,
        algo1=args.algo1,
        algo2=args.algo2,
        display=args.display,
        delay=args.delay,
        first_player=args.first_player,
        use_tournament=args.use_tournament,
        transition_threshold=args.transition_threshold,
        depths=args.depths
    )
    
    await game_ui.run()
