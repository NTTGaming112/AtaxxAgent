#!/usr/bin/env python3

from typing import Dict, List, Tuple, Optional
import threading

class MoveScoreManager:
    
    def __init__(self):
        self._scores = {}
        self._lock = threading.Lock()
        self._enabled = False
        self._agent_name = ""
        
    def enable_score_collection(self, agent_name: str = ""):
        with self._lock:
            self._enabled = True
            self._scores.clear()
            self._agent_name = agent_name
    
    def disable_score_collection(self):
        with self._lock:
            self._enabled = False
            self._agent_name = ""
    
    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled
    
    def store_move_score(self, move: Tuple[int, int, int, int], score: float, agent_source: str = ""):
        with self._lock:
            if self._enabled:
                self._scores[move] = score
    
    def store_move_scores(self, move_scores: Dict[Tuple[int, int, int, int], float], agent_source: str = ""):
        with self._lock:
            if self._enabled:
                self._scores.update(move_scores)
    
    def get_move_scores(self) -> Dict[Tuple[int, int, int, int], float]:
        with self._lock:
            return self._scores.copy()
    
    def get_move_score(self, move: Tuple[int, int, int, int]) -> Optional[float]:
        with self._lock:
            return self._scores.get(move)
    
    def clear_scores(self):
        with self._lock:
            self._scores.clear()
    
    def get_destination_scores(self) -> Dict[Tuple[int, int], float]:
        with self._lock:
            dest_scores = {}
            for (r, c, nr, nc), score in self._scores.items():
                dest_pos = (nr, nc)
                if dest_pos not in dest_scores or score > dest_scores[dest_pos]:
                    dest_scores[dest_pos] = score
            return dest_scores
    
    def get_current_agent_name(self) -> str:
        with self._lock:
            return self._agent_name

move_score_manager = MoveScoreManager()