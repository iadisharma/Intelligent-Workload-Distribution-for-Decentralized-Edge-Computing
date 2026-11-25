import sys
import os
import pickle
sys_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(sys_path)
from NEAT.neat import run

if __name__ == "__main__":
    winner, stats = run()
    print(winner)
    
    # Salva il genoma vincente
    os.makedirs(os.path.join("logs", "NEAT"), exist_ok=True)
    with open(os.path.join("logs", "NEAT", "winner_genome.pkl"), "wb") as f:
        pickle.dump(winner, f)