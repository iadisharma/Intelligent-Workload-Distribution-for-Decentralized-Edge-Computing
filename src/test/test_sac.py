import torch
import sys
import os
sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(sys_path)
from env.env import TrafficManagementEnv
from SAC.SAC import SAC
import matplotlib.pyplot as plt
import csv

state_dim = 5  
action_dim = 3  
agent = SAC(state_dim, action_dim, device=torch.device("cpu"))

path_to_weights = os.path.join('logs', 'SAC', 'SAC_weights')
agent.load_weights_SAC(path_to_weights)

env = TrafficManagementEnv()  
num_episodes = 50
all_episode_rewards = []
all_episode_rejections = []
all_managed_requests_per_episode = []
congestione_counts_per_episode = []

prev_total_requests = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, truncated, done, info = env.step(action)
        episode_reward += reward
        state = next_state
        
    print(f"Episodio {episode + 1}: Ricompensa Totale = {episode_reward}")
    # Registra le metriche
    all_episode_rewards.append(episode_reward)
    all_episode_rejections.append(env.total_rejected_requests)
    # Calcolo delle richieste gestite per l'episodio corrente
    managed_requests_this_episode = env.total_managed_requests - prev_total_requests
    all_managed_requests_per_episode.append(managed_requests_this_episode)
    # Aggiornamento del valore di prev_total_requests per il prossimo episodio
    prev_total_requests = env.total_managed_requests 
    # Registra il numero di steps in congestione e poi azzeralo per il prossimo episodio
    congestione_counts_per_episode.append(env.congestione_one_count)
    env.congestione_one_count = 0
# Calcola la percentuale di richieste rifiutate per ogni episodio
rejection_percentages = [(rejections/requests) * 100 if requests != 0 else 0 for rejections, requests in zip(all_episode_rejections, all_managed_requests_per_episode)]

path_to_save_csv = os.path.join("logs", "SAC", "results.csv")
os.makedirs(os.path.dirname(path_to_save_csv), exist_ok=True)
with open(path_to_save_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Scrivi l'intestazione del CSV
    csv_writer.writerow(["Episodio", "Ricompensa Totale", "Richieste Rifiutate", "Richieste Gestite", "Steps in Congestione", "Percentuale di Richieste Rifiutate"])
    # Scrivi i dati per ogni episodio
    for i in range(num_episodes):
        csv_writer.writerow([i + 1, all_episode_rewards[i], all_episode_rejections[i], all_managed_requests_per_episode[i], congestione_counts_per_episode[i], rejection_percentages[i]])

# Plotting
plt.figure(figsize=(24, 5))

# subplot per la ricompensa
plt.subplot(1, 4, 1)
plt.plot(all_episode_rewards, marker='o', linestyle='-')
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)

# subplot per la percentuale di rifiuti
plt.subplot(1, 4, 2)
plt.plot(rejection_percentages, marker='o', linestyle='-', color='blue')
plt.title('Rejected Requests (%) per Episode')
plt.xlabel('Episode')
plt.ylabel('Rejected Requests (%)')
plt.ylim(0, 100) 
plt.grid(True)

# subplot per il numero assoluto di rifiuti e il numero totale di richieste gestite
plt.subplot(1, 4, 3)
plt.plot(all_episode_rejections, marker='o', linestyle='-', color='red', label='Rejected Requests')
plt.title('Rejected Requests Count per Episode')
plt.xlabel('Episode')
plt.ylabel('Number of Requests')
plt.grid(True)

# subplot per il numero di steps in congestione per episodio
plt.subplot(1, 4, 4)
plt.plot(congestione_counts_per_episode, marker='o', linestyle='-', color='purple', label='Congestion Steps')
plt.title('Congestion Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Congestion Steps')
max_congestion = max(congestione_counts_per_episode) 
plt.yticks(range(0, max_congestion + 1))
plt.grid(True)

plt.suptitle('SAC Evaluation Results (50 Episodes)', fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig(os.path.join("results", "sac_test_results.png"), dpi=150)
plt.show()
