import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from catch import*


class PolicyNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(PolicyNetwork, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)

	def softmax(self,preds):
		#print(preds)
		temperature = 100
		x = preds/temperature
		z = x - x.max()
		return torch.exp(z)/torch.sum(torch.exp(z),axis=1)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = self.softmax(self.fc2(x))
		return x

class REINFORCE():
	def __init__(self, input_size, hidden_size, output_size, learning_rate, gamma, entropy_weight):
		self.policy_network = PolicyNetwork(input_size, hidden_size, output_size)
		self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
		self.gamma = gamma
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.entropy_weight = entropy_weight
		self.lr=learning_rate
	def set_learning_rate(self,alpha):
		self.policy_network = PolicyNetwork(self.input_size, self.hidden_size, self.output_size)
		self.optimizer = optim.Adam(self.policy_network.parameters(), lr=alpha)
			
	def select_action(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		probabilities = self.policy_network(state)
		#print(probabilities)
		action = torch.multinomial(probabilities, 1).item()
		log_prob = torch.log(probabilities).squeeze(0)


		probabilities=probabilities.detach().numpy()
		probabilities=probabilities[0]
		probabilities=torch.FloatTensor(probabilities)
		return action, log_prob,probabilities
	
	
	def update(self, rewards, log_probabilities,probabilities):
		
		# Compute discounted rewards
		discounted_rewards = np.zeros_like(rewards)
		
		R = 0
		for t in reversed(range(len(rewards))):
			R= (R *((self.gamma))+rewards[t])
			discounted_rewards[t] = R
		

		discounted_rewards = torch.tensor(discounted_rewards).float()
		#discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())
		
		# Compute policy loss with entropy regularization
		policy_loss = (-(log_probabilities * discounted_rewards).sum())/len(rewards)
		entropy = (-(probabilities * log_probabilities).sum(dim=0).sum())/len(rewards)
		loss = policy_loss + self.entropy_weight * entropy
		#print(loss)
		# Update policy network
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def simplify_state(self,state):
		simple_state=[]
		for s in state:
			s_m=np.reshape(s,-1)
			simple_state.append(s_m)
		flatten_list = [item for sublist in simple_state for item in sublist]
		flatten_list = np.array(flatten_list)
		#print(flatten_list)
		return flatten_list

	def train(self, env, num_episodes):
		score=[]
		for episode in range(num_episodes):
			state = env.reset()
			rewards = []
			log_probabilities = []
			probabilities=[]
			r=0
			while True:
				#env.render()
				state=self.simplify_state(state)
				action, log_prob,prob = self.select_action(state)
				next_state, reward, done, _ = env.step(action)
				log_probabilities.append(log_prob[action])
				probabilities.append(prob[action])
				rewards.append(reward)
				r+=reward
				state = next_state
				if done:
					break
			score.append(r)
			self.update(rewards, torch.stack(log_probabilities),torch.stack(probabilities))
			#if episode % 50 == 0 and episode>0:
			#	print('Trajectory {}\tAverage Score: {:.2f}'.format(episode, np.mean(score[-50:-1])))
		return score

if __name__ == '__main__':
	obs_size = 7*7*2
	n_actions = 3
	hidden=256


	
	rows = 7
	columns = 7
	speed = 1.0
	max_steps = 250
	max_misses = 10
	observation_type ='pixel' #'vector' # 'pixel'
	seed = None
	learning_rate = 0.01
	gamma=0.95
	entropy_weight=0.0001
	# Initialize environment and Q-array

	env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
			max_misses=max_misses, observation_type=observation_type, seed=seed)
	
	agent=REINFORCE(obs_size,hidden,n_actions,learning_rate,gamma,entropy_weight)
	agent.train(env,1000)