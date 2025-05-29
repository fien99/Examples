import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np


from catch import *
#policy and value network
class Network(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Network, self).__init__()
		self.actor = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size),
		)
		self.critic = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1)
		)
	def softmax(self,preds):
		#print(preds)
		
		temp = 2000
		x = preds/temp
		z = x - x.max()
		return torch.exp(z)/torch.sum(torch.exp(z),axis=1)  
	 	
		
	
	def forward(self, x):
		actor=self.actor(x)
		actor=self.softmax(actor)
		#actor=torch.softmax(actor,dim=1)
		#print(actor)
		return actor, self.critic(x)
	



class ActorCritic():

	def __init__(self, input_size, hidden_size, output_size, alpha, gamma,entropy_weight,bootstrap,baseline):
		self.gamma = gamma
		self.entropy_weight =entropy_weight
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		self.model= Network(input_size,hidden_size,output_size)
		self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
		self.bootstraps =bootstrap
		self.baselines=baseline
		self.learning_rate = alpha

	def set_entropy_weight(self,entropy_weihgt):
		self.entropy_weight=entropy_weihgt
	def set_learning_rate(self, alpha):
		self.model= Network(self.input_size,self.hidden_size,self.output_size)
		self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

	def select_action(self, state):
		

		state = torch.from_numpy(state).float().unsqueeze(0)
		probabilities, critic_val = self.model(state)
		action = torch.multinomial(probabilities, 1).item()
		log_prob = torch.log(probabilities).squeeze(0)

		probabilities = probabilities.detach().numpy()
		probabilities = probabilities[0]
		probabilities = torch.FloatTensor(probabilities)

		return action, log_prob, probabilities, critic_val
	#flatten state to 1d array
	def simplify_state(self, state):

		simple_state = []
		for s in state:
			s_m = np.reshape(s, -1)
			simple_state.append(s_m)
		flatten_list = [item for sublist in simple_state for item in sublist]
		flatten_list = np.array(flatten_list)
		return flatten_list
	#implementation of bootstrap
	def bootstrap(self, trace,values):
		
		estimate=values
		rewards = [reward for state, action, reward, next_state, done in trace]
		estimated_reward =np.zeros_like(rewards)
		for t in reversed(range(len(rewards))):

			estimate = self.gamma**t *estimate +rewards[t]
			estimated_reward[t]=estimate

		return estimated_reward
	
	#implementation of the Monte Carlo
	def notbootstrap(self,trace):
		# Compute discounted rewards
		rewards = [reward for state, action, reward, next_state, done in trace]
		discounted_rewards = np.zeros_like(rewards)
		
		R = 0
		for t in reversed(range(len(rewards))):
			R= (R *((self.gamma))+rewards[t])
			discounted_rewards[t] = R
		return 
	
	#implementation of the baseline	
	def baseline(self, estimated ,values):

		advantage = estimated - values
		return advantage
	
	#caculating the policy loss
	def update_actor(self, estimate, trace, log_probabilities):
		
		loss = 0
		estimate=torch.FloatTensor(estimate)
		
		loss=(-(log_probabilities*estimate).sum())/len(estimate)
		

		return loss

	#caculating the value loss
	def update_critic(self, estimate,values):

		loss= (-(estimate-values)**2).mean()

		return loss
	#training loop
	def train(self, env, num_episodes):

		score = []
		for episode in range(num_episodes):
			state = env.reset()
			done = False
			trace = []
			probabilities = []
			log_probabilities = []
			values=[]
			r=0
			# Collect the trace
			while not done:
				convertedState = self.simplify_state(state)
				#convertedState=state
				action, log_probs, action_probs,value = self.select_action(convertedState)
				next_state, reward, done, _ = env.step(action)

				trace.append((state, action, reward, next_state, done))
				values.append(value.item())
				probabilities.append(action_probs[action])
				log_probabilities.append(log_probs[action])

				state = next_state
				r += reward

			score.append(r)
			convertedState = self.simplify_state(next_state)
			#convertedState = next_state
			_, _, _,value = self.select_action(convertedState)
	
			# Bootstrap
			if self.bootstraps and not self.baselines:
				estimate = self.bootstrap(trace,value)
			elif self.baselines and not self.bootstraps:
				estimate = self.notbootstrap(trace)
				estimate = self.baseline(estimate,values)
			elif self.bootstraps and self.baselines:
				estimate = self.bootstrap(trace,value)
				estimate = self.baseline(estimate,values)

			# Update the networks

			actor_loss=self.update_actor(estimate, trace, torch.stack(log_probabilities))
			critic_loss=self.update_critic(estimate,values)
			entropy = (-(torch.stack(probabilities) * torch.stack(log_probabilities)).sum(dim=0)).mean()
			loss = actor_loss + critic_loss+ self.entropy_weight*entropy
			#print(loss)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.25)
			self.optimizer.step()

			# if episode % 50 == 0 and episode>0:
			# 	print('Trajectory {}\tAverage Score: {:.2f}'.format(episode, np.mean(score[-50:-1])))
		return score


def main():
	obs_size = 7*7*2
	n_action =3
	hidden_size = 256

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

	env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps, 
				max_misses=max_misses, observation_type=observation_type, seed=seed)

	agent = ActorCritic(obs_size, hidden_size, n_action, learning_rate, gamma,entropy_weight,True,False)
	agent.train(env,1000)

if __name__ == '__main__':
	main()




	

