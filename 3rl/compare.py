from reinforce3 import *
from actor_critic import *
from catch import *
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import time
import argparse
import logging 

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def tune_lr():
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
	rates=[0.1,0.01,0.001]
	average_score=[]
	for learning_rate in tqdm(rates):
		scores=[]
		for _ in tqdm(range(50)):
			agent=REINFORCE(obs_size,hidden,n_actions,learning_rate,gamma,entropy_weight)
			score=agent.train(env,1000)
			scores.append(score)
		scores=np.array(scores)
		scores=np.average(scores,axis=0)
		
		#scores=savgol_filter(scores,len(scores)//3,1)
		label="REINFORCE lr:"+str(learning_rate)
		plt.plot(range(len(score)),smooth(scores,0.9), label=label)


	plt.xlabel(f"Number of episodes trained")
	plt.ylabel(f"Rewards")
	plt.title(f"Progress for REINFORCE agent")
	plt.legend()
	plt.grid()
	plt.savefig("tune_lr.png")

def tune_ew():
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

	weights=[0.0001,0.001,0.01]
	average_score=[]
	for entropy_weight in tqdm(weights):
		
		scores=[]
		for _ in tqdm(range(50)):
			agent=REINFORCE(obs_size,hidden,n_actions,learning_rate,gamma,entropy_weight)
			score=agent.train(env,1000)
			scores.append(score)
		scores=np.array(scores)
		scores=np.average(scores,axis=0)
		label="REINFORCE ew:"+str(entropy_weight)
		plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

	plt.xlabel(f"Number of episodes trained")
	plt.ylabel(f"Rewards")
	plt.title(f"Progress for REINFORCE agent")
	plt.legend()
	plt.grid()
	plt.savefig("tune_ew.png")

def ac_tune_lr():
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
	agents=[ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,entropy_weight,True,False),ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,entropy_weight,False,True),ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,entropy_weight,True,True)]
	labels=['ActorCritic bootstrap','ActorCritic baseline','ActorCritc bstr+bsli']
	lr=[0.01,0.001,0.0001]
	for x in tqdm(range(len(agents))):
		for alpha in tqdm(lr):
			
			scores=[]
			for _ in tqdm(range(50)):
				agent=agents[x]
				agent.set_learning_rate(alpha)
				score=agent.train(env,1000)
				scores.append(score)
			scores=np.array(scores)
			scores=np.average(scores,axis=0)
			label=labels[x]+" lr: "+str(alpha)
			plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

		plt.xlabel(f"Number of episodes trained")
		plt.ylabel(f"Rewards")
		plt.title(f"Progress for Actor Critic agents")
		plt.legend()
		plt.grid()
		plt.savefig("ac_tune_lr.png")

def ac_tune_ew():
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
	agents=[ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.0001,True,False),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.001,True,False),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.01,True,False),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.0001,False,True),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.001,False,True),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.01,False,True),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.0001,True,True),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.001,True,True),
			ActorCritic(obs_size, hidden, n_actions, learning_rate, gamma,0.01,True,True)]
	labels=['ActorCritic bootstrap','ActorCritic baseline','ActorCritc bstr+bsli']
	weights=[0.0001,0.001,0.01,0.0001,0.001,0.01,0.0001,0.001,0.01]
	for x in tqdm(range(len(agents))):
		
		scores=[]
		for _ in tqdm(range(50)):
			agent=agents[x]
			agent.set_learning_rate(learning_rate)
			score=agent.train(env,1000)
			scores.append(score)
		scores=np.array(scores)
		scores=np.average(scores,axis=0)
		label=labels[x]+" ew: "+str(weights[x])
		plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

		plt.xlabel(f"Number of episodes trained")
		plt.ylabel(f"Rewards")
		plt.title(f"Progress for Actor Critic agents")
		plt.legend()
		plt.grid()
		plt.savefig("ac_tune_ew.png")

	pass


def vs():
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

	agents=[REINFORCE(obs_size,hidden,n_actions,0.1,gamma,0.01),
	 		ActorCritic(obs_size, hidden, n_actions, 0.01, gamma,entropy_weight,True,False),
	 		ActorCritic(obs_size, hidden, n_actions, 0.01, gamma,entropy_weight,False,True),
			ActorCritic(obs_size, hidden, n_actions, 0.01, gamma,entropy_weight,True,True)]
	labels=['REINFORCE','ActorCritic bootstrap','ActorCritic baseline','ActorCritc bstr+bsli']
	lr=[0.1,0.01,0.01,0.01]
	for x in tqdm(range(len(agents))):
		
		scores=[]
		for _ in tqdm(range(50)):
			agent=agents[x]
			agent.set_learning_rate(lr[x])
			score=agent.train(env,1000)
			scores.append(score)
		scores=np.array(scores)
		scores=np.average(scores,axis=0)
		label=labels[x]
		plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

	plt.xlabel(f"Number of episodes trained")
	plt.ylabel(f"Rewards")
	plt.title(f"Compare different agents")
	plt.legend()
	plt.grid()
	plt.savefig("compare.png")

def env_speed():
	obs_size = 7*7*2
	n_actions = 3
	hidden=256
	rows = 7
	columns = 7
	#speed = 1.0
	max_steps = 250
	max_misses = 10
	observation_type ='pixel' #'vector' # 'pixel'
	seed = None
	learning_rate = 0.01
	gamma=0.95
	entropy_weight=0.0001
	# Initialize environment and Q-array

	# env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
	# 		max_misses=max_misses, observation_type=observation_type, seed=seed)

	speeds=[0.5,1.0,1.5]
	for speed in tqdm(speeds):
		scores=[]
		env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
			max_misses=max_misses, observation_type=observation_type, seed=seed)
		for _ in tqdm(range(50)):
			agent=ActorCritic(obs_size,hidden,n_actions,0.01,gamma,0.0001,True,True)
			agent.set_learning_rate(0.01)
			score=agent.train(env,1000)
			scores.append(score)
		scores=np.array(scores)
		scores=np.average(scores,axis=0)
		label="AC speed: "+str(speed)
		plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

	plt.xlabel(f"Number of episodes trained")
	plt.ylabel(f"Rewards")
	plt.title(f"Compare different speeds in enviroment")
	plt.legend()
	plt.grid()
	plt.savefig("env_speed.png")
	pass

def env_size():
	#obs_size = 7*7*2
	n_actions = 3
	hidden=256
	#rows = 7
	#columns = 7
	speed = 1.0
	max_steps = 250
	max_misses = 10
	observation_type ='pixel' #'vector' # 'pixel'
	seed = None
	learning_rate = 0.01
	gamma=0.95
	entropy_weight=0.0001
	# Initialize environment and Q-array

	# env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
	# 		max_misses=max_misses, observation_type=observation_type, seed=seed)

	size=[(4,4),(7,7),(10,10)]
	#colums=[4,7,10]
	for rows,columns in tqdm(size):
		scores=[]
		obs_size=rows*columns*2
		env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
			max_misses=max_misses, observation_type=observation_type, seed=seed)
		for _ in tqdm(range(50)):
			agent=ActorCritic(obs_size,hidden,n_actions,0.01,gamma,0.0001,True,True)
			agent.set_learning_rate(0.01)
			score=agent.train(env,1000)
			scores.append(score)
		scores=np.array(scores)
		scores=np.average(scores,axis=0)
		label="AC env_size: "+str(rows)+","+str(columns)
		plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

	plt.xlabel(f"Number of episodes trained")
	plt.ylabel(f"Rewards")
	plt.title(f"Compare different enviroment sizes")
	plt.legend()
	plt.grid()
	plt.savefig("env_size.png")
	pass

def env_vector():	
	obs_size = 3
	n_actions = 3
	hidden=256
	rows = 7
	columns = 7
	#speed = 1.0
	max_steps = 250
	max_misses = 10
	observation_type ='vector' #'vector' # 'pixel'
	seed = None
	learning_rate = 0.01
	gamma=0.95
	entropy_weight=0.0001
	# Initialize environment and Q-array

	# env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
	# 		max_misses=max_misses, observation_type=observation_type, seed=seed)

	speeds=[1.0,1.5]
	for speed in tqdm(speeds):
		scores=[]
		env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
			max_misses=max_misses, observation_type=observation_type, seed=seed)
		for _ in tqdm(range(50)):
			agent=ActorCritic(obs_size,hidden,n_actions,0.001,gamma,0.0001,True,True)
			agent.set_learning_rate(0.001)
			score=agent.train(env,1000)
			scores.append(score)
		scores=np.array(scores)
		scores=np.average(scores,axis=0)
		label="AC speed: "+str(speed)
		plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

	plt.xlabel(f"Number of episodes trained")
	plt.ylabel(f"Rewards")
	plt.title(f"Compare different speeds in enviroment with vector setting")
	plt.legend()
	plt.grid()
	plt.savefig("env_vector.png")

	pass

def env_combi():
	obs_size = 14*7*2
	n_actions = 3
	hidden=256
	rows = 7
	columns = 14
	speed = 0.5
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


	scores=[]

	for _ in tqdm(range(50)):
		agent=ActorCritic(obs_size,hidden,n_actions,0.01,gamma,0.0001,True,True)
		agent.set_learning_rate(0.01)
		score=agent.train(env,1000)
		scores.append(score)
	scores=np.array(scores)
	scores=np.average(scores,axis=0)
	label="AC speed: "+str(speed)
	plt.plot(range(len(scores)), smooth(scores,0.9), label=label)

	plt.xlabel(f"Number of episodes trained")
	plt.ylabel(f"Rewards")
	plt.title(f"Enivorment with width: 14 and higth: 7 and  a speed of 0.5")
	plt.legend()
	plt.grid()
	plt.savefig("env_combi.png")
	pass
	
def init_logger():
	logging.basicConfig(filename= 'RL_a3.log',
						level=logging.INFO,
						format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

	logger = logging.getLogger()
	return logger


if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser(description="Argument Parser")
	parser.add_argument("--re_tune_lr", action="store_true", default=False)
	parser.add_argument("--re_tune_ew", action="store_true", default=False)
	parser.add_argument("--ac_tune_lr", action="store_true", default=False)
	parser.add_argument("--ac_tune_ew", action="store_true", default=False)
	parser.add_argument("--env_speed",action="store_true",default=False)
	parser.add_argument("--env_size",action="store_true",default=False)
	parser.add_argument("--env_vector",action="store_true",default=False)
	parser.add_argument("--env_combi",action="store_true",default=False)
	parser.add_argument("--vs",action="store_true",default=False)



	args = parser.parse_args()
	logger = init_logger()

	experiments = {
		"re_tune_lr": tune_lr,
		"re_tune_ew": tune_ew,
		"ac_tune_lr": ac_tune_lr,
		"ac_tune_ew": ac_tune_ew,
		"vs": vs,
		"env_speed": env_speed,
		"env_size": env_size,
		"env_vector": env_vector,
		"env_combi":env_combi,
		
	}
	for key, val in experiments.items():
		if getattr(args, key):
			logger.info(f"Starting {key}")
			logger.info("------------------------")
			start = time.perf_counter()
			val()
			end = time.perf_counter()
			logger.info(f"Finished {key} in {end - start} seconds")