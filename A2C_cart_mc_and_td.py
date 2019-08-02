# finish a2c cart problem - Monte Carlo and TD(Temporal Difference)
# written by Hyunseok Whang
# date: 2019-08-01
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env._max_episode_steps=5000
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

max_episode = 2000
gamma = 0.99

train_rate = 32
hidden_size1 = 24
hidden_size2 = 12
act_lr = 0.001
cri_lr = 0.001

def discounted_reward1(rewards):
	discount = np.zeros_like(rewards)
	running_add = 0
	
	for t in reversed(range(len(rewards))):
		running_add = rewards[t]+gamma*running_add
		discount[t] = running_add
	return discount

def discounted_reward2(rewards,value):
	discount = np.zeros_like(rewards)
	running_add = value
	
	for t in reversed(range(len(rewards))):
		running_add = rewards[t]+gamma*running_add
		discount[t] = running_add
	return discount
	
observe_input = tf.placeholder(shape=[None,observation_space],dtype=tf.float32)
act_input = tf.placeholder(shape=[None,action_space],dtype=tf.float32)
target_input = tf.placeholder(shape=[None,],dtype=tf.float32)
adv_input = tf.placeholder(shape=[None,],dtype=tf.float32)
with tf.variable_scope('actor'):
	weight_1a = tf.get_variable('weight_1a',shape=[observation_space,hidden_size1],initializer=tf.contrib.layers.xavier_initializer())
	hidden_1a = tf.nn.relu(tf.matmul(observe_input,weight_1a))
	weight_2a = tf.get_variable('weight_2a',shape=[hidden_size1,hidden_size2],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
	hidden_2a = tf.nn.relu(tf.matmul(hidden_1a,weight_2a))
	weight_3a = tf.get_variable('weight_3a',shape=[hidden_size2,action_space],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
	act_out = tf.nn.softmax(tf.matmul(hidden_2a,weight_3a))
	act_loss = -tf.reduce_mean(adv_input*tf.log(tf.reduce_sum(act_input*act_out,axis=1)))
	act_opt = tf.train.AdamOptimizer(act_lr).minimize(act_loss)

with tf.variable_scope('critic'):
	weight_1c = tf.get_variable('weight_1c',shape=[observation_space,hidden_size1],initializer=tf.contrib.layers.xavier_initializer())
	hidden_1c = tf.nn.relu(tf.matmul(observe_input,weight_1c))
	weight_2c = tf.get_variable('weight_2c',shape=[hidden_size1,hidden_size2],initializer=tf.contrib.layers.xavier_initializer())
	hidden_2c = tf.nn.relu(tf.matmul(hidden_1c,weight_2c))
	weight_3c = tf.get_variable('weight_3c',shape=[hidden_size2,1],initializer=tf.contrib.layers.xavier_initializer())
	cri_out = tf.matmul(hidden_2c,weight_3c)
	cri_loss = tf.reduce_mean(tf.square(target_input-cri_out))
	cri_opt = tf.train.AdamOptimizer(cri_lr).minimize(cri_loss)
	
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	epi_reward_buff = []
	for episode in range(max_episode):
		observation = env.reset()
		observation = np.reshape(observation,[1,observation_space])
		
		done = False
		epi_reward = 0	
		epi_max_prob = 0
		state_buff,rewards,act_buff = [],[],[]
		train_step = 0
		while not done:
			
			act_prob = sess.run(act_out,feed_dict={observe_input:observation})
			action = np.random.choice(action_space,1,p=act_prob[0])
			act = np.zeros(action_space)
			act[action] = 1

			if max(act_prob[0]) >= epi_max_prob:
				epi_max_prob = max(act_prob[0])
			
			next_observation,reward,done,_ = env.step(np.argmax(act))
			next_observation = np.reshape(next_observation,[1,observation_space])
			epi_reward += reward
			
			rewards.append(reward)
			state_buff.append(observation[0])
			act_buff.append(act)
			
			if done:
				epi_reward_buff.append(epi_reward)
				target = discounted_reward1(rewards)
				target = (target-np.mean(target))/np.std(target)
				sess.run([act_opt],feed_dict={act_input:act_buff,observe_input:state_buff,adv_input:target})
				if episode % 10 == 0:
					print('Episode : {:d}, Episode_reward : {:d}, Episode_max_policy : {:.3f}%'.format(int(episode),int(epi_reward),epi_max_prob*100))
			elif train_step % train_rate == 0 and episode > 50:
				new_val = sess.run(cri_out,feed_dict={observe_input:[state_buff[-1]]})[0,0]
				target = discounted_reward2(rewards,new_val)
				ad = target-sess.run(cri_out,feed_dict={observe_input:[state_buff[0]]})[0,0]
				target = (target-np.mean(target))/(np.std(target)+1e-15)
				ad = (ad-np.mean(ad))/(np.std(ad)+1e-15)
				sess.run([act_opt,cri_opt],feed_dict={act_input:act_buff,observe_input:state_buff,target_input:target,adv_input:ad})
				
			observation = next_observation[:]
			train_step += 1
			
	plt.plot(epi_reward_buff,'-')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Actor-Critic_TD Cartpole reward result')
	plt.show()			