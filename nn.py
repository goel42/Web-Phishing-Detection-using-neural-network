import arff
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

# # def dense_to_one_hot(labels_dense, num_classes):
# #     """Convert class labels from scalars to one-hot vectors"""
# #     num_labels = labels_dense.shape[0]
# #     index_offset = np.arange(num_labels) * num_classes
# #     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#     labels_one_hot[np.arange(num_labels),labels_dense]
    
# #     return labels_one_hot

# def dense_to_one_hot(labels_dense, num_classes):
#     """Convert class labels from scalars to one-hot vectors"""
#     num_labels = labels_dense.shape[0]
#     print("1111111")
#     print(num_labels)
#     index_offset = np.arange(num_labels) * num_classes
#     print ("2222222")
#     print(index_offset)
#     print("333333333")
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     print(labels_one_hot)
#     idx = index_offset + labels_dense.ravel()
#     print("4444444")
#     print(idx)
#     labels_one_hot.flat[idx] = 1
#     print("5555555")
#     print(labels_one_hot.shape)
#     return labels_one_hot
def dense_to_one_hot(labels_dense, num_classes):
	num_labels = labels_dense.shape[0]
	labels_dense = labels_dense.astype(int)
	# print("######")
	# print(num_labels)
	# print(type(num_labels))
	# print("######")
	b = np.zeros((num_labels, num_classes))
	labels_dense_transpose = labels_dense.T
	b[np.arange(num_labels), labels_dense_transpose] = 1
	return b


def batch_generator(batch_size, dataset_length, dataset, isTrain):
	batch_mask = rng.choice(dataset_length, batch_size)
	sampled_ds = dataset[[batch_mask]]
	batch_x = sampled_ds[:,:-1]
	
	if(isTrain):
		batch_y = sampled_ds[:,-1:]
		batch_y = dense_to_one_hot(batch_y,3)

	return batch_x,batch_y

total_dataset = arff.load('PhishingData.arff')
total_dataset = np.stack(total_dataset)
total_dataset = total_dataset.astype(np.float32) 
split_size = int(total_dataset.shape[0]*0.7)
train_dataset, val_dataset = total_dataset[:split_size], total_dataset[split_size:]

val_dataset_x = val_dataset[:,:-1]
val_dataset_y = val_dataset[:,-1:]

input_num_units = 9;
output_num_units = 3;
hidden_num_units = 250;

x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

epochs = 5
batch_size = 128
learning_rate = 0.01

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(epochs):
		avg_cost = 0
		total_batch = int(train_dataset.shape[0]/batch_size)
		for i in range(total_batch):
			batch_x, batch_y = batch_generator(batch_size, train_dataset.shape[0],train_dataset, "true")
			#doubt in true
			_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

			avg_cost += c / total_batch
            
		print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost) )
    
	print ("\nTraining complete!" )
    
    
	# find predictions on val set
	pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
	# print "Validation Accuracy:", accuracy.eval({x: val_dataset_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_dataset_y)})
	print ("Validation Accuracy:", accuracy.eval({x: val_dataset_x, y: dense_to_one_hot(val_dataset_y,3)}) )

    # predict = tf.argmax(output_layer, 1)
    # pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
