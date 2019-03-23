from Tkinter import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data,n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_classes):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x,n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_classes,batch_size,hm_epochs):
    prediction = neural_network_model(x,n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_classes)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


class MyWindow:
    def __init__(self, win):
        self.heading = Label(win, text="Multilayer Perceptron NN", bg="light green") 
        self.heading.place(x=200,y=10)
        self.lbl1=Label(win, text='No of node in layer1')
        self.lbl2=Label(win, text='No of node in layer2')
        self.lbl3=Label(win, text='No of node in layer3')
        self.t1=Entry(bd=3)
        self.t2=Entry()
        self.t3=Entry()
        self.btn1 = Button(win, text='Train Data')
        self.btn2=Button(win, text='Subtract')
        self.lbl1.place(x=100, y=50)
        self.t1.place(x=300, y=50)
        self.lbl2.place(x=100, y=100)
        self.t2.place(x=300, y=100)
        self.b1=Button(win, text='Start Training model', command=self.add)
        #self.b2=Button(win, text='Subtract')
        #self.b2.bind('<Button-1>', self.sub)
        self.b1.place(x=100, y=300)
        #self.b2.place(x=200, y=150)
        self.lbl3.place(x=100, y=150)
        self.t3.place(x=300, y=150)
        self.lbl4=Label(win, text='Batch Size')
        self.lbl4.place(x=100, y=200)
        self.t4=Entry()
        self.t4.place(x=300, y=200)
        self.lbl5=Label(win, text='epochs')
        self.lbl5.place(x=100, y=250)
        self.t5=Entry()
        self.t5.place(x=300, y=250)
    def add(self):
        num1=int(self.t1.get())
        num2=int(self.t2.get())
        num3=int(self.t3.get())
        num4=int(self.t4.get())
        epochs=int(self.t5.get())
        train_neural_network(x,num1,num2,num3,10,num4,epochs)
        #self.t3.insert(END, str(result))
    def sub(self, event):
        num1=int(self.t1.get())
        num2=int(self.t2.get())
        result=num1-num2
        self.t3.insert(END, str(result))

window=Tk()
mywin=MyWindow(window)
window.title('Hello Python')
window.geometry("500x400+10+10")
window.mainloop()