import tensorflow as tf
import numpy as np
from skimage.io  import imread, imsave
from skimage.transform import resize


#-----------------------Basic operatons block-----------------------------------

PATH2WEIGTHS='./vgg19.npy'
weigths=np.load(PATH2WEIGTHS,encoding='latin1').item()
#print(weigths)

        #el=np.asarray(el).astype(np.float64)
#print((np.asarray(weigths['conv1_1'])).shape)
WIDTH=2048
HEIGTH=2048

print(weigths['conv1_1'][0].dtype)


def conv_layer(x,weigths_w,weigths_b,trainable_w=False,trainable_b=False):

        if x.dtype == 'float64':
            x = tf.cast(x, 'float32')

        weights = tf.Variable(weigths_w.astype(np.float32), name="weights", trainable=trainable_w)
        biases = tf.Variable(weigths_b.astype(np.float32), name="biases", trainable=trainable_b)

        x = tf.nn.conv2d(x, weights,strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x,biases)
        x = tf.nn.relu(x)
        return x

def add_relu_layer(features,name=None):

    return tf.nn.relu(features)


def average_pooling(data,ksize,strides,padding='SAME',name=None):

    return tf.nn.avg_pool(data,ksize,strides,padding,name=name)



def add_dropout(data,rate=0.5,name=None):

    return tf.nn.dropout(data,rate=rate,name=name)


def max_pool(x,ksize,strides,padding,name=None):

        return tf.nn.max_pool(
            x,
            ksize=ksize,
            strides=strides,
            padding=padding,
            name=name)

MEAN_VALUES=np.array([123.68,116.779,103.939]).reshape((1,1,1,3))

def normalize(image):

    image=image*255.0


    return image-MEAN_VALUES

class Vgg19(object):

    def __init__(self):

        self.channel_mean=np.array([103.939,116.779,123.68])



    #return layers -- list of layer names
    def compute(self,data):

        d={}
        print("ok")

        d['conv1_1'] = conv_layer(data,weigths["conv1_1"][0],weigths["conv1_1"][1])
        d['conv1_2'] = conv_layer(d['conv1_1'],weigths["conv1_2"][0],weigths["conv1_2"][1])
        d['pool1'] = average_pooling(d['conv1_2'],[1,2,2,1],[1,2,2,1],name='pool1')

        d['conv2_1'] = conv_layer(d['pool1'],weigths["conv2_1"][0],weigths['conv2_1'][1])
        d['conv2_2'] = conv_layer(d['conv2_1'],weigths["conv2_2"][0],weigths['conv2_2'][1])
        d['pool2'] = average_pooling(d['conv2_2'],[1,2,2,1],[1,2,2,1],name='pool2')


        d['conv3_1'] = conv_layer(d['pool2'],weigths["conv3_1"][0],weigths['conv3_1'][1])
        d['conv3_2'] = conv_layer(d['conv3_1'],weigths["conv3_2"][0],weigths['conv3_2'][1])
        d['conv3_3'] = conv_layer(d['conv3_2'],weigths["conv3_3"][0],weigths['conv3_3'][1])
        d['conv3_4'] = conv_layer(d['conv3_3'],weigths["conv3_4"][0],weigths['conv3_4'][1])
        d['pool3'] = average_pooling(d['conv3_4'],[1,2,2,1],[1,2,2,1],name='pool3')

        d['conv4_1'] = conv_layer(d['pool3'],weigths["conv4_1"][0],weigths['conv4_1'][1])
        d['conv4_2'] = conv_layer(d['conv4_1'],weigths["conv4_2"][0],weigths['conv4_2'][1])
        d['conv4_3'] = conv_layer(d['conv4_2'],weigths["conv4_3"][0],weigths['conv4_3'][1])
        d['conv4_4'] = conv_layer(d['conv4_3'],weigths["conv4_4"][0],weigths['conv4_4'][1])
        d['pool4'] = average_pooling(d['conv4_4'],[1,2,2,1],[1,2,2,1],name='pool4')


        d['conv5_1'] = conv_layer(d['pool4'],weigths["conv5_1"][0],weigths['conv5_1'][1])
        d['conv5_2'] = conv_layer(d['conv5_1'],weigths["conv5_2"][0],weigths['conv5_2'][1])
        d['conv5_3'] = conv_layer(d['conv5_2'],weigths["conv5_3"][0],weigths['conv5_3'][1])
        d['conv5_4'] = conv_layer(d['conv5_3'],weigths["conv5_4"][0],weigths['conv5_4'][1])
        d['pool5'] = average_pooling(d['conv5_4'],[1,2,2,1],[1,2,2,1],name='pool5')

        return d




def cut_imgs(img1,img2):

    print(type(img1),type(img2))
    if ((img1.shape[0]!=img2.shape[0]) or (img1.shape[1]!=img2.shape[2])):
        img1=resize(img1,  (min(img1.shape[0],img2.shape[0],HEIGTH),\
                            min(img1.shape[1],img2.shape[1],WIDTH)),mode='constant')

        img2=resize( img2,  (min(img1.shape[0],img2.shape[0],HEIGTH),\
                            min(img1.shape[1],img2.shape[1],WIDTH)),mode='constant')


    return normalize(np.expand_dims(img1.astype(np.float32),axis=0)),\
            normalize(np.expand_dims(img2.astype(np.float32),axis=0))



def compute_gram_matrix(F,M,N):

    print('----------')
    print(F.shape)
    print('----------')
    Ft=tf.reshape(F,[M,N])

    return tf.matmul(tf.transpose(Ft),Ft)


def generate_noise_image(content_image, noise_ratio = 0.6):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(
            -40, 40,
            content_image.shape).astype(np.float32)
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image



weigthsL=[0.5,1.0,1.5,2.0,2.5]
layers_names=['conv1_2','conv2_2','conv3_2','conv4_2','conv5_2']




class StyleTransferer(object):



    def __init__(self,path2content,path2style,layer_name,alpha=300,beta=5):

        self.__model=Vgg19()
        self.alpha=alpha
        self.beta=beta
        self.layer_name=layer_name
        self.content,self.style=cut_imgs(imread(path2content),imread(path2style))
        #print(":::::AAAAA",self.content.shape,self.style.shape)
        print(self.content.dtype,self.style.dtype)
        imsave("a.jpg",np.clip(self.content[0].astype("uint8"),0,255))
        imsave("b.jpg",np.clip(self.style[0],0,255).astype("uint8"))
        imsave("c.jpg",np.clip(generate_noise_image(self.content)[0].astype("uint8"),0,255))
        self.res= tf.Variable(
                        generate_noise_image(self.content),
                        dtype=tf.float32,
                        trainable=True)


        self.style_features=self.__model.compute(self.style)
        #print('!!!!!!',self.style_features.dtype)
        self.content_features=self.__model.compute(self.content)
        self.res_features=None;


    def content_loss(self,P):

        N=P.shape[3]
        M=P.shape[1]*P.shape[2]
        x=self.__model.compute(self.res)[self.layer_name]
        return tf.reduce_sum(tf.pow(P-x,2))/(int(N)*int(M)*4)


    def style_loss(self):
        loss=0
        self.res_features=self.__model.compute(self.res)
        for w_idx,name in enumerate(layers_names):
            A=self.res_features[name]
            N=A.shape[-1]
            M=A.shape[1]*A.shape[2]
            A=compute_gram_matrix(A,M,N)
            G=compute_gram_matrix(self.style_features[name],M,N)
            loss+=weigthsL[w_idx]*tf.reduce_sum(tf.pow(G-A,2))* (1 / (4 * int(N)**2 * int(M)**2))

        return loss

    def setup_optimizer(self,loss,res,learning_rate=10):

        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=res)


    def compute_loss(self):

        return self.alpha*self.style_loss()+\
            self.beta*self.content_loss(self.content_features[self.layer_name])

    def optimize(self,with_regularizer=False):

        n_steps=1000
        self.loss=self.compute_loss()

        l=3.0
        learning_rate = tf.placeholder_with_default(l,None,name='lr')
        self.opt=self.setup_optimizer(self.loss,self.res,learning_rate)
        print(self.opt)
        with tf.Session(graph=tf.Graph()) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_steps):
                print(i)
                sess.run(self.opt)
                if (i%30==0):
                    print(sess.run(self.loss))
                #print("loss:",l)
            self.res=sess.run(self.res)
            
        return self.res

'''
░░░░░░░░░░░░░░░░░░░░
░░░░░ЗАПУСКАЕМ░░░░░░░
░ГУСЯ░▄▀▀▀▄░░░░░░░░░
▄███▀░◐░░░▌░░░░░░░░░
░░░░▌░░░░░▐░░░░░░░░░
░░░░▐░░░░░▐░░░░░░░░░
░░░░▌░░░░░▐▄▄░░░░░░░
░░░░▌░░░░▄▀▒▒▀▀▀▀▄
░░░▐░░░░▐▒▒▒▒▒▒▒▒▀▀▄
░░░▐░░░░▐▄▒▒▒▒▒▒▒▒▒▒▀▄
░░░░▀▄░░░░▀▄▒▒▒▒▒▒▒▒▒▒▀▄
░░░░░░▀▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▀▄
░░░░░░░░░░░▌▌░▌▌░░░░░
░░░░░░░░░░░▌▌░▌▌░░░░░
░░░░░░░░░▄▄▌▌▄▌▌░░░░░
'''
