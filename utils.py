
drop_rate = 0.2
drop_rate = 0.2
num_classes = 2
input_shape = (100, 32)
image_size = 100  # We'll resize input images to this size
patch_size = 5  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size)
projection_dim = 100
num_heads = 2
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 2

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        #x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

from tensorflow.keras import layers
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded
    
def convF1(inpt, D1, fil_ord, Dr):

    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #filters = inpt._keras_shape[channel_axis]
    filters = int(inpt.shape[-1])
    
    #infx = Activation('relu'')(inpt)
    pre = Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inpt)
    pre = BatchNormalization()(pre)    
    pre = Activation('relu')(pre)
    
    #shared_conv = Conv1D(D1,  fil_ord, strides =(1), padding='same')
    
    inf  = Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(pre)
    inf = BatchNormalization()(inf)    
    inf = Activation('relu')(inf)
    inf = Add()([inf,inpt])
    
    inf1  = Conv1D(D1,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inf)
    inf1 = BatchNormalization()(inf1)  
    inf1 = Activation('relu')(inf1)    
    encode = Dropout(Dr)(inf1)

    return encode


