from modules import *

class Stanford_Dogs():
    """
    A Classification class for specially designed for this case.
    """
    def __init__(self,path):

        """ 
        First of all we need to consider how many breed do we have,
        And we need to know how many images we have.(Using Os module)
        """
        self.path = path
        self.breed_list = os.listdir(f'{self.path}/images/Images/')
        self.breed_number = len(self.breed_list)
        self.breed_number = len(self.breed_list)
        self.total_images = 0
        for breed in self.breed_list:
            self.total_images +=len(os.listdir(f'{self.path}/images/Images/{breed}'))
        """
        Store label names and indices in dictionary for further usage.
        """
        self.label_maps = {}
        self.label_maps_rev = {}
        for i, v in enumerate(self.breed_list):
            self.label_maps.update({v: i})
            self.label_maps_rev.update({i : v})
        """
        Run-----
        """
        self.Run()
    
    def show_image(self,breed,N):
        """
        A method for displaying images for given N number of specific breed
        N must a full squared integer.
        breed is str.
        """
        try:
            plt.figure(figsize=(16,16))
            img_path = f'{self.path}/images/Images/{breed}'
            path_list = random.sample(os.listdir(img_path),N)
            for i in range(N):
                self.img = mpimg.imread(f'{img_path}/{path_list[i]}')
                plt.subplot(N/2,N/2,i+1)
                plt.imshow(self.img)
                plt.axis('off')
            plt.show()
        except:
            print('Enter a full squared number.')

    def crop_data(self):
        """
        Cropping the images and fitting them in same size,
        Control is a variable that make sures this process won't work more than one.
        """
        try:
            os.mkdir('updated_images')
            for breed in self.breed_list:
                os.mkdir(f'updated_images/{breed}')
            for breed in os.listdir('updated_images'):
                for image in os.listdir(f'{self.path}/annotations/Annotation/{breed}'):
                    img = Image.open(f'{self.path}/images/Images/{breed}/{image}.jpg')
                    tree = ET.parse(f'{self.path}/annotations/Annotation/{breed}/{image}')
                    xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
                    xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
                    ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
                    ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
                    img = img.crop((xmin, ymin, xmax, ymax))
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img.save('updated_images/' + breed + '/' + image + '.jpg')
        except:
            print('Cropping is already done.')

    def set_xy(self):
        """ 
        Identifying paths,labels,targets. 
        These will be used as an input-output for our model
        Also, target list converted into catergorical binary array which is important for our model.
        """
        self.paths = list()
        self.labels = list()
        self.targets = list()
        for breed in self.breed_list:
            base = f'updated_images/{breed}/'
            for image in os.listdir(base):
                self.paths.append(base+image)
                self.labels.append(breed)
                self.targets.append(self.label_maps[breed])
        self.targets = np_utils.to_categorical(self.targets, num_classes=self.breed_number)

    def split(self):
        #split data 
        train_paths, val_paths, train_targets, val_targets = train_test_split(self.paths, 
                                        self.targets,
                                        test_size=0.20, 
                                        random_state=1029)
        self.train_gen = Item_Generator(train_paths, train_targets, batch_size=32, shape=(224,224,3), augment=True, breed_number=self.breed_number)
        self.val_gen = Item_Generator(val_paths, val_targets, batch_size=32, shape=(224,224,3), augment=False, breed_number=self.breed_number)
        
    def train(self):
        #Network
        inp = Input((224, 224, 3))
        backbone = DenseNet121(input_tensor=inp,
                            weights=f'{self.path}/densenet/DenseNet-BC-121-32-no-top.h5',
                            include_top=False)
        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        outp = Dense(self.breed_number, activation="softmax")(x)
        model = Model(inp, outp)
        #####
        #Selecting trainable layers.
        for layer in model.layers[:-6] :
            layer.trainable = False
        #####
        # a check point callback to save our best weights
        checkpoint = ModelCheckpoint('dog_breed_classifier_model.h5', 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='max', 
                                    save_weights_only=True)

        # a reducing lr callback to reduce lr when val_loss doesn't increase
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=1, verbose=1, mode='min',
                                        min_delta=0.0001, cooldown=2, min_lr=1e-7)

        # for early stop
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        #####
        model.compile(optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["acc"])
        self.history = model.fit_generator(generator=self.train_gen, 
                              steps_per_epoch=len(self.train_gen), 
                              validation_data=self.val_gen, 
                              validation_steps=len(self.val_gen),
                              epochs=20,
                              callbacks=[checkpoint, reduce_lr, early_stop])
        model.save('Model')
    def plot(self):
        plt.rcParams['figure.figsize'] = (6,6)
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.title('Training and validation accuracy')
        plt.plot(epochs, acc, 'red', label='Training acc')
        plt.plot(epochs, val_acc, 'blue', label='Validation acc')
        plt.legend()
        plt.figure()
        plt.title('Training and validation loss')
        plt.plot(epochs, loss, 'red', label='Training loss')
        plt.plot(epochs, val_loss, 'blue', label='Validation loss')
        plt.legend()
        plt.show()
        plt.savefig('Training and Validation Graph')

    def Run(self):
        self.crop_data()
        self.set_xy()
        self.split()
        self.train()
        self.plot()

#For customized data generator we have to inherit the class from Sequence class.
class Item_Generator(Sequence):
    """
    Slicing the images for given batch size,
    It is important because there are relatively too many images
    """
    def __init__(self,paths,targets,batch_size,shape,augment,breed_number):
        #Initialization. Augment is used for replaceing the data in each epoch.
        #This type of customization needs to define two methods called __len__ and __geitem__
        self.paths = paths
        self.targets = targets
        self.batch_size = batch_size
        self.shape = shape
        self.augment = augment
        self.breed_number = breed_number
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
    def __getitem__(self,idx):
        """
        pass the x and y
        """
        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        y = np.zeros((self.batch_size, self.breed_number, 1))
        for i, path in enumerate(batch_paths):
            x[i] = self.load_image(path)
        y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]
        return x, y
    def load_image(self,path):
        """
        load images, if augmentation true images will be replaced.
        """
        image = imread(path)
        image = preprocess_input(image)
        if self.augment:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.CropAndPad(percent=(-0.25, 0.25)),
                    iaa.Crop(percent=(0, 0.1)),
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])
            ], random_order=True)
            image = seq.augment_image(image)
        return image

path = os.path.dirname(os.path.abspath(__file__))
Test = Stanford_Dogs(path)
Test.label_maps_rev

