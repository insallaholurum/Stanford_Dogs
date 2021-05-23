from modules import *
"""
I had to create label_maps_rev because of poor coding :(
"""
path = os.path.dirname(os.path.abspath(__file__))
breed_list = os.listdir(f'{path}/images/Images/')
label_maps_rev = {}
for i, v in enumerate(breed_list):
    label_maps_rev.update({i : v})
"""
Importing the model.
"""
model = load_model('Model')
model.load_weights('dog_breed_classifier_model.h5')
model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["acc"])
print('model imported')
#predicting the new images by downloading.
def download_and_predict(path,filename):
    # download and save
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img.save(filename)
    # show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    # predict
    img = imread(filename)
    img = preprocess_input(img)
    probs = model.predict(np.expand_dims(img, axis=0))
    for idx in probs.argsort()[0][::-1][:]:
        print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])
#Run
while True:
    path , filename = input().split()
    download_and_predict(path,filename)