import os
import tensorflow as tf
import numpy as np
import cv2

def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    final_counter_sax=0
    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images): # For every possible positive pair.
                final_counter_sax+=1
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                # all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1


        emb_start_idx += nrof_images

    np.random.shuffle(triplets)

    return triplets, num_trips, len(triplets)

def load_data(image_paths):
    no_of_images = len(image_paths)
    images = []
    for i in range(no_of_images):
        current_image = cv2.imread(image_paths[i])
        # current_image = cv2.resize(current_file,(450,300))
        current_image = current_image/255.
        images.append(current_image)
    images = np.asarray(images)
    return images

def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1

    return image_paths, num_per_class

def combine_embeddings(x):
    anchor = tf.expand_dims(x[0],axis=1)
    positive = tf.expand_dims(x[1],axis=1)
    negative = tf.expand_dims(x[2],axis=1)

    return(tf.concat([anchor,positive,negative],axis=1))

def triplet_loss(alpha):
    def loss_func(y_true,y_pred):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[:,0,:], y_pred[:,1,:])), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[:,0,:], y_pred[:,2,:])), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
        return loss
    return loss_func

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        classdir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(classdir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(classdir):
    image_paths = []
    if os.path.isdir(classdir):
        images = os.listdir(classdir)
        image_paths = [os.path.join(classdir,img) for img in images]
    return image_paths